"""
Docker-pinned Mermaid render step.

    python -m tidepool_data_science_simulator.diagramgen.render

``generate()`` stops at the ``.mmd`` boundary: it produces deterministic figure
*source* with recorded provenance, and says nothing about the artifact that is
actually submitted. This module carries determinism and provenance across the
render boundary, so a PNG pasted into the TRSET position paper can be
reproduced years later.

It is a *consumer of the committed figures*, exactly as ``diagramgen`` is a
consumer of the simulator. It never regenerates a ``.mmd``, never imports the
tracer, and is not part of the default generation path -- a host with no Docker
can still run the generator.

Why Docker rather than a conda dependency
-----------------------------------------

``conda-environment.yml`` already installs two packages from absolute local
editable paths and is documented as not runner-reproducible; adding Node,
Puppeteer and a ~280 MB browser to it makes that worse. A digest-pinned image
instead fixes four version strings and, more importantly, **the font set** --
the largest uncontrolled variable in Mermaid layout -- behind one hash, and
confines the renderer to a single ``/data`` mount.

Pinning
-------

The image is referenced by digest, never by tag. A tag-only reference is
*rejected*, not accepted with a warning: the pin is the whole point, and it must
not be losable by editing one string. Note that the upstream tag and the
packaged version genuinely disagree (tag ``11.17.1`` ships
``mermaid-cli-11.17.0.tgz``), which is why every version recorded in the
manifest is read out of the running container rather than restated from config.

Browser sandbox
---------------

The published image's ``ENTRYPOINT`` is ``mmdc -p /puppeteer-config.json``, and
that baked-in file sets ``--no-sandbox``. This render step does **not** override
it. Chromium's own sandbox needs user namespaces that Docker denies by default,
which is why upstream ships it this way; overriding the entrypoint to remove the
flag makes the render fail rather than making it safer. The isolation boundary
here is therefore the container and its single mount, not the browser sandbox,
and the manifest records that fact explicitly rather than leaving a reader to
assume otherwise. ``--network none`` is added on top, since neither figure
fetches anything.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree

from tidepool_data_science_simulator.diagramgen import cli as diagram_cli
from tidepool_data_science_simulator.diagramgen.manifest import MANIFEST_SCHEMA, build_render_block

__all__ = ["RenderError", "RenderConfig", "load_render_config", "render", "main"]

RENDER_CONFIG_FILENAME = "render.json"

# One render of the data-flow figure is seconds, not minutes, but a cold
# `docker pull` of a ~1 GB image on the first run is not.
_RENDER_TIMEOUT_SECONDS = 900
_PROBE_TIMEOUT_SECONDS = 300

# `<anything>@sha256:<64 hex>`. `repo:tag@sha256:...` is still pinned and is
# accepted; `repo:tag` and `repo:latest` are not.
_PINNED_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")

_SUPPORTED_FORMATS = ("png", "svg")

# mmdc's --width is a *viewport cap*, not a target: the output is the diagram's
# natural layout size, clamped to the viewport, multiplied by --scale. Measured
# against the pinned image: -w 1800 and -w 2700 both yield 1418 px for a figure
# whose natural width is 1418. So the measuring pass renders behind a viewport
# far wider than any figure, to read the natural size without clamping it.
_MEASURE_VIEWPORT_PX = 20000

# Chrome rounds the scaled raster, so the final width can miss the target by a
# pixel. More than this means the scale did not take effect and the figure is
# not the size the config asked for.
_WIDTH_TOLERANCE_PX = 2

# The reproducibility level the test suite asserts at. `bytes` may only be set
# once byte-stability has actually been measured on a Docker host -- see the
# README. `dimensions` is the claim that is safe without that measurement.
_REPRODUCIBILITY_LEVELS = ("dimensions", "bytes")

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Version probe. Every field degrades to an empty line independently: a missing
# version is recorded as unresolved, never raised, and never silently dropped.
# Package versions are read by reading `package.json` off disk rather than by
# `require('<name>/package.json')`, which Node 22 blocks for any package whose
# `exports` map does not list it.
_PROBE_SCRIPT = r"""
ROOTS='/home/mermaidcli/node_modules /home/mermaidcli/node_modules/@mermaid-js/mermaid-cli/node_modules'
pkg_version() {
    node -e '
const fs = require("fs");
const roots = process.argv[2].split(" ");
for (const root of roots) {
    try {
        const manifestPath = root + "/" + process.argv[1] + "/package.json";
        process.stdout.write(JSON.parse(fs.readFileSync(manifestPath, "utf8")).version);
        break;
    } catch (error) { /* try the next root */ }
}
' "$1" "$ROOTS" 2>/dev/null
}
printf 'mermaid_cli_version\t%s\n' "$(pkg_version @mermaid-js/mermaid-cli)"
printf 'mermaid_version\t%s\n' "$(pkg_version mermaid)"
printf 'puppeteer_version\t%s\n' "$(pkg_version puppeteer)"
printf 'puppeteer_core_version\t%s\n' "$(pkg_version puppeteer-core)"
printf 'browser\t%s\n' "$(/usr/bin/chromium-browser --version 2>/dev/null | head -1)"
printf 'node_version\t%s\n' "$(node --version 2>/dev/null)"
printf 'base_image\t%s\n' "$(. /etc/os-release 2>/dev/null; printf '%s' "$PRETTY_NAME")"
printf 'entrypoint_puppeteer_config\t%s\n' "$(tr -d ' \n' < /puppeteer-config.json 2>/dev/null)"
printf '\037fonts\037\n'
fc-list -f '%{family}\n' 2>/dev/null | tr ',' '\n' | sed 's/^ *//; s/ *$//' | sort -u
"""

_FONT_SENTINEL = "\x1ffonts\x1f"


class RenderError(RuntimeError):
    """Raised when the render step cannot produce a trustworthy artifact.

    Every path that raises this leaves the output directory exactly as it found
    it. A half-written or stale figure in ``.docs/architecture/`` is worse than
    no figure, because nothing downstream would notice.
    """


def _sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class Output(object):
    """One emitted file.

    ``target_width_px`` is the pixel width the PNG must come out at, and is
    ``None`` for an SVG -- mermaid-cli emits a responsive root
    (``width="100%"`` plus a ``viewBox``), so a vector has no pixel width to
    set and asking for one would be meaningless.
    """

    __slots__ = ("format", "target_width_px", "background")

    def __init__(self, fmt, target_width_px, background):
        self.format = fmt
        self.target_width_px = target_width_px
        self.background = background


class Figure(object):
    """One committed ``.mmd`` and the outputs rendered from it."""

    __slots__ = ("source", "outputs")

    def __init__(self, source, outputs):
        self.source = source
        self.outputs = tuple(outputs)

    def output_name(self, output):
        stem = os.path.splitext(self.source)[0]
        return "{}.{}".format(stem, output.format)


class RenderConfig(object):
    """The committed render settings, validated."""

    __slots__ = (
        "image",
        "figures",
        "mermaid_configs",
        "reproducibility_level",
        "source_path",
        "sha256",
        "font_family",
    )

    def __init__(self, image, figures, mermaid_configs, reproducibility_level, source_path, sha256, font_family):
        self.image = image
        self.figures = tuple(figures)
        self.mermaid_configs = dict(mermaid_configs)
        self.reproducibility_level = reproducibility_level
        self.source_path = source_path
        self.sha256 = sha256
        self.font_family = font_family

    @property
    def config_paths(self):
        """Every committed config file whose SHA-256 the manifest records."""
        return [self.source_path] + [self.mermaid_configs[fmt] for fmt in sorted(self.mermaid_configs)]


def _require(data, key, kind, path):
    if key not in data:
        raise RenderError("{}: required key {!r} is missing".format(path, key))
    value = data[key]
    if not isinstance(value, kind):
        raise RenderError(
            "{}: key {!r} must be {}, got {}".format(path, key, kind.__name__, type(value).__name__)
        )
    return value


def validate_image_reference(image, source_path):
    """Reject anything that is not digest-pinned.

    A floating tag would make the manifest's record of "what rendered this"
    a statement about a name rather than about an artifact, which is the exact
    failure this step exists to close. Rejecting is the only behavior that keeps
    the pin un-losable.
    """
    if not _PINNED_IMAGE.match(image):
        raise RenderError(
            "{}: image reference is not digest-pinned: {!r}.\n"
            "It must read '<repository>@sha256:<64 hex digits>'. A tag such as ':latest' or "
            "':11.17.1' does not pin the renderer, the browser or the font set, and the "
            "manifest would record a name rather than an artifact.".format(source_path, image)
        )
    return image


def _load_mermaid_config(path):
    if not os.path.isfile(path):
        raise RenderError("Mermaid config not found: {}".format(path))
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError as error:
        raise RenderError("{}: not valid JSON: {}".format(path, error))


def _font_family_of(mermaid_config, path):
    variables = mermaid_config.get("themeVariables")
    if not isinstance(variables, dict) or not variables.get("fontFamily"):
        raise RenderError(
            "{}: themeVariables.fontFamily is missing. The font is part of the pinned "
            "toolchain: leaving it unset lets Mermaid resolve it through whatever fonts the "
            "host happens to have, which is the reproducibility gap this step closes.".format(path)
        )
    return variables["fontFamily"]


def load_render_config(path):
    """Read, validate and cross-check the committed render settings."""
    if not os.path.isfile(path):
        raise RenderError("Render config not found: {}".format(path))
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as error:
        raise RenderError("{}: not valid JSON: {}".format(path, error))
    if not isinstance(data, dict):
        raise RenderError("{}: expected a JSON object at the top level".format(path))

    image = validate_image_reference(_require(data, "image", str, path), path)

    level = _require(data, "reproducibility_level", str, path)
    if level not in _REPRODUCIBILITY_LEVELS:
        raise RenderError(
            "{}: reproducibility_level must be one of {}, got {!r}".format(
                path, ", ".join(_REPRODUCIBILITY_LEVELS), level
            )
        )

    config_dir = os.path.dirname(os.path.abspath(path))

    raw_configs = _require(data, "mermaid_configs", dict, path)
    mermaid_configs = {}
    font_families = {}
    for fmt in _SUPPORTED_FORMATS:
        if fmt not in raw_configs:
            raise RenderError("{}: mermaid_configs is missing an entry for {!r}".format(path, fmt))
        config_path = os.path.join(config_dir, raw_configs[fmt])
        loaded = _load_mermaid_config(config_path)
        mermaid_configs[fmt] = config_path
        font_families[fmt] = _font_family_of(loaded, config_path)

    # One font for the whole toolchain, or the manifest's single `font_family`
    # would be a half-truth.
    if len(set(font_families.values())) != 1:
        raise RenderError(
            "{}: the Mermaid configs name different fonts ({}). The manifest records one "
            "font_family for the run, so they must agree.".format(
                path, ", ".join("{}={!r}".format(k, v) for k, v in sorted(font_families.items()))
            )
        )

    figures = []
    raw_figures = _require(data, "figures", list, path)
    if not raw_figures:
        raise RenderError("{}: key 'figures' must not be empty".format(path))
    for entry in raw_figures:
        if not isinstance(entry, dict):
            raise RenderError("{}: each figure must be a JSON object".format(path))
        source = _require(entry, "source", str, path)
        raw_outputs = _require(entry, "outputs", list, path)
        if not raw_outputs:
            raise RenderError("{}: figure {!r} has no outputs".format(path, source))
        outputs = []
        for raw in raw_outputs:
            if not isinstance(raw, dict):
                raise RenderError("{}: each output of {!r} must be a JSON object".format(path, source))
            fmt = _require(raw, "format", str, path)
            if fmt not in _SUPPORTED_FORMATS:
                raise RenderError(
                    "{}: figure {!r} names an unsupported format {!r}; expected one of {}".format(
                        path, source, fmt, ", ".join(_SUPPORTED_FORMATS)
                    )
                )
            if fmt == "svg":
                if "width_px" in raw:
                    raise RenderError(
                        "{}: figure {!r} sets width_px on its svg output. mermaid-cli emits a "
                        "responsive root (width=\"100%\" plus a viewBox), so a vector has no "
                        "pixel width to pin; the intrinsic viewBox size is recorded "
                        "instead.".format(path, source)
                    )
                width = None
            else:
                width = _require(raw, "width_px", int, path)
                if isinstance(width, bool) or width <= 0:
                    raise RenderError(
                        "{}: figure {!r} output {!r} needs a positive integer width_px. The width "
                        "is a target the render step hits by measuring each figure and deriving a "
                        "scale for it, never a fixed scale factor from config -- a fixed factor "
                        "would give a different width, and a different effective DPI, every time "
                        "the traced graph changed.".format(path, source, fmt)
                    )
            outputs.append(Output(fmt, width, _require(raw, "background", str, path)))
        figures.append(Figure(source, outputs))

    return RenderConfig(
        image=image,
        figures=figures,
        mermaid_configs=mermaid_configs,
        reproducibility_level=level,
        source_path=os.path.abspath(path),
        sha256=_sha256_file(path),
        font_family=sorted(set(font_families.values()))[0],
    )


def _run_docker(docker, args, timeout):
    """Run one ``docker`` command, returning ``(returncode, stdout, stderr)``.

    A missing ``docker`` is a distinct, expected condition rather than a crash:
    the caller turns it into the documented "install Docker Desktop" message.
    """
    try:
        completed = subprocess.run(
            [docker] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RenderError(
            "{!r} was not found on PATH. The render step runs the pinned mermaid-cli image and "
            "has no host fallback -- a host `mmdc` would silently defeat the pin this step "
            "exists to provide. Install Docker Desktop, or pass --docker with a path to a "
            "compatible client.".format(docker)
        )
    except subprocess.TimeoutExpired:
        raise RenderError("`{} {}` timed out after {}s.".format(docker, " ".join(args[:2]), timeout))
    except OSError as error:
        raise RenderError("`{}` could not be executed: {}".format(docker, error))
    return (
        completed.returncode,
        completed.stdout.decode("utf-8", "replace"),
        completed.stderr.decode("utf-8", "replace"),
    )


def _docker_run_args(image, image_args, mount=None, entrypoint=None):
    args = ["run", "--rm", "--network", "none"]
    if mount is not None:
        args += ["--volume", "{}:/data".format(mount)]
    if entrypoint is not None:
        args += ["--entrypoint", entrypoint]
    return args + [image] + list(image_args)


def probe_toolchain(docker, image):
    """Read the renderer's identity out of the running container.

    Returns ``(facts, font_families, probe_status)``. Every value is read from
    inside the pinned image rather than restated from config, because the tag
    and the packaged version genuinely disagree upstream. Follows
    ``manifest.package_provenance()``: a failing probe is recorded as a reason,
    never raised -- a figure that rendered correctly should not be discarded
    because a version string could not be read.
    """
    returncode, stdout, stderr = _run_docker(
        docker,
        _docker_run_args(image, ["-c", _PROBE_SCRIPT], entrypoint="/bin/sh"),
        _PROBE_TIMEOUT_SECONDS,
    )
    if returncode != 0:
        reason = stderr.strip() or "docker exited {}".format(returncode)
        return {}, set(), {"attempted": True, "reason": reason}

    head, _, tail = stdout.partition(_FONT_SENTINEL)
    facts = {}
    for line in head.splitlines():
        if "\t" not in line:
            continue
        key, _, value = line.partition("\t")
        facts[key.strip()] = value.strip()
    fonts = {line.strip() for line in tail.splitlines() if line.strip()}
    return facts, fonts, {"attempted": True, "reason": None}


def _verify_font(font_family, available, probe_status, config_path):
    """Fail loudly if the pinned font is not in the image.

    Silent fallback to whatever fontconfig picks is the precise failure mode the
    font pin exists to prevent, and it is invisible in the output: the figure
    still renders, just with different metrics. So an unavailable family is an
    error. An *unreadable font list* is different -- that is the probe failing,
    not the font being absent, and it degrades to a recorded reason.
    """
    if not available:
        return "font list could not be read from the image ({})".format(
            probe_status.get("reason") or "fc-list returned nothing"
        )
    if font_family in available:
        return None
    raise RenderError(
        "{}: fontFamily {!r} is not present in the pinned image.\n"
        "Fontconfig would fall back silently to another family and the figure would still "
        "render, with different metrics and no indication anything had changed. Pick one of "
        "the families the image actually ships:\n  {}".format(
            config_path, font_family, "\n  ".join(sorted(available))
        )
    )


def _png_dimensions(path):
    """Pixel dimensions from the PNG IHDR chunk. No Pillow dependency needed."""
    with open(path, "rb") as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != _PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise RenderError(
            "{} is not a valid PNG. mmdc exited successfully but did not write the raster "
            "this step promised.".format(os.path.basename(path))
        )
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


def _svg_dimensions(path):
    """Pixel dimensions from an SVG root element, or ``(None, None)``.

    Unlike the PNG case these are advisory: an SVG is resolution-independent, so
    a missing or unit-bearing width is recorded as unresolved rather than
    treated as a render failure.
    """
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as error:
        raise RenderError("{} is not well-formed XML: {}".format(os.path.basename(path), error))

    def _as_pixels(value):
        if not value:
            return None
        match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:px)?\s*$", value)
        return int(round(float(match.group(1)))) if match else None

    width = _as_pixels(root.get("width"))
    height = _as_pixels(root.get("height"))
    if width is None or height is None:
        viewbox = (root.get("viewBox") or "").split()
        if len(viewbox) == 4:
            try:
                width = width if width is not None else int(round(float(viewbox[2])))
                height = height if height is not None else int(round(float(viewbox[3])))
            except ValueError:
                pass
    return width, height


def _dimensions(path, fmt):
    return _png_dimensions(path) if fmt == "png" else _svg_dimensions(path)


def _mmdc(docker, image, work, source, output_name, mermaid_config, background, scale=None):
    """One ``mmdc`` invocation inside the pinned container."""
    args = [
        "--input", "/data/{}".format(source),
        "--output", "/data/{}".format(output_name),
        "--configFile", "/data/{}".format(mermaid_config),
        # A viewport wider than any figure, so --width never clamps the layout.
        # It is a measuring frame, not the output size.
        "--width", str(_MEASURE_VIEWPORT_PX),
        "--backgroundColor", background,
    ]
    if scale is not None:
        args += ["--scale", "{:.6f}".format(scale)]

    returncode, stdout, stderr = _run_docker(
        docker, _docker_run_args(image, args, mount=work), _RENDER_TIMEOUT_SECONDS
    )
    produced = os.path.join(work, output_name)
    if returncode != 0:
        raise RenderError(
            "mmdc failed on {} -> {} (docker exited {}).\n{}".format(
                source, output_name, returncode, (stderr or stdout).strip()
            )
        )
    if not os.path.isfile(produced) or os.path.getsize(produced) == 0:
        raise RenderError("mmdc reported success but wrote no output for {}.".format(source))
    return produced


def _render_one(docker, config, figure, output, work):
    """Render one output and return what the manifest records about it.

    A PNG is rendered **twice**. mmdc has no "make it N pixels wide" flag:
    ``--width`` caps the viewport and ``--scale`` multiplies whatever the layout
    naturally came out as. A scale factor fixed in config would therefore give a
    different width, and a different effective DPI, every time the traced graph
    changed -- which is exactly the failure mode to avoid. So the first pass
    measures the figure's natural width behind a viewport wide enough not to
    clamp it, and the second renders at ``target / natural``. The configured
    pixel width is then hit whatever the architecture does, and the derived
    scale is recorded rather than chosen.

    An SVG is rendered once. mermaid-cli emits a responsive root, so there is no
    pixel width to hit; its intrinsic ``viewBox`` size is recorded instead.
    """
    name = figure.output_name(output)
    mermaid_config = os.path.basename(config.mermaid_configs[output.format])

    if output.format != "png":
        produced = _mmdc(
            docker, config.image, work, figure.source, name, mermaid_config, output.background
        )
        width, height = _svg_dimensions(produced)
        return {
            "source": figure.source,
            "output": name,
            "format": output.format,
            "target_width_px": None,
            "width_px": width,
            "height_px": height,
            "natural_width_px": width,
            "natural_height_px": height,
            "scale": None,
            "background": output.background,
            "sha256": _sha256_file(produced),
        }

    probe_name = "measure-{}".format(name)
    probe = _mmdc(
        docker, config.image, work, figure.source, probe_name, mermaid_config, output.background
    )
    natural_width, natural_height = _png_dimensions(probe)
    os.remove(probe)

    if natural_width >= _MEASURE_VIEWPORT_PX:
        raise RenderError(
            "{} laid out at least {} px wide, so the measuring viewport clamped it and its "
            "natural width cannot be read. Raise _MEASURE_VIEWPORT_PX.".format(
                figure.source, _MEASURE_VIEWPORT_PX
            )
        )

    scale = float(output.target_width_px) / float(natural_width)
    produced = _mmdc(
        docker, config.image, work, figure.source, name, mermaid_config, output.background, scale=scale
    )
    width, height = _png_dimensions(produced)
    if abs(width - output.target_width_px) > _WIDTH_TOLERANCE_PX:
        raise RenderError(
            "{} came out {} px wide against a target of {} px (natural {} px, scale {:.6f}). "
            "The scale did not take effect and the figure is not the size the config asked "
            "for.".format(name, width, output.target_width_px, natural_width, scale)
        )

    return {
        "source": figure.source,
        "output": name,
        "format": output.format,
        "target_width_px": output.target_width_px,
        "width_px": width,
        "height_px": height,
        "natural_width_px": natural_width,
        "natural_height_px": natural_height,
        "scale": round(scale, 6),
        "background": output.background,
        "sha256": _sha256_file(produced),
    }


def render(output_dir, render_config_path=None, docker="docker", repo_root=diagram_cli.REPO_ROOT):
    """Render every configured figure and record the toolchain in the manifest.

    Renders into a scratch directory and moves the results into ``output_dir``
    only once every output has been produced and validated, so a failure part
    way through leaves no partial or stale artifact behind. The container only
    ever sees that scratch directory: the trace, the manifest and the two
    maintained config files are never mounted into it.

    Returns the manifest dict as written.
    """
    output_dir = os.path.abspath(output_dir)
    if render_config_path is None:
        render_config_path = os.path.join(output_dir, RENDER_CONFIG_FILENAME)
    config = load_render_config(render_config_path)

    # Validate every input before touching Docker, so the common mistakes fail
    # in a second rather than after a 1 GB pull.
    manifest_path = os.path.join(output_dir, diagram_cli.MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise RenderError(
            "{} not found in {}. The render step records its provenance into the manifest that "
            "generation produced; run `python -m tidepool_data_science_simulator.diagramgen` "
            "first.".format(diagram_cli.MANIFEST_FILENAME, output_dir)
        )
    for figure in config.figures:
        source_path = os.path.join(output_dir, figure.source)
        if not os.path.isfile(source_path):
            raise RenderError(
                "Figure source not found: {}. The render step consumes the *committed* .mmd "
                "files and never regenerates them.".format(source_path)
            )

    facts, fonts, probe_status = probe_toolchain(docker, config.image)
    font_note = _verify_font(config.font_family, fonts, probe_status, config.mermaid_configs["png"])

    scratch = tempfile.mkdtemp(prefix="trset51-render-")
    try:
        work = os.path.join(scratch, "data")
        os.makedirs(work)

        # The container sees this directory and nothing else: the trace, the
        # manifest and the two maintained config files are never mounted.
        for figure in config.figures:
            shutil.copyfile(
                os.path.join(output_dir, figure.source), os.path.join(work, figure.source)
            )
        for fmt, config_path in config.mermaid_configs.items():
            shutil.copyfile(config_path, os.path.join(work, os.path.basename(config_path)))

        rendered = []
        for figure in config.figures:
            for output in figure.outputs:
                rendered.append(_render_one(docker, config, figure, output, work))

        # Everything rendered and validated: only now does anything enter the
        # committed directory.
        for entry in rendered:
            shutil.move(os.path.join(work, entry["output"]), os.path.join(output_dir, entry["output"]))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    render_block = build_render_block(
        repo_root=repo_root,
        image=config.image,
        facts=facts,
        probe_status=probe_status,
        font_family=config.font_family,
        font_note=font_note,
        config_paths=config.config_paths,
        figures=rendered,
        reproducibility_level=config.reproducibility_level,
    )

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["render"] = render_block
    # The render block is the only thing /2 adds, and it is purely additive, so a
    # /1 manifest plus this block *is* a valid /2 document. Declaring it keeps the
    # committed file self-consistent: a manifest that still claimed /1 while
    # carrying a block /1 does not define would be lying about its own shape to
    # every consumer that checks the schema before reading.
    manifest["schema"] = MANIFEST_SCHEMA

    temporary = manifest_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)

    return manifest


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m tidepool_data_science_simulator.diagramgen.render",
        description=(
            "Render the committed Mermaid figures with a digest-pinned mermaid-cli image and "
            "record the render toolchain in manifest.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=diagram_cli.DEFAULT_OUTPUT_DIR,
        help="Directory holding the committed .mmd files; the figures are written beside them.",
    )
    parser.add_argument(
        "--render-config",
        default=None,
        help="Render settings JSON (defaults to <output-dir>/{}).".format(RENDER_CONFIG_FILENAME),
    )
    parser.add_argument("--docker", default="docker", help="Docker client executable.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)

    try:
        manifest = render(
            output_dir=output_dir,
            render_config_path=args.render_config,
            docker=args.docker,
            repo_root=diagram_cli.REPO_ROOT,
        )
    except RenderError as error:
        sys.stderr.write("Render failed: {}\n".format(error))
        sys.stderr.write("Nothing was written to {}.\n".format(output_dir))
        return 1

    block = manifest["render"]
    sys.stdout.write(
        "Rendered {} figures with {}\n".format(len(block["figures"]), block["image"])
    )
    for entry in block["figures"]:
        sys.stdout.write(
            "  {:<24} {}x{} px  sha256:{}\n".format(
                entry["output"], entry["width_px"], entry["height_px"], entry["sha256"][:12]
            )
        )
    sys.stdout.write("  font: {}\n".format(block["font_family"]))
    if block.get("font_verification_error"):
        sys.stdout.write("  font verification: {}\n".format(block["font_verification_error"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
