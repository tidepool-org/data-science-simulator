"""
Tests for the Docker-pinned Mermaid render step.

Split in two, deliberately.

The **unit** tests here need nothing but the repository: they cover the pin
validation, the config cross-checks, the dimension readers, the manifest
degradation contract, and the two behaviors that have to hold on a host with no
Docker at all -- the render exits non-zero without writing anything, and the
rendered figures stay out of the drift gate.

The **integration** tests are marked ``docker`` and run the real pinned image
against real fixtures. Nothing about the render boundary is mocked, on the same
reasoning that kept the Swift boundary unmocked in TRSET-51: a mocked renderer
would test this module against a fiction, and the artifact being validated is
the one a regulatory submission carries.

When Docker is absent those tests **skip with a stated reason** rather than
passing quietly. A test that silently no-ops provides no evidence, which in a
validation record is worse than a failure. Run ``pytest -rs`` to see the skip
and its reason reported; run ``pytest -m docker`` to demand them.
"""

import json
import os
import shutil
import struct
import subprocess

import pytest

from tidepool_data_science_simulator.diagramgen import cli
from tidepool_data_science_simulator.diagramgen import manifest as diagram_manifest
from tidepool_data_science_simulator.diagramgen import render as diagram_render
from tidepool_data_science_simulator.diagramgen.render import RenderError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHITECTURE_DIR = os.path.join(REPO_ROOT, ".docs", "architecture")
COMMITTED_RENDER_CONFIG = os.path.join(ARCHITECTURE_DIR, diagram_render.RENDER_CONFIG_FILENAME)
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "test_data", "diagramgen")

FIXTURE_FLOWCHART = "render_flowchart.mmd"
FIXTURE_SEQUENCE = "render_sequence.mmd"

# Small enough that a fixture render is seconds. The committed figures use the
# print-targeted widths in render.json; what these tests assert is that the
# configured width is the width that comes out, not what the number should be.
FIXTURE_WIDTH_PX = 900


def _docker_reason():
    """Why the Docker tests cannot run here, or ``None`` if they can."""
    if shutil.which("docker") is None:
        return "no `docker` on PATH; install Docker Desktop to exercise the pinned render"
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return "`docker info` could not be run: {}: {}".format(type(error).__name__, error)
    if completed.returncode != 0:
        return "`docker info` failed, the daemon is not reachable: {}".format(
            completed.stderr.decode("utf-8", "replace").strip() or "exit {}".format(completed.returncode)
        )
    return None


_DOCKER_REASON = _docker_reason()

requires_docker = pytest.mark.skipif(
    _DOCKER_REASON is not None,
    reason="{} -- the render boundary is not mocked, so this test has no evidence to offer "
           "without it".format(_DOCKER_REASON),
)


def _committed(name):
    with open(os.path.join(ARCHITECTURE_DIR, name), encoding="utf-8") as handle:
        return json.load(handle)


def _prepare(output_dir, width_px=FIXTURE_WIDTH_PX, figures=None):
    """Assemble a directory the render step will accept.

    Fixture ``.mmd`` files, copies of the two *committed* Mermaid configs -- so
    the real pinned font is exercised -- and a minimal manifest standing in for
    one a generation run produced.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for name in (FIXTURE_FLOWCHART, FIXTURE_SEQUENCE):
        shutil.copyfile(os.path.join(FIXTURE_DIR, name), os.path.join(output_dir, name))
    for name in ("mermaid_config_png.json", "mermaid_config_svg.json"):
        shutil.copyfile(os.path.join(ARCHITECTURE_DIR, name), os.path.join(output_dir, name))

    committed = _committed(diagram_render.RENDER_CONFIG_FILENAME)
    if figures is None:
        figures = [
            {
                "source": source,
                "outputs": [
                    {"format": "png", "width_px": width_px, "background": "white"},
                    {"format": "svg", "background": "white"},
                ],
            }
            for source in (FIXTURE_FLOWCHART, FIXTURE_SEQUENCE)
        ]
    config = {
        "schema": committed["schema"],
        "image": committed["image"],
        "reproducibility_level": committed["reproducibility_level"],
        "mermaid_configs": committed["mermaid_configs"],
        "figures": figures,
    }
    with open(os.path.join(output_dir, diagram_render.RENDER_CONFIG_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    with open(os.path.join(output_dir, cli.MANIFEST_FILENAME), "w", encoding="utf-8") as handle:
        json.dump({"schema": diagram_manifest.MANIFEST_SCHEMA, "generator": {"version": "test"}}, handle)

    return output_dir


def _rendered_names(output_dir):
    return sorted(
        name for name in os.listdir(str(output_dir)) if name.endswith(".png") or name.endswith(".svg")
    )


# -- the pin --------------------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    [
        "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli",
        "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli:latest",
        "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli:11.17.1",
        "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli@sha256:deadbeef",
        "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli@md5:" + "a" * 64,
    ],
)
def test_an_unpinned_image_reference_is_rejected(reference):
    """The pin must not be losable by editing one string.

    A tag is a name, not an artifact: ``:11.17.1`` can be repushed, and it
    already disagrees with the version it packages upstream. Accepting one with
    a warning would leave a manifest that looks pinned and is not.
    """
    with pytest.raises(RenderError) as raised:
        diagram_render.validate_image_reference(reference, "render.json")
    assert "digest-pinned" in str(raised.value)


def test_a_digest_pinned_reference_is_accepted():
    pinned = "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli@sha256:" + "a1" * 32
    assert diagram_render.validate_image_reference(pinned, "render.json") == pinned
    # A tag alongside the digest is still pinned by the digest.
    tagged = "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli:11.17.1@sha256:" + "a1" * 32
    assert diagram_render.validate_image_reference(tagged, "render.json") == tagged


def test_the_committed_render_config_is_digest_pinned_and_loadable():
    config = diagram_render.load_render_config(COMMITTED_RENDER_CONFIG)
    assert "@sha256:" in config.image
    assert config.reproducibility_level in ("dimensions", "bytes")
    assert {figure.source for figure in config.figures} == set(cli.GATED_FILES)
    for figure in config.figures:
        assert {output.format for output in figure.outputs} == {"png", "svg"}
        for output in figure.outputs:
            assert output.background == "white"
            if output.format == "png":
                assert output.target_width_px > 0
            else:
                assert output.target_width_px is None, "a vector has no pixel width to pin"


# -- the two configs ------------------------------------------------------


def test_the_png_config_keeps_html_labels_and_the_svg_config_drops_them():
    """The two outputs exist for different consumers and differ in exactly this.

    Chrome rasterizes its own ``foreignObject``, so the PNG keeps HTML labels.
    Nothing outside a browser implements it, so the committed vector must not.
    """
    png = _committed("mermaid_config_png.json")
    svg = _committed("mermaid_config_svg.json")
    assert png["htmlLabels"] is True and png["flowchart"]["htmlLabels"] is True
    assert svg["htmlLabels"] is False and svg["flowchart"]["htmlLabels"] is False


def test_neither_config_reaches_for_the_css_injection_surfaces():
    """``securityLevel: loose``, ``themeCSS`` and ``--cssFile`` stay unused."""
    for name in ("mermaid_config_png.json", "mermaid_config_svg.json"):
        config = _committed(name)
        assert config.get("securityLevel") == "strict", name
        assert "themeCSS" not in config, name


def test_both_configs_pin_the_same_font_explicitly():
    families = {
        name: _committed(name)["themeVariables"]["fontFamily"]
        for name in ("mermaid_config_png.json", "mermaid_config_svg.json")
    }
    assert len(set(families.values())) == 1, families
    assert diagram_render.load_render_config(COMMITTED_RENDER_CONFIG).font_family == next(
        iter(families.values())
    )


def test_a_mermaid_config_with_no_font_family_is_rejected(tmp_path):
    output_dir = _prepare(tmp_path / "out")
    unpinned = {"theme": "default", "themeVariables": {}}
    with open(os.path.join(output_dir, "mermaid_config_png.json"), "w", encoding="utf-8") as handle:
        json.dump(unpinned, handle)

    with pytest.raises(RenderError) as raised:
        diagram_render.load_render_config(os.path.join(output_dir, diagram_render.RENDER_CONFIG_FILENAME))
    assert "fontFamily" in str(raised.value)


def test_configs_that_disagree_about_the_font_are_rejected(tmp_path):
    output_dir = _prepare(tmp_path / "out")
    other = _committed("mermaid_config_svg.json")
    other["themeVariables"]["fontFamily"] = "FreeSans"
    with open(os.path.join(output_dir, "mermaid_config_svg.json"), "w", encoding="utf-8") as handle:
        json.dump(other, handle)

    with pytest.raises(RenderError) as raised:
        diagram_render.load_render_config(os.path.join(output_dir, diagram_render.RENDER_CONFIG_FILENAME))
    assert "different fonts" in str(raised.value)


def test_a_font_the_image_does_not_ship_fails_loudly():
    """Silent fallback is the failure the font pin exists to prevent.

    It leaves a figure that rendered fine, with different metrics, and nothing
    anywhere saying so -- so an absent family has to stop the run.
    """
    with pytest.raises(RenderError) as raised:
        diagram_render._verify_font(
            "Comic Sans MS", {"DejaVu Sans", "FreeSans"}, {"attempted": True, "reason": None}, "cfg.json"
        )
    message = str(raised.value)
    assert "not present in the pinned image" in message
    assert "DejaVu Sans" in message, "the error should list what the image does ship"


def test_an_unreadable_font_list_degrades_instead_of_failing():
    """A degraded probe is not a wrong figure, and must not be treated as one."""
    note = diagram_render._verify_font(
        "DejaVu Sans", set(), {"attempted": True, "reason": "fc-list missing"}, "cfg.json"
    )
    assert note and "fc-list missing" in note


@pytest.mark.parametrize("width", [0, -100, "2700", 2700.0, True])
def test_a_png_width_that_is_not_a_positive_integer_is_rejected(tmp_path, width):
    """The PNG width is a target the step hits, never a scale factor from config.

    A fixed scale would give a different width, and a different effective DPI,
    every time the traced graph changed.
    """
    figures = [
        {"source": FIXTURE_FLOWCHART, "outputs": [{"format": "png", "width_px": width, "background": "white"}]}
    ]
    output_dir = _prepare(tmp_path / "out", figures=figures)
    with pytest.raises(RenderError):
        diagram_render.load_render_config(os.path.join(output_dir, diagram_render.RENDER_CONFIG_FILENAME))


def test_a_width_on_an_svg_output_is_rejected(tmp_path):
    """mermaid-cli emits a responsive root, so a vector has no width to pin."""
    figures = [
        {"source": FIXTURE_FLOWCHART, "outputs": [{"format": "svg", "width_px": 2700, "background": "white"}]}
    ]
    output_dir = _prepare(tmp_path / "out", figures=figures)
    with pytest.raises(RenderError) as raised:
        diagram_render.load_render_config(os.path.join(output_dir, diagram_render.RENDER_CONFIG_FILENAME))
    assert "responsive root" in str(raised.value)


# -- the header the flowchart parser rejects ------------------------------


def test_bare_comment_markers_are_padded_so_the_flowchart_parser_accepts_them():
    """Mermaid's comment strip needs a character after ``%%``.

    A bare ``%%`` line survives the strip, reaches the parser and makes the
    flowchart grammar fail outright. The sequenceDiagram grammar tolerates it,
    which is why only ``data_flow.mmd`` is affected.
    """
    source = "%% header\n%%\n%% more\n%%   \n\nflowchart LR\n    a --> b\n"
    normalized, changed = diagram_render.normalize_mermaid_source(source)

    assert changed == 1, "only the truly bare marker needs padding"
    assert "\n%% \n" in normalized
    # Nothing but the bare markers moves.
    assert normalized.replace("%% \n", "%%\n") == source
    for line in normalized.split("\n"):
        # `.lstrip()`, not `.strip()`: the padded form is exactly what we want,
        # and it strips back to "%%".
        assert line.lstrip() != "%%", "a bare marker survived normalization"


def test_the_committed_figures_are_the_reason_normalization_exists():
    """Both committed figures carry bare markers today; this is not hypothetical."""
    for name in cli.GATED_FILES:
        with open(os.path.join(ARCHITECTURE_DIR, name), encoding="utf-8") as handle:
            source = handle.read()
        _normalized, changed = diagram_render.normalize_mermaid_source(source)
        assert changed > 0, "{} no longer needs normalization -- the emitter may be fixed, in " \
                            "which case drop the workaround".format(name)


def test_normalization_never_touches_the_committed_file(tmp_path):
    output_dir = _prepare(tmp_path / "out")
    before = open(os.path.join(output_dir, FIXTURE_FLOWCHART), encoding="utf-8").read()
    with pytest.raises(RenderError):
        diagram_render.render(output_dir, docker=str(tmp_path / "no-such-docker"))
    assert open(os.path.join(output_dir, FIXTURE_FLOWCHART), encoding="utf-8").read() == before


# -- dimension readers ----------------------------------------------------


def test_png_dimensions_are_read_from_the_ihdr_chunk(tmp_path):
    path = tmp_path / "fake.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", 2700, 1431)
        + b"\x08\x06\x00\x00\x00"
    )
    assert diagram_render._png_dimensions(str(path)) == (2700, 1431)


def test_something_that_is_not_a_png_is_not_reported_as_one(tmp_path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>")
    with pytest.raises(RenderError) as raised:
        diagram_render._png_dimensions(str(path))
    assert "not a valid PNG" in str(raised.value)


@pytest.mark.parametrize(
    "attributes, expected",
    [
        ('width="2700" height="1431"', (2700, 1431)),
        ('width="2700px" height="1431px"', (2700, 1431)),
        ('viewBox="0 0 2700 1431"', (2700, 1431)),
        ('width="100%" viewBox="0 0 2700 1431"', (2700, 1431)),
        ('width="100%"', (None, None)),
    ],
)
def test_svg_dimensions_fall_back_from_attributes_to_the_viewbox(tmp_path, attributes, expected):
    path = tmp_path / "fake.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" {}></svg>'.format(attributes), encoding="utf-8")
    assert diagram_render._svg_dimensions(str(path)) == expected


# -- the drift gate -------------------------------------------------------


def test_rendered_figures_stay_out_of_the_drift_gate():
    """Raster and renderer output are not byte-stable the way a .mmd body is.

    Gating them would fire the *architecture* drift gate on a browser, font or
    Mermaid version bump -- the same reasoning that correctly kept
    ``coverage_report.md`` out. This test exists so a later, well-meaning
    addition trips instead of quietly making the gate cry wolf.
    """
    config = diagram_render.load_render_config(COMMITTED_RENDER_CONFIG)
    rendered = {
        figure.output_name(output) for figure in config.figures for output in figure.outputs
    }
    assert rendered, "the committed render config produces no figures at all"
    assert rendered.isdisjoint(cli.GATED_FILES)
    assert rendered.isdisjoint(cli.ARTIFACT_FILENAMES)
    assert all(name.endswith((".png", ".svg")) for name in rendered)


# -- failure behavior, with no Docker needed ------------------------------


def test_docker_unavailable_exits_non_zero_and_writes_nothing(tmp_path, capsys):
    """Integration plan #7. Runs everywhere; needs no daemon to prove."""
    output_dir = _prepare(tmp_path / "out")
    before = sorted(os.listdir(output_dir))

    exit_code = diagram_render.main(
        ["--output-dir", output_dir, "--docker", str(tmp_path / "no-such-docker")]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Render failed" in captured.err
    assert "was not found on PATH" in captured.err
    assert _rendered_names(output_dir) == []
    assert sorted(os.listdir(output_dir)) == before, "a failed render left something behind"
    # The manifest generation produced is untouched, not half-rewritten.
    with open(os.path.join(output_dir, cli.MANIFEST_FILENAME), encoding="utf-8") as handle:
        assert "render" not in json.load(handle)


def test_no_host_mmdc_fallback_exists():
    """A host ``mmdc`` would silently defeat the pin, so there is no fallback."""
    source = open(diagram_render.__file__, encoding="utf-8").read()
    assert "shutil.which(\"mmdc\")" not in source
    assert "'mmdc'" not in source.replace("mmdc -p /puppeteer-config.json", "")


def test_a_missing_source_mmd_is_rejected_before_docker_is_touched(tmp_path):
    output_dir = _prepare(tmp_path / "out")
    os.remove(os.path.join(output_dir, FIXTURE_FLOWCHART))

    with pytest.raises(RenderError) as raised:
        diagram_render.render(output_dir, docker=str(tmp_path / "no-such-docker"))
    assert "Figure source not found" in str(raised.value)
    assert _rendered_names(output_dir) == []


def test_a_missing_manifest_is_rejected_with_the_command_that_fixes_it(tmp_path):
    output_dir = _prepare(tmp_path / "out")
    os.remove(os.path.join(output_dir, cli.MANIFEST_FILENAME))

    with pytest.raises(RenderError) as raised:
        diagram_render.render(output_dir, docker=str(tmp_path / "no-such-docker"))
    assert "python -m tidepool_data_science_simulator.diagramgen" in str(raised.value)


# -- manifest recording ---------------------------------------------------


def _render_block(facts, probe_status, **overrides):
    kwargs = dict(
        repo_root=REPO_ROOT,
        image="ghcr.io/x/y@sha256:" + "a1" * 32,
        facts=facts,
        probe_status=probe_status,
        font_family="DejaVu Sans",
        font_note=None,
        config_paths=[COMMITTED_RENDER_CONFIG],
        figures=[],
        reproducibility_level="dimensions",
    )
    kwargs.update(overrides)
    return diagram_manifest.build_render_block(**kwargs)


def test_the_generated_manifest_declares_schema_2():
    assert diagram_manifest.MANIFEST_SCHEMA == "trset51-architecture-manifest/2"


def test_a_failing_probe_is_recorded_rather_than_raised():
    block = _render_block({}, {"attempted": True, "reason": "docker exited 125"})
    assert block["toolchain_probe"] == {"attempted": True, "reason": "docker exited 125"}
    assert block["mermaid_cli_version"] is None
    assert "docker exited 125" in block["toolchain_errors"]["mermaid_cli_version"]


def test_unresolved_and_not_attempted_do_not_look_alike():
    """A validation record must distinguish "we could not read it" from "we did not look"."""
    unresolved = _render_block({"mermaid_version": "11.12.0"}, {"attempted": True, "reason": None})
    not_attempted = _render_block({}, {"attempted": False, "reason": "probe skipped"})

    assert unresolved["mermaid_version"] == "11.12.0"
    assert unresolved["mermaid_cli_version"] is None
    assert unresolved["toolchain_errors"]["mermaid_cli_version"] == (
        "the probe returned no value for this field"
    )
    assert not_attempted["toolchain_errors"]["mermaid_cli_version"] == "the version probe was not run"
    assert unresolved["toolchain_probe"]["attempted"] is True
    assert not_attempted["toolchain_probe"]["attempted"] is False


def test_a_fully_resolved_probe_records_no_errors():
    facts = {field: "x" for field in diagram_manifest._RENDER_TOOLCHAIN_FIELDS}
    block = _render_block(facts, {"attempted": True, "reason": None})
    assert block["toolchain_errors"] is None


def test_the_render_block_names_the_sandbox_it_does_not_have():
    """The pinned image's entrypoint sets ``--no-sandbox``; the manifest says so.

    Leaving it unrecorded would let a reader assume an isolation property this
    render does not have.
    """
    block = _render_block({}, {"attempted": True, "reason": None})
    assert block["isolation"]["chromium_sandbox"] is False
    assert "--no-sandbox" in block["isolation"]["chromium_sandbox_reason"]
    assert block["isolation"]["network"] == "none"


def test_the_render_block_records_no_absolute_path():
    block = _render_block({}, {"attempted": True, "reason": None})
    assert block["configs"][0]["path"] == ".docs/architecture/{}".format(
        diagram_render.RENDER_CONFIG_FILENAME
    )
    assert "/Users/" not in json.dumps(block)


def test_a_config_from_outside_the_repo_keeps_its_basename_not_its_path(tmp_path):
    """``repo_relative`` falls back to an absolute path outside the repo and $HOME.

    A manifest is committed, so it must not carry one -- a render driven by
    ``--render-config /somewhere/else/render.json`` would otherwise name a
    directory on somebody's machine.
    """
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")
    block = _render_block({}, {"attempted": True, "reason": None}, config_paths=[str(outside)])

    assert block["configs"][0]["path"] == "elsewhere.json"
    assert block["configs"][0]["outside_repo"] is True
    assert str(tmp_path) not in json.dumps(block)


# -- the real image -------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_render(tmp_path_factory):
    """One real render of both fixtures, shared by the assertions that read it."""
    output_dir = _prepare(tmp_path_factory.mktemp("render"))
    manifest = diagram_render.render(output_dir, repo_root=REPO_ROOT)
    return output_dir, manifest


@requires_docker
@pytest.mark.docker
def test_both_fixtures_render_to_both_formats_at_the_configured_width(fixture_render):
    """Integration plan #1 and #2."""
    output_dir, manifest = fixture_render

    assert _rendered_names(output_dir) == [
        "render_flowchart.png",
        "render_flowchart.svg",
        "render_sequence.png",
        "render_sequence.svg",
    ]
    entries = {entry["output"]: entry for entry in manifest["render"]["figures"]}
    assert len(entries) == 4
    for name, entry in entries.items():
        assert os.path.getsize(os.path.join(output_dir, name)) > 0, name
        assert entry["height_px"] and entry["height_px"] > 0, name
        assert entry["sha256"] == _sha256(os.path.join(output_dir, name)), name

    # The PNG is the deliverable, and it lands on the configured width because
    # the step measured the figure and derived a scale for it -- not because a
    # scale factor was guessed in config.
    for name in ("render_flowchart.png", "render_sequence.png"):
        entry = entries[name]
        assert entry["target_width_px"] == FIXTURE_WIDTH_PX, name
        assert abs(entry["width_px"] - FIXTURE_WIDTH_PX) <= 2, name
        assert entry["natural_width_px"] and entry["natural_width_px"] != FIXTURE_WIDTH_PX, (
            "{}: the fixture should need a real scale, or this proves nothing".format(name)
        )
        assert abs(entry["scale"] - FIXTURE_WIDTH_PX / entry["natural_width_px"]) < 1e-4, name
        actual = diagram_render._png_dimensions(os.path.join(output_dir, name))[0]
        assert abs(actual - FIXTURE_WIDTH_PX) <= 2, name

    # The SVG is resolution-independent; its intrinsic viewBox size is recorded
    # and no scale is applied.
    for name in ("render_flowchart.svg", "render_sequence.svg"):
        entry = entries[name]
        assert entry["target_width_px"] is None, name
        assert entry["scale"] is None, name
        assert entry["width_px"] and entry["width_px"] > 0, name


@requires_docker
@pytest.mark.docker
def test_the_committed_svg_carries_no_foreign_object(fixture_render):
    """Integration plan #2: proof that ``htmlLabels: false`` took effect.

    ``<br/>`` in the fixture's labels is exactly what Mermaid renders through a
    ``foreignObject`` when HTML labels are on. Nothing outside a browser
    implements it, so its presence would make the committed vector useless to
    every importer that will be asked to read it.
    """
    output_dir, _manifest = fixture_render
    for name in ("render_flowchart.svg", "render_sequence.svg"):
        with open(os.path.join(output_dir, name), encoding="utf-8") as handle:
            markup = handle.read()
        assert "foreignObject" not in markup, name
        assert "<svg" in markup, name


@requires_docker
@pytest.mark.docker
def test_the_manifest_render_block_records_the_whole_toolchain(fixture_render):
    """Integration plan #3."""
    _output_dir, manifest = fixture_render
    block = manifest["render"]

    assert manifest["schema"] == "trset51-architecture-manifest/2"
    assert "@sha256:" in block["image"]
    assert block["mermaid_cli_version"], block["toolchain_errors"]
    assert block["mermaid_version"], block["toolchain_errors"]
    assert block["puppeteer_version"] or block["puppeteer_core_version"], block["toolchain_errors"]
    assert block["browser"], block["toolchain_errors"]
    assert block["font_family"] == "DejaVu Sans"
    assert block["font_verification_error"] is None, "the font list should be readable in the image"
    assert block["rendered_utc"].endswith("+00:00")

    recorded_configs = {entry["path"] for entry in block["configs"]}
    assert len(recorded_configs) == 3, recorded_configs
    for entry in block["configs"]:
        assert len(entry["sha256"]) == 64


@requires_docker
@pytest.mark.docker
def test_rendering_an_older_schema_1_manifest_upgrades_its_declaration(tmp_path):
    """A /1 manifest plus a render block is a valid /2 document, and must say so.

    Manifests generated before this change declare /1. Adding a block that /1
    does not define while leaving the declaration alone would make the committed
    file misdescribe its own shape to any consumer that checks the schema first.
    """
    output_dir = _prepare(tmp_path / "out")
    manifest_path = os.path.join(output_dir, cli.MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"schema": "trset51-architecture-manifest/1", "packages": []}, handle)

    manifest = diagram_render.render(output_dir, repo_root=REPO_ROOT)

    assert manifest["schema"] == "trset51-architecture-manifest/2"
    assert manifest["packages"] == [], "the rest of the manifest must survive untouched"
    with open(manifest_path, encoding="utf-8") as handle:
        assert json.load(handle)["schema"] == "trset51-architecture-manifest/2"


@requires_docker
@pytest.mark.docker
def test_no_absolute_host_path_appears_anywhere_in_the_manifest(fixture_render):
    """Integration plan #4. Two of four packages install from ``-e file:/Users/...``."""
    _output_dir, manifest = fixture_render
    serialized = json.dumps(manifest)
    assert os.path.expanduser("~") not in serialized
    assert "/Users/" not in serialized
    assert "/private/var/folders" not in serialized


@requires_docker
@pytest.mark.docker
def test_a_second_render_reproduces_the_figure(tmp_path):
    """Integration plan #5, asserted at the level ``render.json`` claims.

    ``dimensions`` is the claim that holds without measurement. The byte
    comparison is *reported* either way, so the empirical answer is captured on
    a real host rather than assumed; raise ``reproducibility_level`` to
    ``bytes`` in ``render.json`` once it has been observed to hold.
    """
    level = diagram_render.load_render_config(COMMITTED_RENDER_CONFIG).reproducibility_level

    first = diagram_render.render(_prepare(tmp_path / "first"), repo_root=REPO_ROOT)
    second = diagram_render.render(_prepare(tmp_path / "second"), repo_root=REPO_ROOT)

    left = {entry["output"]: entry for entry in first["render"]["figures"]}
    right = {entry["output"]: entry for entry in second["render"]["figures"]}
    assert set(left) == set(right)

    byte_identical = True
    for name in sorted(left):
        assert (left[name]["width_px"], left[name]["height_px"]) == (
            right[name]["width_px"],
            right[name]["height_px"],
        ), "{} changed size between two renders of the same pinned image".format(name)
        byte_identical = byte_identical and left[name]["sha256"] == right[name]["sha256"]

    print("\nmeasured reproducibility: {} (render.json claims {!r})".format(
        "byte-identical" if byte_identical else "dimension-identical only", level
    ))
    if level == "bytes":
        assert byte_identical, (
            "render.json claims byte-stability and this host does not reproduce it; "
            "lower reproducibility_level to 'dimensions'"
        )


@requires_docker
@pytest.mark.docker
def test_the_drift_gate_still_passes_after_a_render(fixture_render):
    """Integration plan #6: rendering must not disturb the gated ``.mmd`` bodies."""
    output_dir, _manifest = fixture_render
    for name in (FIXTURE_FLOWCHART, FIXTURE_SEQUENCE):
        with open(os.path.join(str(output_dir), name), encoding="utf-8") as handle:
            rendered_source = handle.read()
        with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as handle:
            committed_source = handle.read()
        assert rendered_source == committed_source, "{} was modified by the render step".format(name)


@requires_docker
@pytest.mark.docker
def test_a_broken_mmd_leaves_no_partial_artifact_behind(tmp_path):
    """A non-zero ``mmdc`` exit must not leave a half-figure in the output."""
    output_dir = _prepare(tmp_path / "out")
    with open(os.path.join(output_dir, FIXTURE_FLOWCHART), "w", encoding="utf-8") as handle:
        handle.write("flowchart LR\n    this is not ((valid)) mermaid -->\n")

    with pytest.raises(RenderError) as raised:
        diagram_render.render(output_dir, repo_root=REPO_ROOT)
    assert "mmdc failed" in str(raised.value)
    assert _rendered_names(output_dir) == []
    with open(os.path.join(output_dir, cli.MANIFEST_FILENAME), encoding="utf-8") as handle:
        assert "render" not in json.load(handle)


def _sha256(path):
    import hashlib

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()
