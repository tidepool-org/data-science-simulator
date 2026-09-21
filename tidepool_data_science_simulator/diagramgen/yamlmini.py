"""
A deliberately tiny reader for the two version-controlled config files.

PyYAML is not a dependency of this repo and adding one to an environment that is
already difficult to reproduce (see the ticket's out-of-scope notes) buys very
little: ``allowlist.yml`` and ``exclusions.yml`` are flat mappings of a key to
either a scalar or a list of scalars. This reader accepts exactly that subset
and *rejects* everything else loudly, so a config file that grows a construct
this reader would silently misread fails the run instead.

Accepted grammar (the files remain valid YAML, readable by PyYAML if it is ever
added):

    # comment line
    key: scalar
    key:
      - item
      - item          # trailing comment

Rejected: tabs, nested mappings, flow style (``[a, b]`` / ``{a: b}``), anchors,
aliases, multi-line scalars, documents (``---``), and duplicate keys.
"""

__all__ = ["MiniYamlError", "load_mini_yaml"]


class MiniYamlError(ValueError):
    """Raised when a config file uses a construct outside the accepted subset."""


def _strip_comment(line):
    """Remove a trailing ``#`` comment.

    A ``#`` only starts a comment at the start of the line or when preceded by
    whitespace, so a value such as ``build#1`` survives.
    """
    out = []
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out)


def _parse_scalar(raw, lineno):
    value = raw.strip()
    if not value:
        return ""
    if value[0] in "[{":
        raise MiniYamlError("line {}: flow style is not supported: {!r}".format(lineno, raw))
    if value[0] in "*&":
        raise MiniYamlError("line {}: anchors and aliases are not supported: {!r}".format(lineno, raw))
    if value[0] in "|>":
        raise MiniYamlError("line {}: block scalars are not supported: {!r}".format(lineno, raw))
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def load_mini_yaml(text):
    """Parse ``text`` into a ``dict`` of ``str`` -> ``str`` or ``list[str]``.

    Parameters
    ----------
    text: str
        Contents of a config file.

    Returns
    -------
    dict
        Keys in file order; list values in file order.
    """
    result = {}
    current_key = None
    current_list = None

    for lineno, line in enumerate(text.splitlines(), start=1):
        if "\t" in line:
            raise MiniYamlError("line {}: tabs are not permitted in YAML indentation".format(lineno))

        stripped = _strip_comment(line).rstrip()
        if not stripped.strip():
            continue
        if stripped.strip() in ("---", "..."):
            raise MiniYamlError("line {}: multi-document files are not supported".format(lineno))

        indent = len(stripped) - len(stripped.lstrip(" "))
        body = stripped.strip()

        if body.startswith("- "):
            if current_list is None:
                raise MiniYamlError("line {}: list item outside of a key".format(lineno))
            if indent == 0:
                raise MiniYamlError("line {}: list items must be indented under their key".format(lineno))
            current_list.append(_parse_scalar(body[2:], lineno))
            continue

        if indent != 0:
            raise MiniYamlError("line {}: nested mappings are not supported: {!r}".format(lineno, line))

        if ":" not in body:
            raise MiniYamlError("line {}: expected 'key:' or '- item', got {!r}".format(lineno, line))

        key, _, rest = body.partition(":")
        key = key.strip()
        if not key:
            raise MiniYamlError("line {}: empty key".format(lineno))
        if key in result:
            raise MiniYamlError("line {}: duplicate key {!r}".format(lineno, key))

        if rest.strip():
            result[key] = _parse_scalar(rest, lineno)
            current_key, current_list = None, None
        else:
            current_list = []
            current_key = key
            result[key] = current_list

    del current_key
    return result
