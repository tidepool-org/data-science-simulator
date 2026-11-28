"""
Minimal span-aware JSON reader plus text-level edits.

Generated configs must preserve every byte except the intended change, so edits are made on the original text
at the exact span of the value being changed. A parsed ``Node`` records where each value starts and ends.
"""
import json


class Node:
    __slots__ = ("kind", "start", "end", "members", "items")

    def __init__(self, kind, start, end, members=None, items=None):
        self.kind, self.start, self.end = kind, start, end
        self.members = members or []   # obj: list of Member
        self.items = items or []       # arr: list of Node

    def value(self, text):
        return json.loads(text[self.start:self.end])


class Member:
    __slots__ = ("key", "key_start", "node")

    def __init__(self, key, key_start, node):
        self.key, self.key_start, self.node = key, key_start, node


class _Parser:
    def __init__(self, text):
        self.t, self.i = text, 0

    def ws(self):
        while self.i < len(self.t) and self.t[self.i] in " \t\r\n":
            self.i += 1

    def value(self):
        self.ws()
        c = self.t[self.i]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c == '"':
            return self.string()
        start = self.i
        while self.i < len(self.t) and self.t[self.i] not in ",}] \t\r\n":
            self.i += 1
        return Node("other", start, self.i)

    def string(self):
        start = self.i
        self.i += 1
        while self.t[self.i] != '"':
            self.i += 2 if self.t[self.i] == "\\" else 1
        self.i += 1
        return Node("str", start, self.i)

    def obj(self):
        start = self.i
        self.i += 1
        members = []
        self.ws()
        if self.t[self.i] == "}":
            self.i += 1
            return Node("obj", start, self.i, members=members)
        while True:
            self.ws()
            ks = self.i
            key = json.loads(self.t[ks:self.string().end])
            self.ws()
            assert self.t[self.i] == ":"
            self.i += 1
            members.append(Member(key, ks, self.value()))
            self.ws()
            c = self.t[self.i]
            self.i += 1
            if c == "}":
                return Node("obj", start, self.i, members=members)
            assert c == ","

    def arr(self):
        start = self.i
        self.i += 1
        items = []
        self.ws()
        if self.t[self.i] == "]":
            self.i += 1
            return Node("arr", start, self.i, items=items)
        while True:
            items.append(self.value())
            self.ws()
            c = self.t[self.i]
            self.i += 1
            if c == "]":
                return Node("arr", start, self.i, items=items)
            assert c == ","


def parse(text):
    p = _Parser(text)
    root = p.value()
    p.ws()
    if p.i != len(text):
        raise ValueError("trailing content after JSON value")
    return root


def get(node, path):
    """Node at ``path`` (keys / list indexes) or None."""
    for k in path:
        if node is None:
            return None
        if node.kind == "obj" and isinstance(k, str):
            node = next((m.node for m in node.members if m.key == k), None)
        elif node.kind == "arr" and isinstance(k, int) and k < len(node.items):
            node = node.items[k]
        else:
            return None
    return node


def apply(text, edits):
    """Apply non-overlapping (start, end, replacement) edits."""
    for s, e, r in sorted(edits, key=lambda x: x[0], reverse=True):
        text = text[:s] + r + text[e:]
    return text


def replace_edit(node, new_value):
    return (node.start, node.end, json.dumps(new_value))


def set_member_edit(text, obj, key, new_value):
    """Edit that sets ``key`` on an object node: replaces the value in place, or appends the member matching the
    surrounding layout (multi-line with the last member's indentation, or inline)."""
    for m in obj.members:
        if m.key == key:
            return replace_edit(m.node, new_value)
    val = json.dumps(new_value)
    if not obj.members:
        return (obj.start, obj.end, "{" + json.dumps(key) + ": " + val + "}")
    last = obj.members[-1]
    prev_end = obj.members[-2].node.end if len(obj.members) > 1 else obj.start + 1
    if "\n" in text[prev_end:last.key_start]:
        line_start = text.rfind("\n", 0, last.key_start) + 1
        indent = text[line_start:last.key_start]
        sep = ",\n" + indent
    else:
        sep = ", "
    return (last.node.end, last.node.end, sep + json.dumps(key) + ": " + val)
