"""The same structured tool schemas are used by compilation and the harness."""

TOOL_PROPERTIES = {
    "memory.write": {"key": {"type": "string"}, "text": {"type": "string"}},
    "memory.read": {"key": {"type": "string"}},
    "memory.update": {"record_id": {"type": "string"}, "text": {"type": "string"}},
    "memory.delete": {"record_id": {"type": "string"}},
    "history.search": {"query": {"type": "string"}},
    "history.open": {"id": {"type": "string"}},
}


def tool_schemas(names=None):
    names = list(TOOL_PROPERTIES) if names is None else names
    if any(name not in TOOL_PROPERTIES for name in names):
        raise ValueError("unknown tool name")
    return [{"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": TOOL_PROPERTIES[name],
                       "required": list(TOOL_PROPERTIES[name]), "additionalProperties": False},
    }} for name in names]
