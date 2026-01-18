import json

from ai_gateway.config.settings import TOOLS_CONFIG


def _serialize_tool(name: str) -> dict[str, object]:
    cfg = TOOLS_CONFIG.raw(name)
    return {
        "name": name,
        "description": cfg.description,
        "tags": cfg.tags,
        "disallowed_tags": cfg.disallowed_tags,
    }


def list_available_tools(tags: list[str]) -> str:
    """
    Return tools filtered by tag intersection and disallowed tags as JSON.
    """
    if not tags:
        payload = {
            "tools": [],
            "warning": "No tags provided. Provide agent tags to filter tools.",
        }
        return json.dumps(payload, ensure_ascii=True)

    tag_set = set(tags)
    tools = []
    for name in TOOLS_CONFIG.list_names():
        cfg = TOOLS_CONFIG.raw(name)
        tool_tags = set(cfg.tags)
        if tool_tags and not tool_tags.intersection(tag_set):
            continue
        if set(cfg.disallowed_tags).intersection(tag_set):
            continue
        tools.append(_serialize_tool(name))

    payload = {"tools": tools, "filtered_by_tags": tags}
    return json.dumps(payload, ensure_ascii=True)


def list_all_tools() -> str:
    """
    Return all tools with tags and disallowed tags as JSON.
    """
    tools = [_serialize_tool(name) for name in TOOLS_CONFIG.list_names()]
    payload = {"tools": tools}
    return json.dumps(payload, ensure_ascii=True)
