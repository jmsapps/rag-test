import orjson


def dump_json(x):
    try:
        return orjson.dumps(
            x, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        ).decode()
    except Exception:
        import json

        return json.dumps(x, indent=2, ensure_ascii=False)
