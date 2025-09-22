import orjson


def dump_json(obj):
    return orjson.dumps(
        obj,
        default=lambda o: o.__dict__,
        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
    ).decode("utf-8")
