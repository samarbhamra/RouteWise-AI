"""OSM highway / maxspeed parsing helpers."""

HIGHWAY_CODE = {
    "motorway": 5,
    "motorway_link": 5,
    "trunk": 4,
    "trunk_link": 4,
    "primary": 3,
    "primary_link": 3,
    "secondary": 2,
    "secondary_link": 2,
    "tertiary": 1,
    "tertiary_link": 1,
    "residential": 0,
    "unclassified": 0,
    "living_street": 0,
    "service": 0,
}


def parse_highway(value) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return 0
    return HIGHWAY_CODE.get(value.lower(), 0)


def is_highway(code: int) -> bool:
    return int(code) >= 4


def parse_maxspeed(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.lower().replace("mph", "").replace("km/h", "").replace("kmh", "").strip()
        try:
            num = float(raw)
        except ValueError:
            return None
        if "mph" in value.lower():
            return num * 1.60934
        return num
    return None
