from src.nlp.query_parser import parse_route_query


def test_parse_from_to_and_constraints():
    parsed = parse_route_query(
        "from downtown to the airport avoiding highways and minimize delay risk within 30 min"
    )
    assert "downtown" in parsed["origin"].lower()
    assert "airport" in parsed["destination"].lower()
    assert parsed["avoid_highways"] is True
    assert parsed["minimize_delay_risk"] is True
    assert parsed["max_travel_time_min"] == 30.0


def test_empty_query():
    parsed = parse_route_query("")
    assert parsed["origin"] == ""
    assert parsed["destination"] == ""
