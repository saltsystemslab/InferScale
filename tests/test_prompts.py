from locomo_jasper_bench.prompts import parse_judge_response


def test_parse_judge_response_from_json_object():
    correct, reason = parse_judge_response('{"correct": true, "reason": "same fact"}')

    assert correct is True
    assert reason == "same fact"


def test_parse_judge_response_from_wrapped_json():
    correct, reason = parse_judge_response('Result:\n{"correct": false, "reason": "wrong person"}')

    assert correct is False
    assert reason == "wrong person"
