from locomo_jasper_bench.config import parse_args


def test_context_mode_defaults_to_full():
    config = parse_args([])

    assert config.context_mode == "full"


def test_context_mode_accepts_retrieval():
    config = parse_args(["--context-mode", "retrieval"])

    assert config.context_mode == "retrieval"
