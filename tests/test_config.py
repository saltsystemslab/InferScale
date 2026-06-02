from locomo_jasper_bench.config import parse_args


def test_context_mode_defaults_to_mem0():
    config = parse_args([])

    assert config.context_mode == "mem0"


def test_context_mode_accepts_full():
    config = parse_args(["--context-mode", "full"])

    assert config.context_mode == "full"


def test_context_mode_accepts_retrieval_alias():
    config = parse_args(["--context-mode", "retrieval"])

    assert config.context_mode == "mem0"
