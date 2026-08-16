from mtplx.benchmarks.validators.basic import (
    validate_balanced_delimiters,
    validate_benchmark_output,
    validate_no_degenerate_loop,
)


def test_degenerate_loop_validator_passes_normal_text():
    text = (
        "Create a parser that reads prompt cases, validates each field, "
        "and reports clear errors without changing unrelated files."
    )

    result = validate_no_degenerate_loop(text)

    assert result.passed


def test_degenerate_loop_validator_flags_repeated_phrases():
    phrase = "the model repeats the same broken clause forever "
    text = phrase * 8

    result = validate_no_degenerate_loop(text, ngram_size=6, max_ngram_repeats=3)

    assert not result.passed
    assert "repeated 6-gram" in result.detail


def test_balanced_delimiters_passes_nested_code_shape():
    result = validate_balanced_delimiters("def f(x):\n    return {'value': [x + 1]}\n")

    assert result.passed


def test_balanced_delimiters_flags_unclosed_shape():
    result = validate_balanced_delimiters("def f(x):\n    return {'value': [x + 1}\n")

    assert not result.passed
    assert result.name == "balanced_delimiters"


def test_benchmark_output_runs_python_syntax_for_full_python_prompt():
    results = validate_benchmark_output(
        "from pathlib import Path\n\nVALUE: int = 1\n",
        category="warm_coding",
        prompt_id="python_modules_long_high_acceptance",
    )

    assert {item.name for item in results} >= {
        "no_degenerate_loop",
        "balanced_delimiters",
        "python_syntax",
    }


def test_benchmark_output_skips_python_syntax_for_continuation_fragment():
    results = validate_benchmark_output(
        "    row = {'value': value}\n    return row\n",
        category="warm_coding",
        prompt_id="long_warm_code_continuation",
    )

    assert "balanced_delimiters" in {item.name for item in results}
    assert "python_syntax" not in {item.name for item in results}
