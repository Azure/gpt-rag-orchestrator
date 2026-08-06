from pathlib import Path


def test_runtime_image_precaches_tokenizer_for_private_hosted_execution():
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert 'ENV TIKTOKEN_CACHE_DIR="/app/.cache/tiktoken"' in dockerfile
    assert "tiktoken.get_encoding('o200k_base')" in dockerfile
    assert 'chmod -R a+rX "$TIKTOKEN_CACHE_DIR"' in dockerfile
