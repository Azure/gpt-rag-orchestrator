from pathlib import Path


def test_frontend_build_uses_locked_dependencies_including_dev_tools():
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    frontend_stage = dockerfile.split(
        "FROM mcr.microsoft.com/devcontainers/python:", 1
    )[0]

    install = "RUN cd frontend && npm ci --include=dev --no-audit --no-fund"
    assert install in frontend_stage
    assert "npm install" not in frontend_stage
    assert "npm upgrade" not in frontend_stage
    assert frontend_stage.index("COPY frontend/package*.json ./frontend/") < (
        frontend_stage.index(install)
    )
    assert frontend_stage.index(install) < frontend_stage.index(
        "RUN cd frontend && npm run build"
    )


def test_runtime_image_precaches_tokenizer_for_private_hosted_execution():
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert 'ENV TIKTOKEN_CACHE_DIR="/app/.cache/tiktoken"' in dockerfile
    assert "tiktoken.get_encoding('o200k_base')" in dockerfile
    assert 'chmod -R a+rX "$TIKTOKEN_CACHE_DIR"' in dockerfile
