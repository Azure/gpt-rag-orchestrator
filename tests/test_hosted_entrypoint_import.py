"""Regression test: the hosted-agent entrypoint must import cleanly on a
fresh interpreter.

``src/api/hosted_entrypoint.py`` is intentionally Cosmos-free and, unlike
``main.py``, never imports anything from ``connectors`` before it imports
``dependencies``. That ordering previously triggered a circular import
(``dependencies`` -> ``connectors`` (package init) -> ``connectors.cosmosdb``
-> ``dependencies`` again, before ``dependencies.get_config`` existed on the
partially-initialized module) that crashed the process on the very first
import -- before uvicorn could ever bind a port or serve ``GET /readiness``.
Deployed as a Foundry hosted agent, this manifested as the platform reporting
HTTP 424 ``session_not_ready`` on invoke ("container started but /readiness
didn't return HTTP 200 within the timeout"), because the container's Python
process never got that far.

A regular in-process pytest test cannot reproduce this: by the time this test
module runs, other test modules have already imported ``dependencies`` and
``connectors`` into ``sys.modules`` in some order that happens to avoid the
cycle, which would mask a regression. This test therefore spawns a fresh
interpreter, exactly like the container's ``uvicorn ...:app`` entrypoint
does, and imports only what the hosted entrypoint needs -- nothing else gets
a chance to "warm up" ``connectors``/``dependencies`` first.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _run_fresh_import(import_statement: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    # Keep the probe hermetic: no App Configuration endpoint to reach, no
    # Azure Monitor connection string, auth disabled so nothing else needs a
    # live dependency just to construct the ASGI app object.
    env.pop("APP_CONFIG_ENDPOINT", None)
    env.pop("APPLICATIONINSIGHTS_CONNECTION_STRING", None)
    env["DISABLE_AUTH"] = "true"
    return subprocess.run(
        [sys.executable, "-c", import_statement],
        cwd=SRC_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_hosted_entrypoint_imports_cleanly_on_a_fresh_interpreter() -> None:
    """``import api.hosted_entrypoint`` must not raise ImportError.

    This mirrors exactly how the hosted container's startup command
    (``uvicorn api.hosted_entrypoint:app`` -- see ``HOSTED_STARTUP_COMMAND``
    in the GPT-RAG umbrella's ``config/deployment/hosted_image.py``) loads
    this module: a fresh process whose first touch of this package tree is
    the hosted entrypoint itself.
    """
    completed = _run_fresh_import("import api.hosted_entrypoint")

    assert completed.returncode == 0, (
        "api.hosted_entrypoint failed to import on a fresh interpreter "
        f"(this is what the hosted agent container runs at startup):\n"
        f"{completed.stderr}"
    )
    assert "ImportError" not in completed.stderr
    assert "circular import" not in completed.stderr


def test_dependencies_module_alone_imports_without_touching_connectors() -> None:
    """``dependencies`` must not eagerly import ``connectors`` at module
    level -- that eager reference is exactly what created the cycle with
    connector submodules (``cosmosdb``, ``search``, ``aifoundry``, ...) that
    import ``dependencies`` back. ``AppConfigClient`` construction must stay
    deferred inside ``get_config()``.
    """
    completed = _run_fresh_import(
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import dependencies\n"
        "after = sorted(m for m in set(sys.modules) - before if m == 'connectors' or m.startswith('connectors.'))\n"
        "print(json.dumps(after))\n"
    )

    assert completed.returncode == 0, completed.stderr
    import json as _json

    loaded_connectors_modules = _json.loads(completed.stdout)
    assert loaded_connectors_modules == [], (
        "Importing `dependencies` alone must not pull in `connectors` at "
        f"module level (it did: {loaded_connectors_modules}); that eager "
        "coupling is what caused the dependencies<->connectors.cosmosdb "
        "circular import."
    )
