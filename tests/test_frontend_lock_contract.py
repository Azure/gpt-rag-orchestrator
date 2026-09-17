"""Offline contracts for the dashboard's locked dependencies and build tools."""

import base64
import json
from pathlib import Path
from urllib.parse import urlsplit


FRONTEND = Path(__file__).parents[1] / "frontend"


def test_frontend_lock_uses_default_registry_tarballs():
    lock = json.loads((FRONTEND / "package-lock.json").read_text(encoding="utf-8"))

    for key, entry in lock["packages"].items():
        if not key:
            continue
        url = urlsplit(entry["resolved"])
        assert url.scheme == "https", key
        assert url.netloc == "registry.npmjs.org", key
        assert not url.query and not url.fragment, key
        name = entry.get("name", key.rsplit("node_modules/", 1)[-1])
        filename = f"{name.rsplit('/', 1)[-1]}-{entry['version']}.tgz"
        assert url.path == f"/{name}/-/{filename}", key


def test_frontend_lock_retains_valid_integrity_for_every_package():
    lock = json.loads((FRONTEND / "package-lock.json").read_text(encoding="utf-8"))
    digest_sizes = {"sha1": 20, "sha256": 32, "sha384": 48, "sha512": 64}

    for key, entry in lock["packages"].items():
        if not key:
            continue
        assert entry["version"], key
        assert entry["integrity"], key
        for integrity in entry["integrity"].split():
            algorithm, encoded = integrity.split("-", 1)
            assert algorithm in digest_sizes, key
            digest = base64.b64decode(encoded, validate=True)
            assert len(digest) == digest_sizes[algorithm], key


def test_frontend_manifest_and_lock_include_the_ci_build_tools():
    manifest = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((FRONTEND / "package-lock.json").read_text(encoding="utf-8"))

    assert lock["lockfileVersion"] == 3
    for field in ("name", "version", "dependencies", "devDependencies"):
        assert lock["packages"][""][field] == manifest[field]

    # npm ci must include dev dependencies: tsc and Vite are build prerequisites.
    assert manifest["scripts"]["build"] == "tsc -b && vite build"
    for package, executable in (("typescript", "tsc"), ("vite", "vite")):
        assert package in manifest["devDependencies"]
        entry = lock["packages"][f"node_modules/{package}"]
        assert entry["dev"] is True
        assert executable in entry["bin"]
