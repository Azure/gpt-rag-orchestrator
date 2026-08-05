from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / ".github" / "skills" / "release" / "SKILL.md"
RELEASE_AGENT_PATH = ROOT / ".github" / "agents" / "release.agent.md"


def _skill_parts() -> tuple[dict[str, str], str]:
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    _, frontmatter, body = content.split("---\n", maxsplit=2)
    return yaml.safe_load(frontmatter), body


def test_release_skill_has_discoverable_frontmatter() -> None:
    frontmatter, _ = _skill_parts()

    assert frontmatter["name"] == "release"
    description = frontmatter["description"].lower()
    for trigger in ("release", "version", "tag", "release notes"):
        assert trigger in description


def test_release_skill_locks_repository_release_contract() -> None:
    _, body = _skill_parts()

    required_contracts = (
        "release/X.Y.Z",
        "origin/develop",
        "pull request from `release/X.Y.Z` to\n`main`",
        "Semantic Versioning",
        "title exactly `vX.Y.Z`",
        "root `VERSION`",
        "`CHANGELOG.md`",
        "pytest -q",
        "explicit human approval",
        "Private Azure",
        "Rollback and reconciliation",
    )
    for contract in required_contracts:
        assert contract in body


def test_abandoned_release_agent_is_absent() -> None:
    assert not RELEASE_AGENT_PATH.exists()
