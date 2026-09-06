"""Mutation fixtures for the source-only quality gate; no Azure imports."""

import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


QUALITY = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / ".github/scripts/check-quality.py")
)


def tree(tmp_path, files):
    for name, text in files.items():
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return QUALITY["collect"](tmp_path, ["src"])


def rules(**overrides):
    return {
        "forbidden": [{"from": ["connectors", "plugins", "telemetry"],
                       "to": ["main", "api"]}],
        "private_access": [],
        "dynamic_imports": [],
        **overrides,
    }


def graph(tmp_path, files, **overrides):
    modules = tree(tmp_path, files)
    edges, findings = QUALITY["architecture"](modules, rules(**overrides))
    return edges, findings


@pytest.mark.parametrize("files", [
    {"a.py": "import b", "b.py": "import a"},
    {"a.py": "import b", "b.py": "import c", "c.py": "import a"},
    {"a.py": "def f():\n import b", "b.py": "import a"},
    {"a.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n import b",
     "b.py": "import a"},
    {"a.py": "import pkg.b", "pkg/__init__.py": "", "pkg/b.py": "import a"},
])
def test_cycles_cover_flat_cross_root_late_and_type_only_imports(tmp_path, files):
    _, findings = graph(tmp_path, files)
    cycles = [f for f in findings if f["rule"] == "cycle"]
    assert cycles
    assert cycles[0]["dependency_path"][0] == cycles[0]["dependency_path"][-1]


def test_relative_sibling_does_not_invent_parent_initialization_edge(tmp_path):
    edges, findings = graph(tmp_path, {
        "pkg/__init__.py": "from . import a",
        "pkg/a.py": "from . import b",
        "pkg/b.py": "VALUE = 1",
    })
    assert not findings
    assert edges["pkg.a"] == {"pkg.b"}
    assert edges["pkg"] == {"pkg.a"}


def test_qualified_child_attribute_does_not_invent_facade_back_edge(tmp_path):
    edges, findings = graph(tmp_path, {
        "pkg/__init__.py": "from . import client",
        "pkg/client.py": "import pkg.impl\nvalue = pkg.impl.value",
        "pkg/impl.py": "value = 1",
    })
    assert not findings
    assert edges["pkg.client"] == {"pkg.impl"}


def test_real_facade_reexports_are_followed(tmp_path):
    _, findings = graph(tmp_path, {
        "a.py": "from pkg import value",
        "pkg/__init__.py": "from .impl import value",
        "pkg/impl.py": "import a\nvalue = 1",
    })
    assert any(f["rule"] == "cycle" for f in findings)


@pytest.mark.parametrize("statement", [
    "import api",
    "from api import route",
    "import helper",
])
def test_forbidden_direct_and_transitive_directions(tmp_path, statement):
    _, findings = graph(tmp_path, {
        "connectors/__init__.py": "",
        "connectors/client.py": statement,
        "helper.py": "import api.route",
        "api/__init__.py": "",
        "api/route.py": "",
    })
    assert any(f["rule"] == "forbidden-import" for f in findings)


@pytest.mark.parametrize("statement", [
    "from pkg._impl import public",
    "import pkg._impl as impl",
    "from pkg import _impl as impl",
    "from pkg.public import _secret as secret",
    "import pkg.public as p\nvalue = p._secret",
    "import pkg.public\nvalue = pkg.public._secret",
    "from pkg.public import Client as C\nvalue = C._secret",
    "import pkg.public as p\nvalue = p.Client._secret",
    "import pkg.public as p\nvalue = getattr(p, '_secret')",
])
def test_private_module_and_member_access(tmp_path, statement):
    _, findings = graph(tmp_path, {
        "consumer.py": statement, "pkg/__init__.py": "",
        "pkg/_impl.py": "public = 1",
        "pkg/public.py": "_secret = 1\nclass Client:\n _secret = 1",
    })
    assert any(f["rule"] == "private-access" for f in findings)


def test_private_facade_forwarding_and_explicit_compatibility_access(tmp_path):
    _, findings = graph(tmp_path, {
        "consumer.py": "from pkg import public",
        "pkg/__init__.py": "from ._impl import public",
        "pkg/_impl.py": "public = 1",
        "compat.py": "from pkg._impl import public",
    }, private_access=[{"importer": "compat", "target": "pkg._impl"}])
    assert not findings


def test_sibling_subpackages_cannot_access_each_others_private_members(tmp_path):
    _, findings = graph(tmp_path, {
        "pkg/__init__.py": "", "pkg/one/__init__.py": "", "pkg/two/__init__.py": "",
        "pkg/one/client.py": "from pkg.two.impl import _secret",
        "pkg/two/impl.py": "_secret = 1",
    })
    assert any(f["rule"] == "private-access" for f in findings)


@pytest.mark.parametrize("statement", [
    "import importlib\nimportlib.import_module(name)",
    "from importlib import import_module as load\nload(name)",
    "__import__(name)",
    "from builtins import __import__ as load\nload(name)",
])
def test_uninventoried_dynamic_loading_is_not_certified_safe(tmp_path, statement):
    _, findings = graph(tmp_path, {"a.py": statement})
    assert any(f["rule"] == "dynamic-import" for f in findings)


def test_literal_dynamic_import_participates_in_cycles(tmp_path):
    _, findings = graph(tmp_path, {
        "a.py": "import importlib as i\ni.import_module('b')",
        "b.py": "import a",
    })
    assert any(f["rule"] == "cycle" for f in findings)


def test_missing_first_party_target_is_an_error(tmp_path):
    _, findings = graph(tmp_path, {
        "a.py": "import pkg.missing", "pkg/__init__.py": "",
    })
    assert any(f["rule"] == "unresolved-import" for f in findings)


def test_collection_never_executes_runtime_source(tmp_path):
    modules = tree(tmp_path, {"main.py": "raise RuntimeError('do not execute')"})
    assert set(modules) == {"main"}


@pytest.mark.parametrize("header,prefix", [
    ("except:", ""),
    ("except Exception:", ""),
    ("except BaseException:", ""),
    ("except (ValueError, Exception):", ""),
    ("except builtins.Exception:", "import builtins\n"),
    ("except b.BaseException:", "import builtins as b\n"),
    ("except E:", "from builtins import Exception as E\n"),
    ("except E:", "E = Exception\n"),
    ("except* Exception:", ""),
])
@pytest.mark.parametrize("body", ["pass", "raise", "logging.exception('failed')"])
def test_broad_inventory_includes_logged_rethrown_aliased_and_group_handlers(
    tmp_path, header, prefix, body,
):
    modules = tree(tmp_path, {
        "a.py": f"{prefix}try:\n work()\n{header}\n {body}\n",
    })
    handlers = QUALITY["handlers"](modules)
    assert len(handlers) == 1
    assert handlers[0]["caught_types"]


def test_indirect_exception_expression_needs_review(tmp_path):
    modules = tree(tmp_path, {
        "a.py": "try:\n work()\nexcept types_to_catch():\n pass",
    })
    assert QUALITY["handlers"](modules)[0]["caught_types"] == ["<indirect>"]


def test_exception_alias_assigned_by_call_needs_review(tmp_path):
    modules = tree(tmp_path, {
        "a.py": "E = choose_errors()\ntry:\n work()\nexcept E:\n pass",
    })
    assert QUALITY["handlers"](modules)[0]["caught_types"] == ["<indirect>"]


def test_narrow_exception_is_not_broad(tmp_path):
    modules = tree(tmp_path, {
        "a.py": "try:\n work()\nexcept (ValueError, TypeError):\n raise",
    })
    assert not QUALITY["handlers"](modules)


def approval(handler):
    return {
        **handler, "id": "test-boundary", "status": "active",
        "boundary": "test boundary", "reason": "translate provider failure",
        "failure_outcome": "propagation", "diagnostic_path": "caller exception",
        "evidence_tests": ["tests/test_failure.py::test_failure"],
        "review": "protected-base review", "review_by_stage": "next-expansion",
    }


def test_exception_records_are_exact_and_require_executed_evidence(tmp_path):
    modules = tree(tmp_path, {"a.py": "try:\n work()\nexcept Exception:\n raise"})
    handlers = QUALITY["handlers"](modules)
    record = approval(handlers[0])
    passed = {"tests/test_failure.py::test_failure"}
    assert not QUALITY["check_exceptions"](handlers, [record], passed)
    assert QUALITY["check_exceptions"](handlers, [record], set())
    changed = tree(tmp_path, {"a.py": "try:\n work()\nexcept Exception:\n pass"})
    assert QUALITY["check_exceptions"](QUALITY["handlers"](changed), [record], passed)
    assert QUALITY["check_exceptions"]([], [record], passed)
    record["status"] = "proposed"
    assert QUALITY["check_exceptions"](handlers, [record], passed)


def test_changed_try_operation_and_duplicate_handlers_cannot_reuse_approval(tmp_path):
    code = "try:\n work()\nexcept Exception:\n raise"
    modules = tree(tmp_path / "before", {"a.py": code})
    handlers = QUALITY["handlers"](modules)
    record = approval(handlers[0])
    passed = {"tests/test_failure.py::test_failure"}
    changed = tree(tmp_path / "after", {"a.py": code.replace("work()", "other()")})
    assert QUALITY["check_exceptions"](QUALITY["handlers"](changed), [record], passed)
    duplicated = tree(tmp_path / "duplicate", {"a.py": code + "\n" + code})
    current = QUALITY["handlers"](duplicated)
    assert QUALITY["check_exceptions"](current, [record], passed)
    assert not QUALITY["approved_handler_sites"](current, [record], passed)


def test_baseline_is_an_individual_multiset_not_a_count():
    finding = {"module_id": "a", "symbol": "f", "source_fingerprint": "context-a",
               "rule": "assignment", "message_fingerprint": "message-a"}
    record = {**finding, "occurrences": 1}
    compare = QUALITY["ratchet"]
    assert compare([finding], [record]) == []
    assert compare([{**finding, "source_fingerprint": "context-b"}], [record])
    assert compare([finding, finding], [record])
    assert compare([], [record])  # Retire absent debt; no reusable allowances.


def test_module_moves_keep_identity_and_ambiguous_copies_fail(tmp_path):
    before = tree(tmp_path / "old", {"old.py": "x: int = 1"})
    after = tree(tmp_path / "new", {"new.py": "x: int = 1"})
    ids, findings = QUALITY["module_identities"](before, after)
    assert ids == {"new": "old"}
    assert not findings
    copies = tree(tmp_path / "copies", {"one.py": "x: int = 1", "two.py": "x: int = 1"})
    assert QUALITY["module_identities"](before, copies)[1]


def test_deleted_annotations_and_new_suppressions_fail(tmp_path):
    before = tree(tmp_path / "old", {"a.py": "def f(x: str) -> str:\n return x"})
    after = tree(tmp_path / "new", {"a.py": "def f(x):\n return x # type: ignore"})
    findings = QUALITY["source_policy"](before, after, {"a": "a"}, {"a"})
    assert {f["rule"] for f in findings} >= {"annotation-removed", "suppression"}


def test_baseline_schema_rejects_unknown_and_duplicate_entries():
    with pytest.raises(QUALITY["QualityError"]):
        QUALITY["validate_records"]({"schema_version": 99, "entries": []})
    with pytest.raises(QUALITY["QualityError"]):
        QUALITY["validate_records"](
            {"schema_version": 1, "entries": [{"id": "same"}, {"id": "same"}]}
        )


@pytest.mark.parametrize("value", [
    {"schema_version": True, "entries": []},
    {"schema_version": 1, "entries": [{"id": ""}]},
    {"schema_version": 1, "entries": [{"id": 3}]},
])
def test_invalid_record_identity_is_not_accepted(value):
    with pytest.raises(QUALITY["QualityError"]):
        QUALITY["validate_records"](value)


@pytest.mark.parametrize("value", ["skipped", "cancelled", "failure", "neutral", None])
def test_aggregate_rejects_missing_or_non_success_jobs(value):
    results = {name: "success" for name in QUALITY["REQUIRED_JOBS"]}
    results["tests"] = value
    assert not QUALITY["jobs_passed"](results)


def test_aggregate_rejects_stale_or_tampered_report():
    report = QUALITY["seal_report"](
        {"base_sha": "base", "head_sha": "head", "checks": {"lint": {"status": "passed"}}}
    )
    assert QUALITY["report_valid"](report, "base", "head", {"lint"})
    assert not QUALITY["report_valid"](report, "base", "different", {"lint"})
    report["checks"]["lint"]["status"] = "violations"
    assert not QUALITY["report_valid"](report, "base", "head", {"lint"})


def test_failed_tools_and_invalid_structured_output_fail_closed():
    with pytest.raises(QUALITY["QualityError"]):
        QUALITY["tool_json"](
            subprocess.CompletedProcess(["ruff"], 2, "[]", "tool failed"), "ruff"
        )
    with pytest.raises(QUALITY["QualityError"]):
        QUALITY["tool_json"](
            subprocess.CompletedProcess(["ruff"], 0, "not json", ""), "ruff"
        )
    with pytest.raises(QUALITY["QualityError"]):
        QUALITY["tool_json"](
            subprocess.CompletedProcess(["ruff"], 1, "[]", ""), "ruff"
        )


def test_tool_timeout_is_an_error():
    with pytest.raises(QUALITY["QualityError"], match="timed out"):
        QUALITY["run"]([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.01)


def test_invalid_cli_writes_error_report(tmp_path):
    output = tmp_path / "quality.json"
    proc = subprocess.run(
        [sys.executable, QUALITY["__file__"], "--base-ref", "not-a-ref",
         "--check", "policy", "--report", str(output)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert json.loads(output.read_text())["checks"]["policy"]["status"] == "error"


def test_real_ruff_and_mypy_reject_regressions(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("value: int = 'wrong'\nprint(missing)\n")
    lint = QUALITY["tool_json"](QUALITY["run"]([
        sys.executable, "-m", "ruff", "check", "--isolated", "--select", "F821",
        "--output-format", "json", str(source),
    ]), "ruff")
    assert [(item["code"], item["location"]["row"]) for item in lint] == [("F821", 2)]
    config = tmp_path / "mypy.ini"
    config.write_text("[mypy]\npython_version = 3.12\n")
    typing = QUALITY["tool_json"](QUALITY["run"]([
        sys.executable, "-m", "mypy", "--config-file", str(config),
        "--no-incremental", "--output", "json", str(source),
    ]), "mypy")
    assert {item["code"] for item in typing if item["severity"] == "error"} == {
        "assignment", "name-defined",
    }


def test_grimp_overlap_never_imports_package_initializers(tmp_path):
    modules = tree(tmp_path, {
        "pkg/__init__.py": "raise RuntimeError('source is not executable')",
        "pkg/a.py": "from . import b",
        "pkg/b.py": "value = 1",
        "flat.py": "import pkg.a",
    })
    edges, findings = QUALITY["architecture"](modules, rules())
    assert not findings
    assert QUALITY["grimp_check"](tmp_path, modules, edges) == 3


@pytest.fixture
def policy_repo(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True,
        ).stdout.strip()
    git("init", "--quiet")
    git("config", "user.name", "Quality fixture")
    git("config", "user.email", "quality@example.invalid")
    records = {
        "typing-scope.json": {"module_ids": ["a", "b"], "schema_version": 1},
        "typing-baseline.json": {"entries": [{"id": "old"}], "schema_version": 1},
    }
    for name, data in records.items():
        path = tmp_path / ".quality" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))
    for name in (".github/scripts/check-quality.py", ".github/CODEOWNERS"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# protected\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "fixture")
    return tmp_path, git("rev-parse", "HEAD"), records


@pytest.mark.parametrize("path,text", [
    (".github/scripts/check-quality.py", "print('passed')"),
    (".github/CODEOWNERS", "# no owners"),
    (".quality/typing-scope.json", '{"module_ids": ["a"], "schema_version": 1}'),
    (".quality/typing-baseline.json",
     '{"entries": [{"id": "old"}, {"id": "new"}], "schema_version": 1}'),
])
def test_candidate_cannot_bless_checker_baseline_owner_or_scope_weakening(policy_repo, path, text):
    root, base, records = policy_repo
    (root / path).write_text(text)
    candidate = json.loads(json.dumps(records))
    if path.startswith(".quality"):
        candidate[Path(path).name] = json.loads(text)
    findings = QUALITY["policy_changes"](root, base, records, candidate)
    assert [item["rule"] for item in findings] == ["protected-policy"]


def test_monotonic_scope_expansion_and_debt_retirement_are_allowed(policy_repo):
    root, base, records = policy_repo
    candidate = json.loads(json.dumps(records))
    candidate["typing-scope.json"]["module_ids"].append("new")
    candidate["typing-baseline.json"]["entries"] = []
    for name, value in candidate.items():
        (root / ".quality" / name).write_text(json.dumps(value))
    assert not QUALITY["policy_changes"](root, base, records, candidate)


def test_untracked_policy_file_cannot_avoid_review(policy_repo):
    root, base, records = policy_repo
    path = root / ".github/scripts/new-policy.py"
    path.write_text("# candidate-owned evaluator\n")
    findings = QUALITY["policy_changes"](root, base, records, records)
    assert [f["path"] for f in findings] == [".github/scripts/new-policy.py"]


@pytest.mark.parametrize("statement", [
    "from pkg.public import missing",
    "from pkg import missing",
    "from pkg.public import missing as alias",
])
def test_unresolved_first_party_export_is_not_a_valid_edge(tmp_path, statement):
    _, findings = graph(tmp_path, {
        "consumer.py": statement, "pkg/__init__.py": "", "pkg/public.py": "value = 1",
    })
    assert any(f["rule"] == "unresolved-import" for f in findings)


def test_real_symbol_reexport_remains_public(tmp_path):
    _, findings = graph(tmp_path, {
        "consumer.py": "from pkg import exported",
        "pkg/__init__.py": "from .public import value as exported",
        "pkg/public.py": "value = 1",
    })
    assert not findings


def test_dynamic_import_inventory_requires_exact_site_targets_and_evidence(tmp_path):
    code = "from importlib import import_module as load\nload(module_name)"
    modules = tree(tmp_path, {"loader.py": code, "target.py": "value = 1"})
    site = next(n for n in ast.walk(modules["loader"]["tree"]) if isinstance(n, ast.Call))
    record = {
        "id": "loader-target", "module_id": "loader",
        "symbol": "<module>",
        "source_fingerprint": QUALITY["syntax"](site), "targets": ["target"],
        "evidence_tests": ["tests/test_loader.py::test_target"], "review": "base review",
    }
    edges, findings = QUALITY["architecture"](
        modules, rules(dynamic_imports=[record]), {"tests/test_loader.py::test_target"})
    assert not findings
    assert edges["loader"] == {"target"}
    assert QUALITY["architecture"](modules, rules(dynamic_imports=[record]))[1]
    assert QUALITY["architecture"](
        tree(tmp_path, {"loader.py": code.replace("module_name", "different_name")}),
        rules(dynamic_imports=[record]), {"tests/test_loader.py::test_target"})[1]


def test_unused_dynamic_import_approval_fails(tmp_path):
    modules = tree(tmp_path, {"a.py": "value = 1"})
    record = {
        "id": "unused", "module_id": "a", "source_fingerprint": "old-site",
        "symbol": "<module>",
        "targets": ["a"], "evidence_tests": ["tests/test_loader.py::test_target"],
        "review": "base review",
    }
    assert QUALITY["architecture"](
        modules, rules(dynamic_imports=[record]), {"tests/test_loader.py::test_target"})[1]


def test_dynamic_site_cannot_duplicate_an_approved_allowance(tmp_path):
    modules = tree(tmp_path, {
        "loader.py": "import importlib\nimportlib.import_module(name)\nimportlib.import_module(name)",
        "target.py": "",
    })
    call = next(n for n in ast.walk(modules["loader"]["tree"]) if isinstance(n, ast.Call))
    record = {
        "id": "once", "module_id": "loader", "symbol": "<module>",
        "source_fingerprint": QUALITY["syntax"](call), "targets": ["target"],
        "evidence_tests": ["tests/test_loader.py::test_target"], "review": "base review",
    }
    assert QUALITY["architecture"](
        modules, rules(dynamic_imports=[record]), {"tests/test_loader.py::test_target"})[1]


@pytest.mark.parametrize("statement", [
    "import importlib\nload = getattr(importlib, 'import_module')\nload(name)",
    "from importlib import import_module\nload = import_module\nload(name)",
])
def test_indirect_loader_alias_needs_review(tmp_path, statement):
    _, findings = graph(tmp_path, {"loader.py": statement})
    assert any(f["rule"] == "dynamic-import" for f in findings)


def test_relative_literal_dynamic_import_resolves_without_running_code(tmp_path):
    edges, findings = graph(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/loader.py": "import importlib\nimportlib.import_module('.target', 'pkg')",
        "pkg/target.py": "raise RuntimeError('never execute source')",
    })
    assert not findings
    assert edges["pkg.loader"] == {"pkg.target"}


@pytest.mark.parametrize("expression", [
    "choose_errors()", "(ValueError, Exception)", "Exception if condition else ValueError",
])
def test_computed_exception_alias_is_reviewed(tmp_path, expression):
    modules = tree(tmp_path, {
        "a.py": f"E = {expression}\ntry:\n work()\nexcept E:\n pass",
    })
    assert QUALITY["handlers"](modules)[0]["caught_types"] == ["<indirect>"]


def test_computed_import_machinery_access_cannot_bypass_inventory(tmp_path):
    _, findings = graph(tmp_path, {
        "a.py": "import importlib\ngetattr(importlib, method_name)(module_name)",
    })
    assert any(f["rule"] == "dynamic-import" for f in findings)


def test_conflicting_local_alias_cannot_hide_broad_global_catch(tmp_path):
    modules = tree(tmp_path, {
        "a.py": "from builtins import Exception as E\n"
                "def helper():\n from builtins import ValueError as E\n"
                "try:\n work()\nexcept E:\n pass",
    })
    with pytest.raises(QUALITY["QualityError"], match="Ambiguous import alias"):
        QUALITY["handlers"](modules)


def test_conflicting_local_module_alias_cannot_hide_private_global_access(tmp_path):
    modules = tree(tmp_path, {
        "a.py": "import pkg as p\n"
                "def helper():\n import other as p\n"
                "value = p._private\n",
        "pkg.py": "_private = 1", "other.py": "value = 1",
    })
    with pytest.raises(QUALITY["QualityError"], match="Ambiguous import alias"):
        QUALITY["architecture"](modules, rules())


def test_local_variables_are_not_public_module_exports(tmp_path):
    _, findings = graph(tmp_path, {
        "consumer.py": "from pkg import local",
        "pkg.py": "def f():\n local = 1\n return local",
    })
    assert any(f["rule"] == "unresolved-import" for f in findings)


def test_audit_proposals_bind_source_without_authorizing_themselves():
    root = Path(QUALITY["__file__"]).resolve().parents[2]
    records = QUALITY["load_records"](root)["exceptions.json"]["entries"]
    current = QUALITY["handlers"](QUALITY["collect"](root, ["src"]))
    current = [h for h in current if h["module_id"] in (
        "telemetry.audit", "telemetry.audit_contract", "telemetry.audit_sanitizer")]
    assert len(current) == len(records) == 10
    passed = {test for record in records for test in record["evidence_tests"]}
    assert QUALITY["check_exceptions"](current, records, passed)
    reviewed_fixture = [{**record, "status": "active"} for record in records]
    assert not QUALITY["check_exceptions"](current, reviewed_fixture, passed)


@pytest.mark.parametrize("file,mutate", [
    ("policy.json", lambda data: data.update(extra_setting=True)),
    ("policy.json", lambda data: data["toolchain"].update(ruff=">=0.16")),
    ("policy.json", lambda data: data["contracts"]["private_access"].append(
        {"importer": "*", "target": "anything"})),
    ("typing-scope.json", lambda data: data.update(module_ids="schemas")),
    ("typing-scope.json", lambda data: data.update(move_map=[{"old": "new"}])),
    ("exceptions.json", lambda data: data["entries"][0].update(caught_types="Exception")),
    ("exceptions.json", lambda data: data["entries"][0].update(evidence_tests=[])),
    ("exceptions.json", lambda data: data["entries"][0].update(failure_outcome="ignore everything")),
])
def test_policy_documents_are_strictly_validated(tmp_path, file, mutate):
    root = Path(QUALITY["__file__"]).resolve().parents[2]
    (tmp_path / ".quality").mkdir()
    for name in QUALITY["RECORDS"]:
        data = json.loads((root / ".quality" / name).read_text())
        if name == file:
            mutate(data)
        (tmp_path / ".quality" / name).write_text(json.dumps(data))
    (tmp_path / "requirements-quality.txt").write_text(
        (root / "requirements-quality.txt").read_text())
    with pytest.raises(QUALITY["QualityError"]):
        QUALITY["load_records"](tmp_path)


def test_evidence_does_not_credit_skipped_failed_or_missing_tests(tmp_path):
    path = tmp_path / "pytest.xml"
    path.write_text(
        '<testsuite><testcase classname="tests.test_failure" name="test_pass"/>'
        '<testcase classname="tests.test_failure" name="test_skip"><skipped/></testcase>'
        '<testcase classname="tests.test_failure" name="test_fail"><failure/></testcase></testsuite>'
    )
    assert QUALITY["evidence_tests"](path) == {"tests/test_failure.py::test_pass"}


@pytest.mark.parametrize("mutation", ["missing", "stale", "tampered", "skipped", "policy"])
def test_actual_aggregate_fails_closed(tmp_path, mutation):
    for check in QUALITY["CHECKS"]:
        report = QUALITY["seal_report"]({
            "base_sha": "base", "head_sha": "head", "policy_sha": "policy",
            "checks": {check: {"status": "passed"}},
        })
        if check == "lint":
            if mutation == "missing":
                continue
            if mutation == "stale":
                report = QUALITY["seal_report"]({**report, "head_sha": "old"})
            if mutation == "tampered":
                report["head_sha"] = "edited"
            if mutation == "policy":
                report = QUALITY["seal_report"]({**report, "policy_sha": "other-policy"})
        (tmp_path / f"quality-{check}.json").write_text(json.dumps(report))
    needs = {name: {"result": "success"} for name in ("tests", "frontend", "quality")}
    if mutation == "skipped":
        needs["tests"]["result"] = "skipped"
    proc = subprocess.run(
        [sys.executable, str(Path(QUALITY["__file__"]).with_name("aggregate-quality.py")),
         "--reports", str(tmp_path), "--base-sha", "base", "--head-sha", "head",
         "--needs", json.dumps(needs)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
