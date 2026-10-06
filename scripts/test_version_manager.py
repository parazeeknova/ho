"""Tests for the changeset-style automatic version manager."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import version_manager as vm

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git binary not available")


# -- bump math ---------------------------------------------------------------


def test_bump_simple_patch() -> None:
    assert vm.bump_version("0.1.0") == "0.1.1"


def test_bump_from_zero() -> None:
    assert vm.bump_version("0.0.0") == "0.0.1"


def test_bump_patch_carries_into_minor() -> None:
    assert vm.bump_version("0.0.99") == "0.1.0"


def test_bump_patch_carry_mid_range() -> None:
    assert vm.bump_version("0.1.99") == "0.2.0"


def test_bump_minor_carries_into_major() -> None:
    assert vm.bump_version("0.99.99") == "1.0.0"


def test_bump_major_keeps_growing() -> None:
    assert vm.bump_version("9.99.99") == "10.0.0"


def test_bump_large_numbers() -> None:
    assert vm.bump_version("12.34.56") == "12.34.57"


def test_bump_tolerates_surrounding_whitespace() -> None:
    assert vm.bump_version("  0.0.5\n") == "0.0.6"


@pytest.mark.parametrize("raw", ["1.2", "1.2.3.4", "v1.2.3", "1.2.3-beta", "", "a.b.c", "1..3"])
def test_parse_rejects_non_semver(raw: str) -> None:
    with pytest.raises(ValueError):
        vm.parse_version(raw)
    with pytest.raises(ValueError):
        vm.bump_version(raw)


# -- scope mapping ------------------------------------------------------------


def test_owning_scope_nested_file() -> None:
    scopes = ["packages/autofill", "packages/node"]
    assert vm.owning_scope("packages/node/ats/ashby.ts", scopes) == "packages/node"


def test_owning_scope_root_file_is_none() -> None:
    scopes = ["packages/node"]
    assert vm.owning_scope("package.json", scopes) is None
    assert vm.owning_scope("lefthook.yml", scopes) is None


def test_owning_scope_respects_segment_boundary() -> None:
    scopes = ["packages/node"]
    assert vm.owning_scope("packages/node2/file.ts", scopes) is None
    assert vm.owning_scope("packages/node", scopes) == "packages/node"


def test_owning_scope_prefers_longest_prefix() -> None:
    scopes = ["packages/a", "packages/a/b"]
    assert vm.owning_scope("packages/a/b/c.ts", scopes) == "packages/a/b"


def test_package_dirs_lists_sorted_subdirs(tmp_path: Path) -> None:
    (tmp_path / "packages" / "node").mkdir(parents=True)
    (tmp_path / "packages" / "autofill").mkdir()
    (tmp_path / "package.json").write_text("{}")
    assert vm.package_dirs(tmp_path) == ["packages/autofill", "packages/node"]


def test_package_dirs_missing_packages_dir(tmp_path: Path) -> None:
    assert vm.package_dirs(tmp_path) == []


# -- manifest IO ---------------------------------------------------------------


def _write_package_json(path: Path, version: str) -> None:
    path.write_text(json.dumps({"name": "x", "version": version}) + "\n")


def test_package_json_round_trip_preserves_keys(tmp_path: Path) -> None:
    manifest_file = tmp_path / "package.json"
    manifest_file.write_text('{\n  "name": "x",\n  "version": "0.0.99"\n}\n')
    manifest = vm.Manifest(path=manifest_file, kind="package-json", scope="root")
    assert vm.read_version(manifest) == "0.0.99"
    vm.write_version(manifest, "0.1.0")
    data = json.loads(manifest_file.read_text())
    assert data == {"name": "x", "version": "0.1.0"}
    assert manifest_file.read_text().endswith("\n")


def test_package_json_without_version_is_skipped(tmp_path: Path) -> None:
    manifest_file = tmp_path / "package.json"
    manifest_file.write_text('{"name": "x"}\n')
    assert vm.manifests_in(tmp_path, "root") == []


def test_pyproject_round_trip_preserves_rest_of_file(tmp_path: Path) -> None:
    manifest_file = tmp_path / "pyproject.toml"
    before = (
        '[project]\nname = "x"\nversion = "0.1.99"  # keep me\ndescription = "y"\n\n'
        "[tool.ruff]\nline-length = 100\n"
    )
    manifest_file.write_text(before)
    manifest = vm.Manifest(path=manifest_file, kind="pyproject", scope="root")
    assert vm.read_version(manifest) == "0.1.99"
    vm.write_version(manifest, "0.2.0")
    after = manifest_file.read_text()
    assert 'version = "0.2.0"  # keep me' in after
    assert after.replace('version = "0.2.0"', 'version = "0.1.99"') == before


def test_pyproject_ignores_non_project_sections(tmp_path: Path) -> None:
    manifest_file = tmp_path / "pyproject.toml"
    manifest_file.write_text(
        '[project]\nname = "x"\nversion = "0.0.1"\n\n[project.urls]\nversion = "9.9.9"\n'
    )
    manifest = vm.Manifest(path=manifest_file, kind="pyproject", scope="root")
    vm.write_version(manifest, "0.0.2")
    text = manifest_file.read_text()
    assert 'version = "0.0.2"' in text
    assert 'version = "9.9.9"' in text


def test_pyproject_without_project_version(tmp_path: Path) -> None:
    manifest_file = tmp_path / "pyproject.toml"
    manifest_file.write_text("[tool.ruff]\nline-length = 100\n")
    assert vm.read_version(vm.Manifest(path=manifest_file, kind="pyproject", scope="root")) is None
    assert vm.manifests_in(tmp_path, "root") == []


# -- planning ------------------------------------------------------------------


def _fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "packages" / "node").mkdir(parents=True)
    (root / "packages" / "ingest").mkdir(parents=True)
    (root / "package.json").write_text('{"name": "r", "version": "0.0.1"}\n')
    (root / "pyproject.toml").write_text('[project]\nname = "r"\nversion = "0.0.1"\n')
    (root / "packages" / "node" / "package.json").write_text('{"name": "n", "version": "0.0.1"}\n')
    return root


def test_plan_always_bumps_root(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    plan = vm.plan_bumps(root, [])
    scopes = {(b.manifest.scope, str(b.manifest.path.name)) for b in plan}
    assert ("root", "package.json") in scopes
    assert ("root", "pyproject.toml") in scopes
    assert all(b.old == "0.0.1" and b.new == "0.0.2" for b in plan)


def test_plan_bumps_only_touched_package(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    plan = vm.plan_bumps(root, ["packages/node/ats/x.ts", "README.md"])
    by_scope = {b.manifest.scope for b in plan}
    assert by_scope == {"root", "packages/node"}


def test_plan_ignores_manifest_less_package(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path)
    plan = vm.plan_bumps(root, ["packages/ingest/src/a.py"])
    assert {b.manifest.scope for b in plan} == {"root"}


def test_plan_skips_unparseable_version(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _fake_repo(tmp_path)
    (root / "packages" / "node" / "package.json").write_text('{"name": "n", "version": "1.2"}\n')
    plan = vm.plan_bumps(root, ["packages/node/a.ts"])
    assert {b.manifest.scope for b in plan} == {"root"}
    assert "skipping" in capsys.readouterr().err


# -- end to end inside a real git repo ------------------------------------------


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        [GIT or "git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


@needs_git
def test_e2e_touched_package_bumps_and_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    root = _fake_repo(tmp_path)
    (root / "packages" / "ml").mkdir()
    (root / "packages" / "ml" / "pyproject.toml").write_text(
        '[project]\nname = "m"\nversion = "0.0.1"\n'
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")

    (root / "packages" / "node" / "ats").mkdir()
    (root / "packages" / "node" / "ats" / "x.ts").write_text("1")
    _git(root, "add", "packages/node/ats/x.ts")

    monkeypatch.chdir(root)
    assert vm.main([]) == 0
    out = capsys.readouterr().out
    assert "packages/node" in out and "0.0.1 -> 0.0.2" in out

    assert json.loads((root / "package.json").read_text())["version"] == "0.0.2"
    assert (
        json.loads((root / "packages" / "node" / "package.json").read_text())["version"] == "0.0.2"
    )
    assert "0.0.1" in (root / "packages" / "ml" / "pyproject.toml").read_text()

    staged = _git(root, "diff", "--cached", "--name-only")
    assert "package.json" in staged
    assert "packages/node/package.json" in staged


@needs_git
def test_e2e_root_only_change_bumps_just_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_repo(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")

    (root / "notes.md").write_text("hi")
    _git(root, "add", "notes.md")

    monkeypatch.chdir(root)
    assert vm.main([]) == 0
    assert json.loads((root / "package.json").read_text())["version"] == "0.0.2"
    assert (
        json.loads((root / "packages" / "node" / "package.json").read_text())["version"] == "0.0.1"
    )


@needs_git
def test_e2e_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    root = _fake_repo(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")

    monkeypatch.chdir(root)
    assert vm.main(["--dry-run"]) == 0
    assert "would bump" in capsys.readouterr().out
    assert json.loads((root / "package.json").read_text())["version"] == "0.0.1"
    assert _git(root, "status", "--porcelain") == ""
