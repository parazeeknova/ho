"""Changeset-style automatic version manager.

Version scheme is base-100 semver: every bump increments the patch part, and
a part rolls over into the next one once it passes 99::

    0.0.99 -> 0.1.0 -> 0.1.1 -> ... -> 0.1.99 -> 0.2.0 -> ... -> 0.99.99 -> 1.0.0

Behavior (wired into the ``pre-commit`` hook so it runs on every commit):

* root ``package.json`` and root ``pyproject.toml`` are bumped on EVERY run.
* a package manifest (``packages/*/package.json``, ``packages/*/pyproject.toml``
  or ``apps/*/package.json``) is bumped only when the staged commit touches
  files under that scope directory. Scopes without a manifest
  (e.g. ``packages/ingest``) only contribute to the root bump.

Bumped manifests are staged with ``git add`` so they land in the same commit.
When a ``pyproject.toml`` version moves, the matching ``[[package]]`` entry in
``uv.lock`` is synced to the same version (otherwise ``uv run`` would dirty the
tree right after every commit).
The tool never fails the commit: unreadable files or unparseable versions
produce a warning and are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_PART = 99

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_PYPROJECT_VERSION_RE = re.compile(r"""version\s*=\s*(['"])([^'"]+)\1""")

PACKAGE_JSON = "package.json"
PYPROJECT_TOML = "pyproject.toml"


@dataclass
class Manifest:
    """A versioned manifest file. ``scope`` is ``root`` or ``packages/<name>``."""

    path: Path
    kind: str  # "package-json" | "pyproject"
    scope: str


@dataclass
class Bump:
    manifest: Manifest
    old: str
    new: str


def parse_version(raw: str) -> tuple[int, int, int]:
    """Parse ``major.minor.patch`` into ints, raising ValueError if malformed."""
    match = _VERSION_RE.match(raw.strip())
    if not match:
        raise ValueError(f"not a base-100 semver version: {raw!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump_version(raw: str) -> str:
    """Bump one step with base-100 carry (``0.0.99`` -> ``0.1.0``)."""
    major, minor, patch = parse_version(raw)
    patch += 1
    if patch > MAX_PART:
        patch = 0
        minor += 1
    if minor > MAX_PART:
        minor = 0
        major += 1
    return f"{major}.{minor}.{patch}"


def repo_root(start: Path) -> Path:
    """Best-effort repo root: git toplevel, falling back to ``start``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except subprocess.CalledProcessError, FileNotFoundError:
        return start


def staged_files(root: Path) -> list[str]:
    """POSIX-style paths staged in the index (empty list when git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError, FileNotFoundError:
        return []
    return [p for p in out.stdout.split("\0") if p]


SCOPE_DIRS = ("packages", "apps")


def package_dirs(root: Path) -> list[str]:
    """Top-level scopes, e.g. ``packages/node`` or ``apps/web`` (sorted, POSIX)."""
    scopes: list[str] = []
    for scope_dir in SCOPE_DIRS:
        base = root / scope_dir
        if not base.is_dir():
            continue
        scopes.extend(f"{scope_dir}/{entry.name}" for entry in base.iterdir() if entry.is_dir())
    return sorted(scopes)


def owning_scope(path: str, scopes: list[str]) -> str | None:
    """Longest-prefix scope owning ``path`` (``None`` for root-level files)."""
    owned = [scope for scope in scopes if path == scope or path.startswith(scope + "/")]
    return max(owned, key=len, default=None)


def manifests_in(directory: Path, scope: str) -> list[Manifest]:
    """Versioned manifests sitting directly inside ``directory``."""
    found: list[Manifest] = []
    pkg_json = directory / PACKAGE_JSON
    if pkg_json.is_file() and _read_package_json_version(pkg_json) is not None:
        found.append(Manifest(path=pkg_json, kind="package-json", scope=scope))
    pyproject = directory / PYPROJECT_TOML
    if pyproject.is_file() and _read_pyproject_version(pyproject) is not None:
        found.append(Manifest(path=pyproject, kind="pyproject", scope=scope))
    return found


def discover_manifests(root: Path) -> list[Manifest]:
    """Root manifests first, then one entry per package manifest found on disk."""
    manifests = manifests_in(root, "root")
    for scope in package_dirs(root):
        manifests.extend(manifests_in(root / scope, scope))
    return manifests


def _read_package_json_version(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return version if isinstance(version, str) else None


def _read_pyproject_field(path: Path, field: str) -> str | None:
    pattern = re.compile(rf"""{field}\s*=\s*(['"])([^'"]+)\1""")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    in_project = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = pattern.match(stripped)
            if match:
                return match.group(2)
    return None


def _read_pyproject_version(path: Path) -> str | None:
    return _read_pyproject_field(path, "version")


def read_version(manifest: Manifest) -> str | None:
    if manifest.kind == "package-json":
        return _read_package_json_version(manifest.path)
    return _read_pyproject_version(manifest.path)


def write_package_json_version(path: Path, version: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_pyproject_version(path: Path, version: str) -> None:
    """Rewrite only the ``[project]`` version line, preserving everything else."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_project = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if in_project and _PYPROJECT_VERSION_RE.match(stripped):
            indent = line[: len(line) - len(line.lstrip())]
            tail = ""
            match = _PYPROJECT_VERSION_RE.match(stripped)
            assert match is not None
            rest = stripped[match.end() :]
            if "#" in rest:
                tail = "  " + rest[rest.index("#") :]
            lines[i] = f'{indent}version = "{version}"{tail}\n'
            break
    else:
        raise ValueError(f"no [project] version found in {path}")
    path.write_text("".join(lines), encoding="utf-8")


def write_version(manifest: Manifest, version: str) -> None:
    if manifest.kind == "package-json":
        write_package_json_version(manifest.path, version)
    else:
        write_pyproject_version(manifest.path, version)


def plan_bumps(root: Path, staged: list[str]) -> list[Bump]:
    """Root manifests always bump; package manifests bump only when touched."""
    scopes = package_dirs(root)
    touched = {scope for path in staged if (scope := owning_scope(path, scopes)) is not None}
    plan: list[Bump] = []
    for manifest in discover_manifests(root):
        if manifest.scope != "root" and manifest.scope not in touched:
            continue
        old = read_version(manifest)
        if old is None:
            print(f"warn: skipping {manifest.path}: no readable version", file=sys.stderr)
            continue
        try:
            plan.append(Bump(manifest=manifest, old=old, new=bump_version(old)))
        except ValueError as exc:
            print(f"warn: skipping {manifest.path}: {exc}", file=sys.stderr)
    return plan


def stage_files(root: Path, rels: list[str]) -> None:
    for rel in rels:
        try:
            subprocess.run(["git", "add", "--", rel], cwd=root, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"warn: could not stage {rel}: {exc}", file=sys.stderr)


def sync_uv_lock(root: Path, project_name: str, version: str) -> bool:
    """Sync matching ``[[package]]`` version entries in ``uv.lock``.

    Only bare ``version = ...`` lines inside the named package block are
    touched; inline ``{ name = ... }`` dependency specs and ``[package.*]``
    metadata subtables are left alone. Returns True when the file changed.
    """
    lock = root / "uv.lock"
    try:
        lines = lock.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return False
    name_pattern = re.compile(rf'name\s*=\s*"{re.escape(project_name)}"')
    in_target = False
    changed = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            in_target = False
            continue
        if re.match(r"name\s*=", stripped):
            in_target = name_pattern.match(stripped) is not None
            continue
        if in_target and re.match(r"version\s*=", stripped):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f'{indent}version = "{version}"\n'
            changed = True
            in_target = False
    if changed:
        lock.write_text("".join(lines), encoding="utf-8")
    return changed


def apply_plan(root: Path, plan: list[Bump], *, stage: bool) -> None:
    for bump in plan:
        try:
            write_version(bump.manifest, bump.new)
        except (OSError, ValueError) as exc:
            print(f"warn: could not write {bump.manifest.path}: {exc}", file=sys.stderr)
            continue
        rel = str(bump.manifest.path.relative_to(root))
        print(f"bumped {rel} ({bump.manifest.scope}): {bump.old} -> {bump.new}")
        if stage:
            stage_files(root, [rel])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bump root + touched-package versions.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the bump plan without writing or staging anything",
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="write version files but do not git-add them",
    )
    args = parser.parse_args(argv)

    root = repo_root(Path.cwd())
    plan = plan_bumps(root, staged_files(root))
    if not plan:
        print("version-manager: nothing to bump")
        return 0
    if args.dry_run:
        for bump in plan:
            rel = bump.manifest.path.relative_to(root)
            print(f"would bump {rel} ({bump.manifest.scope}): {bump.old} -> {bump.new}")
        return 0
    apply_plan(root, plan, stage=not args.no_stage)
    for bump in plan:
        if bump.manifest.kind != "pyproject":
            continue
        project_name = _read_pyproject_field(bump.manifest.path, "name")
        if project_name and sync_uv_lock(root, project_name, bump.new):
            print(f"synced uv.lock ({project_name}): -> {bump.new}")
            if not args.no_stage:
                stage_files(root, ["uv.lock"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
