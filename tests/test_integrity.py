"""Integrity tests.

These encode the claims this repository makes about itself:

* every SHA-256 manifest, at the root or beside a sub-analysis, still
  accounts for the file it hashes, so the plans and analysis code are
  byte-identical to a recorded state;
* committed results parse;
* no personal filesystem paths have leaked in.

Amendment chains are expected. A plan that was amended mid-study is hashed
twice: once as first written, once after the amendment. A file therefore
passes if it matches ANY manifest that hashes it, and the test reports which
link of the chain it matched. Matching none means the file changed without a
record, which is the thing worth catching.

Standard library only, so CI needs nothing but pytest.

    pytest -v
"""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# A manifest may hash a file under a name it no longer has. boltzbench's
# PLAN.sha256 hashes the v1 plan as "README.md"; that file is preserved byte
# for byte as PLAN_V1_README.md so the hash still reproduces.
MANIFEST_RENAMES = {"PLAN.sha256": {"README.md": "PLAN_V1_README.md"}}

# Files named in a manifest that are deliberately not in the public tree.
# Keep the hash line: deleting it would hide that the file ever existed.
ALLOWED_MISSING = {"launch_v2.sh"}

# Amended files whose post-amendment hash was never recorded. The amendment
# itself is documented in the script header and in the matching RESULTS file.
# Remove an entry here once its amendment manifest exists.
UNRECORDED_AMENDMENTS = {
    "alphamissense/am_metabolic.py":
        "amended 2026-10-23 (Amendment 1, documented in the file header and "
        "RESULTS_AM.md, both runs kept); no am_metabolic_plan_amend1.sha256 was written",
}

# Container paths from the environment the analyses ran in. They are left in
# place on purpose: several of these scripts are hashed by a plan manifest, so
# rewriting their paths would break the record of what was actually run.
KNOWN_ENVIRONMENT_PREFIXES = ("/home/claude/", "/mnt/user-data/")

PERSONAL_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
CREDENTIAL_SHAPED = re.compile(r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")

# Protein sequences can match credential patterns by coincidence. Checked by hand.
SECRET_SCAN_EXEMPT = {"v2_sequences.json"}

SKIP_DIRS = {".git", ".github", "__pycache__", ".venv", "node_modules",
             "logs", "runs", "inputs", "inputs_v2", "weights", "emb", "cells", "data"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".sh", ".yml", ".yaml", ".toml", ".cfg"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_manifests():
    out = []
    for pattern in ("**/*.sha256", "**/*sha256*.txt"):
        for p in ROOT.glob(pattern):
            if p.is_file() and not any(d in SKIP_DIRS for d in p.relative_to(ROOT).parts):
                out.append(p)
    return sorted(set(out))


def resolve(manifest, name):
    """A manifest names a file either beside itself or relative to the root."""
    for candidate in (manifest.parent / name, ROOT / name):
        if candidate.exists():
            return candidate
    return None


def build_chains():
    """target -> {digest: [manifest names that record it]}"""
    chains = defaultdict(lambda: defaultdict(list))
    missing = []
    for manifest in find_manifests():
        renames = MANIFEST_RENAMES.get(manifest.name, {})
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            digest, _, name = line.partition("  ")
            if not digest or not name:
                continue
            name = renames.get(name, name)
            target = resolve(manifest, name)
            if target is None:
                if name not in ALLOWED_MISSING:
                    missing.append(f"{name}: recorded in {manifest.name}, not in the tree")
                continue
            chains[target][digest].append(manifest.name)
    return chains, missing


CHAINS, MISSING = build_chains()
TARGETS = sorted(CHAINS, key=lambda p: str(p))


def test_every_manifest_entry_resolves():
    assert not MISSING, "\n" + "\n".join(MISSING)


def test_manifests_found():
    assert TARGETS, "no SHA-256 manifest found anywhere in the tree"


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: str(p.relative_to(ROOT)))
def test_hashed_file_matches_its_chain(target):
    """The file matches one of the hashes recorded for it."""
    rel = str(target.relative_to(ROOT))
    recorded = CHAINS[target]
    actual = sha256(target)
    if actual in recorded:
        return
    if rel in UNRECORDED_AMENDMENTS:
        pytest.skip(f"{rel}: {UNRECORDED_AMENDMENTS[rel]}")
    chain = "; ".join(f"{d[:12]} ({', '.join(m)})" for d, m in recorded.items())
    pytest.fail(f"{rel}\n  actual   {actual}\n  recorded {chain}")


def text_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts[:-1]):
            continue
        if p.suffix.lower() in TEXT_SUFFIXES:
            yield p


def test_no_personal_paths_committed():
    """No home-directory path leaked in, beyond the documented run environment."""
    hits = []
    for p in text_files():
        try:
            body = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in PERSONAL_PATH.finditer(body):
            if m.group(0).startswith(KNOWN_ENVIRONMENT_PREFIXES):
                continue
            hits.append(f"{p.relative_to(ROOT)}: {m.group(0)}")
    assert not hits, "personal paths committed:\n" + "\n".join(sorted(set(hits))[:20])


def test_no_credential_shaped_strings():
    hits = []
    for p in text_files():
        if p.name in SECRET_SCAN_EXEMPT:
            continue
        try:
            body = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if CREDENTIAL_SHAPED.search(body):
            hits.append(str(p.relative_to(ROOT)))
    assert not hits, "credential-shaped strings in:\n" + "\n".join(hits)


def test_declared_results_parse():
    tables = sorted(ROOT.glob("results*.csv")) + sorted(ROOT.glob("**/results/**/*.csv"))
    if not tables:
        pytest.skip("no committed CSV results")
    for t in tables[:40]:
        with open(t, newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.reader(fh))
        assert len(rows) >= 2, f"{t.relative_to(ROOT)} has no data rows"


def test_declared_json_parses():
    blobs = sorted(ROOT.glob("*.json")) + sorted(ROOT.glob("**/results/**/*.json"))
    if not blobs:
        pytest.skip("no committed JSON")
    for b in blobs[:40]:
        with open(b, encoding="utf-8") as fh:
            json.load(fh)


def test_licence_and_readme_present():
    for required in ("README.md", "LICENSE"):
        assert (ROOT / required).exists(), f"{required} missing"
