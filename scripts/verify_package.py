"""Verify the frozen package without accessing the parent research project."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "REPRODUCIBILITY_MANIFEST.json"
CHECKSUMS = ROOT / "CHECKSUMS_SHA256.txt"
BANNED_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".ipynb_checkpoints", ".vscode", ".idea",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".cff", ".toml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail_if(condition: bool, message: str) -> None:
    if condition:
        raise RuntimeError(message)


def verify_checksums() -> int:
    rows = {}
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        rows[relative] = digest
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*") if path.is_file() and path != CHECKSUMS
    }
    fail_if(set(rows) != expected, "Checksum ledger file set does not match package file set")
    for relative, expected_digest in rows.items():
        fail_if(sha256(ROOT / relative) != expected_digest, f"Checksum mismatch: {relative}")
    return len(rows)


def verify_manifest() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {row["relative_path"]: row for row in payload["artifacts"]}
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
    fail_if(set(records) != actual, "Manifest file set does not match package file set")
    for relative, row in records.items():
        if relative in {MANIFEST.name, CHECKSUMS.name}:
            continue
        path = ROOT / relative
        fail_if(path.stat().st_size != row["file_size_bytes"], f"Manifest size mismatch: {relative}")
        fail_if(sha256(path) != row["sha256"], f"Manifest checksum mismatch: {relative}")
    return len(records)


def verify_security_and_portability() -> dict:
    bad_directories = [
        path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")
        if path.is_dir() and path.name in BANNED_DIRECTORIES
    ]
    fail_if(bool(bad_directories), f"Development clutter present: {bad_directories}")
    fail_if(any(path.name == ".env" for path in ROOT.rglob("*")), ".env file present")

    absolute_path_pattern = re.compile(
        r"(?:[A-Za-z]:\\(?:[^\\\s]+\\){2,}|/home/[^/\s]+/|/Users/[^/\s]+/)"
    )
    secret_assignment_pattern = re.compile(
        r"(?i)(?:api[_-]?key|password|passwd|access[_-]?token|client[_-]?secret|"
        r"private[_-]?key|aws_access_key_id)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    )
    private_key_pattern = re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")
    email_pattern = re.compile(r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}")
    findings = []
    scanned_text_files = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            # This verifier necessarily contains the detection expressions.
            continue
        scanned_text_files += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in (
            absolute_path_pattern, secret_assignment_pattern, private_key_pattern, email_pattern
        )):
            findings.append(path.relative_to(ROOT).as_posix())

    checkpoint_findings = []
    checkpoints = list((ROOT / "models").rglob("selected_validation_checkpoint.zip"))
    for path in checkpoints:
        with zipfile.ZipFile(path) as archive:
            for member in ("data", "system_info.txt", "_stable_baselines3_version"):
                if member not in archive.namelist():
                    continue
                text = archive.read(member).decode("utf-8", errors="replace")
                if any(pattern.search(text) for pattern in (
                    absolute_path_pattern, private_key_pattern, email_pattern
                )):
                    checkpoint_findings.append(f"{path.relative_to(ROOT).as_posix()}::{member}")
    fail_if(bool(findings), f"Security/path candidates in text files: {findings}")
    fail_if(bool(checkpoint_findings), "Security/path candidate found inside checkpoint archive")
    return {
        "text_files_scanned": scanned_text_files,
        "checkpoint_archives_scanned": len(checkpoints),
        "findings": 0,
    }


def verify_scientific_outputs() -> dict:
    with (ROOT / "results" / "REPRODUCTION_VALIDATION.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        validation = list(csv.DictReader(stream))
    fail_if(not validation, "Reproduction validation is empty")
    failed = [row for row in validation if row["Pass"].lower() != "true"]
    fail_if(bool(failed), f"Reproduction validation contains {len(failed)} failures")

    summary = json.loads(
        (ROOT / "results" / "reproduced" / "REPRODUCTION_SUMMARY.json").read_text(encoding="utf-8")
    )
    fail_if(not summary["passed"], "Reproduction summary is not PASS")
    with (ROOT / "models" / "MODEL_MANIFEST.csv").open(newline="", encoding="utf-8") as stream:
        models = list(csv.DictReader(stream))
    fail_if(len(models) != 30, f"Expected 30 selected checkpoints, found {len(models)}")
    required_tables = {
        "TABLE_MAIN_CONTROLLER_RESULTS.csv", "TABLE_IT_SCENARIO_CHARACTERISTICS.csv",
        "TABLE_IT_SCENARIO_RESULTS.csv", "TABLE_TRUE_ABLATION_RESULTS.csv",
        "TABLE_SEED_RESULTS.csv", "TABLE_SAFETY_RESULTS.csv",
    }
    actual_tables = {path.name for path in (ROOT / "tables").glob("*.csv")}
    fail_if(not required_tables.issubset(actual_tables), "One or more required final tables are missing")
    generated_figures = list((ROOT / "results" / "reproduced" / "figures").glob("*"))
    fail_if(len(generated_figures) != 20, f"Expected 20 regenerated figure files, found {len(generated_figures)}")
    return {
        "metric_comparisons_passed": len(validation),
        "selected_checkpoints": len(models),
        "required_tables": len(required_tables),
        "regenerated_figure_files": len(generated_figures),
        "training_performed": bool(summary["training_performed"]),
    }


def main() -> None:
    report = {
        "manifest_records_verified": verify_manifest(),
        "checksum_entries_verified": verify_checksums(),
        "security_and_portability": verify_security_and_portability(),
        "scientific_outputs": verify_scientific_outputs(),
        "external_project_access_required": False,
        "passed": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
