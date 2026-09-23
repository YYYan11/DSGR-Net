#!/usr/bin/env python3
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".pkl", ".xlsx", ".exe", ".out", ".wth", ".sol", ".cox", ".coa", ".cot"}
FORBIDDEN_PARTS = {"wl-dssat", "dssat2023", "__pycache__", ".pytest_cache", ".ds_store"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".tex", ".cff", ".txt", ".csv", ".gitignore"}
SENSITIVE = re.compile(
    r"/" + r"home/|/" + r"Users/|192" + r"\.168\.\d+\.\d+|"
    r"password\s*=|api[_-]?key\s*=|secret\s*=",
    re.I,
)


def main():
    errors = []
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(ROOT)
        lower_parts = {part.lower() for part in relative.parts}
        if lower_parts & FORBIDDEN_PARTS:
            errors.append(f"forbidden path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file type: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8", errors="replace")
            if SENSITIVE.search(text):
                errors.append(f"sensitive absolute path or credential marker: {relative}")
    source = ROOT / "src/dsgr_net.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = "945ed86a070a7c8de638fcf01e03f746bf2e50a71083c8e848769ddb273687a7"
    if digest != expected:
        errors.append(f"formal V16 source hash changed: {digest}")
    if errors:
        print("RELEASE AUDIT FAILED")
        print("\n".join(f"- {item}" for item in errors))
        return 1
    print(f"RELEASE AUDIT PASSED: {len(files)} files")
    print(f"formal V16 SHA256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
