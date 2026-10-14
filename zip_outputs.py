"""
Package and zip all pipeline outputs, logs, metrics, submissions, and evaluation summaries
into a single compressed archive (argon_outputs_bundle.zip) for inspection.
"""

import os
import zipfile
from pathlib import Path


def create_output_archive(output_zip: str = "argon_outputs_bundle.zip") -> str:
    cwd = Path.cwd()
    print(f"[*] Scanning workspace for pipeline outputs in: {cwd}")

    patterns_and_dirs = [
        "submission.csv",
        "submissions",
        "outputs",
        "PROJECT_STATE.md",
    ]

    files_to_add: list[Path] = []

    # Target specific outputs
    for item in patterns_and_dirs:
        p = cwd / item
        if p.is_file():
            files_to_add.append(p)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and not f.name.endswith(".zip"):
                    files_to_add.append(f)

    # Add all log files
    for log_file in cwd.glob("*.log"):
        if log_file.is_file():
            files_to_add.append(log_file)

    # Deduplicate while preserving order
    unique_files = sorted(set(files_to_add))

    if not unique_files:
        print("[!] No output files discovered yet. Have you executed run_argon_cascade.py?")
        return ""

    zip_path = cwd / output_zip
    print(f"[*] Archiving {len(unique_files)} output file(s) into '{zip_path.name}'...")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in unique_files:
            rel_path = f.relative_to(cwd)
            zf.write(f, arcname=str(rel_path))
            size_kb = f.stat().st_size / 1024
            print(f"  + Added: {rel_path} ({size_kb:.1f} KB)")

    total_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print(f"[✓] Archive successfully created: {zip_path.resolve()}")
    print(f"[✓] Archive size: {total_size_mb:.2f} MB")
    print("=" * 70)
    return str(zip_path)


if __name__ == "__main__":
    create_output_archive()
