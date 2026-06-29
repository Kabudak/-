"""MoE comparison entrypoint for HDFS-streamed production training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_production_hdfs import main


def _ensure_default_arg(name: str, value: str) -> None:
    if name not in sys.argv:
        sys.argv.extend([name, value])


if __name__ == "__main__":
    _ensure_default_arg("--ffn-type", "moe")
    _ensure_default_arg("--output-dir", str(PROJECT_ROOT / "outputs" / "production_hdfs_moe"))
    main()
