"""Content-based run identity without reading credentials or environment variables."""

import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def build_identity() -> dict:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    dependencies = {}
    for package in ("typesafe-sdk", "gym-super-mario-bros", "nes-py", "gymnasium"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {
        "source_sha256": digest.hexdigest(),
        "source_scope": "typesafe_mario/*.py",
        "python": platform.python_version(),
        "dependencies": dependencies,
    }
