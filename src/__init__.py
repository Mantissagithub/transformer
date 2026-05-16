from __future__ import annotations

import ctypes
import site
from pathlib import Path


def _preload_torch_cuda_libs() -> None:
    """Load NVIDIA wheel-shipped shared libs before importing torch."""

    lib_dirs: list[Path] = []
    seen: set[Path] = set()
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        root = Path(base) / "nvidia"
        if not root.exists():
            continue
        for lib_dir in sorted(root.glob("*/lib")):
            resolved = lib_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                lib_dirs.append(resolved)

    for lib_dir in lib_dirs:
        for lib_path in sorted(lib_dir.glob("*.so*")):
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


_preload_torch_cuda_libs()

from . import components  # noqa: F401  - populate registries on import
