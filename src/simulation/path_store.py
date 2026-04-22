"""In-memory path store keyed by (src_as, dst_as) for the evaluation pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class InMemoryPathStore:
    """Stores precomputed path lists between AS pairs."""

    def __init__(self) -> None:
        self._paths: Dict[Tuple[int, int], List[Any]] = {}

    def set_paths(self, src_as: int, dst_as: int, paths: List[Any]) -> None:
        self._paths[(int(src_as), int(dst_as))] = list(paths)

    def find_paths(self, src_as: int, dst_as: int) -> List[Any]:
        return list(self._paths.get((int(src_as), int(dst_as)), []))
