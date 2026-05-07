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

    def save(self, filepath: str) -> None:
        import json
        with open(filepath, "w") as f:
            dump_dict = {f"{k[0]}-{k[1]}": v for k, v in self._paths.items()}
            json.dump(dump_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "InMemoryPathStore":
        import json
        store = cls()
        with open(filepath, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            src_as, dst_as = k.split("-")
            store.set_paths(int(src_as), int(dst_as), v)
        return store
