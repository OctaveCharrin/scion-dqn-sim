"""Compatibility shim: re-export v2 path finder types under legacy names."""

from .pathfinder_v2 import PathFinderV2 as PathFinder, SCIONPath as Path

__all__ = ["PathFinder", "Path"]
