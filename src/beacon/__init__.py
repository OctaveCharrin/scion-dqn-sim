"""
Beacon Simulator Module

High-performance SCION beaconing simulation using GraphBLAS.
Implements:
- Core beaconing with power-series iteration
- Intra-ISD beaconing with BFS
- Efficient segment storage
"""

from .beacon_sim import BeaconSimulator

__all__ = ['BeaconSimulator']