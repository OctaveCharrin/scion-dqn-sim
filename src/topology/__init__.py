"""
Topology layer — BRITE integration, SCION conversion, and native generators.

Modules
-------
``brite_cfg_gen``
    BRITE ``.conf`` generation and JAR invocation.
``brite2scion_converter``
    BRITE ``.brite`` → evaluation graph / SCION-style metadata.
``topology_geo``
    Shared k-means ISD assignment, core ring connectivity, latency helpers, and
    geography PNG export (used by BRITE paths).
    Pure-Python ``TopDownSCIONGenerator`` (no BRITE JAR).
"""
