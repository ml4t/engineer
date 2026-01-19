# mypy: disable-error-code="import-untyped"
"""Validation tests comparing ml4t.engineer.labeling against mlfinpy.

These tests validate ml4t implementations against the open-source mlfinpy
package, which provides reference implementations of De Prado's algorithms.

Requirements:
    pip install mlfinpy  (requires separate venv due to numba version conflict)
"""
