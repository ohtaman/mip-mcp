#!/usr/bin/env python3
"""Test uvx environment simulation."""

import sys
from unittest.mock import patch

# Import the updated executor
from src.mip_mcp.executor.pyodide_executor import PyodideExecutor


def test_uvx_simulation():
    """Test uvx path detection with simulated environment."""
    print("=== Testing UVX Environment Simulation ===")

    # Simulate uvx environment by patching sys.executable
    fake_uvx_python = (
        "/Users/mitsuhisa.ota/.cache/uv/archive-v0/cxlMGn9Yqz5QZ8zLiGoN8/bin/python"
    )

    with patch.object(sys, "executable", fake_uvx_python):
        config = {"execution_timeout": 60.0, "progress_interval": 10.0}
        executor = PyodideExecutor(config)

        print(f"Simulated Python executable: {sys.executable}")

        # Test bundled Pyodide detection
        bundled_path = executor._find_bundled_pyodide()
        print(f"Detected Pyodide path: {bundled_path}")

        if bundled_path:
            expected_path = "/Users/mitsuhisa.ota/.cache/uv/archive-v0/cxlMGn9Yqz5QZ8zLiGoN8/mip_mcp/pyodide/pyodide.js"
            if bundled_path == expected_path:
                print("✓ Correct uvx path detected!")
            else:
                print(f"✗ Wrong path. Expected: {expected_path}")
        else:
            print("✗ No path detected in uvx simulation")


if __name__ == "__main__":
    test_uvx_simulation()
