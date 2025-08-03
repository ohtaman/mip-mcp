#!/usr/bin/env python3
"""Test the Pyodide path detection fix."""

import asyncio
from pathlib import Path

# Import the updated executor
from src.mip_mcp.executor.pyodide_executor import PyodideExecutor


async def test_pyodide_detection():
    """Test the new Pyodide detection logic."""
    print("=== Testing Pyodide Detection Fix ===")

    # Create executor instance
    config = {"execution_timeout": 60.0, "progress_interval": 10.0}
    executor = PyodideExecutor(config)

    # Test bundled Pyodide detection
    print("Testing bundled Pyodide detection...")
    bundled_path = executor._find_bundled_pyodide()
    print(f"Bundled Pyodide path: {bundled_path}")

    if bundled_path:
        print(f"✓ Found bundled Pyodide: {bundled_path}")
        # Verify the file exists
        if Path(bundled_path).exists():
            print("✓ File exists and is accessible")
        else:
            print("✗ File path returned but file doesn't exist")
    else:
        print("✗ No bundled Pyodide found")

    # Test full path detection
    print("\nTesting full path detection...")
    try:
        full_path = await executor._find_pyodide_path()
        if full_path:
            print(f"✓ Full detection successful: {full_path}")
        else:
            print("✗ Full detection failed")
    except Exception as e:
        print(f"✗ Full detection error: {e}")


if __name__ == "__main__":
    asyncio.run(test_pyodide_detection())
