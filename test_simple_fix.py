#!/usr/bin/env python3
"""Test simple fix for uvx Pyodide detection."""

import sys
from pathlib import Path

# Add the mip_mcp package to path for testing
sys.path.insert(
    0, "/Users/mitsuhisa.ota/.local/share/uv/tools/mip-mcp/lib/python3.12/site-packages"
)


def find_pyodide_simple():
    """Simple method to find Pyodide using Python."""

    # Method 1: Check for uvx installation
    python_exe = Path(sys.executable)
    if "uv/tools" in str(python_exe):
        # For uvx: /path/to/uv/tools/mip-mcp/bin/python -> /path/to/uv/tools/mip-mcp
        tool_root = python_exe.parent.parent
        pyodide_js = tool_root / "mip_mcp" / "pyodide" / "pyodide.js"
        if pyodide_js.exists():
            print(f"✓ Found uvx Pyodide: {pyodide_js}")
            return str(pyodide_js)

    # Method 2: Standard package installation
    try:
        import mip_mcp.executor.pyodide_executor as pe_module

        module_path = Path(pe_module.__file__).resolve()
        package_dir = module_path.parent.parent  # .../mip_mcp/executor -> .../mip_mcp
        pyodide_js = package_dir / "pyodide" / "pyodide.js"
        if pyodide_js.exists():
            print(f"✓ Found standard Pyodide: {pyodide_js}")
            return str(pyodide_js)
    except ImportError:
        pass

    # Method 3: Development fallback
    cwd = Path.cwd()
    node_modules_pyodide = cwd / "node_modules" / "pyodide" / "pyodide.js"
    if node_modules_pyodide.exists():
        print(f"✓ Found development Pyodide: {node_modules_pyodide}")
        return str(node_modules_pyodide)

    print("✗ Pyodide not found")
    return None


if __name__ == "__main__":
    print("=== Simple Pyodide Detection Test ===")
    result = find_pyodide_simple()
    if result:
        print(f"SUCCESS: {result}")
    else:
        print("FAILED: No Pyodide found")
