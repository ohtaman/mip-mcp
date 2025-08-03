#!/usr/bin/env python3
"""Debug script to understand uvx path resolution."""

import asyncio
import sys
from pathlib import Path

# Add the mip_mcp package to path for testing
sys.path.insert(
    0, "/Users/mitsuhisa.ota/.local/share/uv/tools/mip-mcp/lib/python3.12/site-packages"
)


async def debug_path_detection():
    """Debug the path detection issue."""

    print("=== UVX Path Detection Debug ===")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {Path.cwd()}")

    # Check where we expect Pyodide files to be
    uvx_pyodide_dir = Path(
        "/Users/mitsuhisa.ota/.local/share/uv/tools/mip-mcp/mip_mcp/pyodide"
    )
    print(f"\nExpected Pyodide location: {uvx_pyodide_dir}")
    print(f"Pyodide directory exists: {uvx_pyodide_dir.exists()}")
    if uvx_pyodide_dir.exists():
        print(f"Pyodide files: {list(uvx_pyodide_dir.glob('*'))}")

    # Test the current Node.js script logic
    print("\n=== Testing Current Node.js Script Logic ===")

    node_script = """
const path = require('path');
const fs = require('fs');

console.log('Node.js __dirname:', __dirname);
console.log('Node.js process.cwd():', process.cwd());

// Current bundled paths from the executor
const bundledPaths = [
    path.join(__dirname, '..', '..', '..', 'mip_mcp', 'pyodide', 'pyodide.js'),
    path.join(__dirname, '..', '..', 'mip_mcp', 'pyodide', 'pyodide.js'),
    path.join(process.cwd(), 'node_modules', 'pyodide', 'pyodide.js')
];

console.log('\\nChecking paths:');
for (let i = 0; i < bundledPaths.length; i++) {
    const resolvedPath = path.resolve(bundledPaths[i]);
    const exists = fs.existsSync(bundledPaths[i]);
    console.log(`${i + 1}. ${bundledPaths[i]}`);
    console.log(`   Resolved: ${resolvedPath}`);
    console.log(`   Exists: ${exists}`);
}
    """

    proc = await asyncio.create_subprocess_exec(
        "node",
        "-e",
        node_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()

    print("Node.js output:")
    print(stdout.decode())
    if stderr.decode():
        print("Node.js errors:")
        print(stderr.decode())

    # Test Python module detection
    print("\n=== Testing Python Module Detection ===")
    try:
        # Test import availability without actually importing the class
        import importlib.util

        spec = importlib.util.find_spec("mip_mcp.executor.pyodide_executor")
        if spec is not None:
            print("✓ Successfully found PyodideExecutor module")

        # Check where the module is located
        import mip_mcp.executor.pyodide_executor as pe_module

        module_path = Path(pe_module.__file__).resolve()
        print(f"PyodideExecutor module location: {module_path}")

        # Check relative path to pyodide directory
        executor_dir = module_path.parent  # .../mip_mcp/executor
        package_dir = executor_dir.parent  # .../mip_mcp
        expected_pyodide = package_dir / "pyodide" / "pyodide.js"
        print(f"Expected Pyodide from module: {expected_pyodide}")
        print(f"Exists: {expected_pyodide.exists()}")

    except ImportError as e:
        print(f"✗ Failed to import: {e}")


if __name__ == "__main__":
    asyncio.run(debug_path_detection())
