"""Simple build hook to run npm install for Pyodide bundling."""

import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class NpmInstallBuildHook(BuildHookInterface):
    """Build hook that runs npm install to download Pyodide for bundling."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Run npm install to download Pyodide files for bundling."""
        build_dir = Path(self.root)
        package_json = build_dir / "package.json"
        node_modules = build_dir / "node_modules"

        # Check if package.json exists
        if not package_json.exists():
            print("⚠ No package.json found - skipping npm install")
            return

        # Check if node_modules already exists and is populated
        pyodide_dir = node_modules / "pyodide"
        if pyodide_dir.exists() and (pyodide_dir / "pyodide.js").exists():
            print("✓ Pyodide already installed in node_modules")
            return

        print("📦 Running npm install to download Pyodide for bundling...")

        try:
            # Run npm install
            result = subprocess.run(
                ["npm", "install"],
                cwd=build_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                print("✓ npm install completed successfully")
                # Verify Pyodide was installed
                if (pyodide_dir / "pyodide.js").exists():
                    print("✓ Pyodide files ready for bundling")
                else:
                    print("⚠ npm install succeeded but Pyodide files not found")
            else:
                print(f"✗ npm install failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("✗ npm install timed out after 5 minutes")
        except FileNotFoundError:
            print("✗ npm not found - please ensure Node.js and npm are installed")
            sys.exit(1)
        except Exception as e:
            print(f"✗ npm install failed: {e}")
