"""Node.js dependency management for uvx installations."""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NodeDependencyManager:
    """Manages Node.js dependencies for uvx installations."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize dependency manager.

        Args:
            project_root: Project root directory. If None, auto-detect.
        """
        self.project_root = project_root or self._find_project_root()
        self.package_json = self.project_root / "package.json"
        self.node_modules = self.project_root / "node_modules"

    def _find_project_root(self) -> Path:
        """Find project root by looking for package.json."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "package.json").exists():
                return current
            current = current.parent

        # Fallback to src/mip_mcp directory
        return Path(__file__).parent.parent

    def has_npm(self) -> bool:
        """Check if npm is available in system."""
        return shutil.which("npm") is not None

    def needs_installation(self) -> bool:
        """Check if Node.js dependencies need installation."""
        if not self.package_json.exists():
            return False

        if not self.node_modules.exists():
            return True

        # Check if pyodide specifically exists
        pyodide_path = self.node_modules / "pyodide"
        return not pyodide_path.exists()

    async def install_dependencies(self) -> bool:
        """Install Node.js dependencies automatically.

        Returns:
            True if installation succeeded, False otherwise.
        """
        if not self.has_npm():
            logger.warning(
                "npm not found. Please install Node.js to use this feature, "
                "or use the PyPI wheel version instead."
            )
            return False

        if not self.needs_installation():
            logger.debug("Node.js dependencies already installed")
            return True

        try:
            logger.info("Installing Node.js dependencies automatically...")
            logger.info(f"Running: npm install in {self.project_root}")

            process = await asyncio.create_subprocess_exec(
                "npm",
                "install",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info("Node.js dependencies installed successfully")
                return True
            else:
                logger.error(f"npm install failed: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Failed to install Node.js dependencies: {e}")
            return False

    async def ensure_dependencies(self) -> bool:
        """Ensure Node.js dependencies are available.

        Returns:
            True if dependencies are available, False otherwise.
        """
        if not self.needs_installation():
            return True

        return await self.install_dependencies()
