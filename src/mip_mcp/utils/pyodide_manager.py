"""Pyodide installation and management utilities."""

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class PyodideManager:
    """Manages Pyodide installation and availability."""

    _pyodide_path: str | None = None
    _initialization_lock = asyncio.Lock()

    @classmethod
    async def ensure_pyodide_available(cls) -> bool:
        """Ensure Pyodide is available, downloading if necessary.

        Returns:
            True if Pyodide is available, False otherwise.
        """
        async with cls._initialization_lock:
            if cls._pyodide_path:
                return True

            logger.info("Checking Pyodide availability...")

            # Check bundled first
            bundled_path = cls._check_bundled_pyodide()
            if bundled_path:
                cls._pyodide_path = bundled_path
                logger.info(f"Using bundled Pyodide at: {bundled_path}")
                return True

            # Check npm installation
            npm_path = await cls._find_npm_pyodide()
            if npm_path:
                cls._pyodide_path = npm_path
                logger.info(f"Using npm-installed Pyodide at: {npm_path}")
                return True

            # Auto-install if possible
            if await cls._auto_install_pyodide():
                # Try npm path again after installation
                npm_path = await cls._find_npm_pyodide()
                if npm_path:
                    cls._pyodide_path = npm_path
                    logger.info(f"Auto-installed Pyodide at: {npm_path}")
                    return True

            # Download directly as fallback
            downloaded_path = await cls._download_pyodide()
            if downloaded_path:
                cls._pyodide_path = downloaded_path
                logger.info(f"Downloaded Pyodide to: {downloaded_path}")
                return True

            logger.error("Failed to make Pyodide available")
            return False

    @classmethod
    def get_pyodide_path(cls) -> str | None:
        """Get the current Pyodide path."""
        return cls._pyodide_path

    @classmethod
    def _check_bundled_pyodide(cls) -> str | None:
        """Check for bundled pyodide installation (from wheel)."""
        try:
            import pkg_resources

            pyodide_js_path = pkg_resources.resource_filename(
                "mip_mcp", "data/pyodide/pyodide.js"
            )
            if Path(pyodide_js_path).exists():
                return pyodide_js_path
        except (ImportError, FileNotFoundError):
            pass
        return None

    @classmethod
    async def _find_npm_pyodide(cls) -> str | None:
        """Find pyodide installation via npm/node."""
        try:
            # Add temporary npm directory to search paths
            import tempfile

            temp_npm_dir = Path(tempfile.gettempdir()) / "mip-mcp-npm"

            proc = await asyncio.create_subprocess_exec(
                "node",
                "-e",
                f"""
try {{
    console.log(require.resolve('pyodide'));
}} catch (e) {{
    // Try alternative paths including temporary npm directory
    const path = require('path');
    const fs = require('fs');

    const searchPaths = [
        path.join(process.cwd(), 'node_modules', 'pyodide'),
        path.join(process.cwd(), '..', 'node_modules', 'pyodide'),
        path.join(process.cwd(), '..', '..', 'node_modules', 'pyodide'),
        path.join(__dirname, '..', '..', 'node_modules', 'pyodide'),
        path.join('{temp_npm_dir}', 'node_modules', 'pyodide')
    ];

    for (const searchPath of searchPaths) {{
        const pyodidePath = path.join(searchPath, 'pyodide.js');
        if (fs.existsSync(pyodidePath)) {{
            console.log(pyodidePath);
            process.exit(0);
        }}
    }}
    process.exit(1);
}}
                """,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                pyodide_path = stdout.decode().strip()
                if Path(pyodide_path).exists():
                    return pyodide_path

        except Exception as e:
            logger.debug(f"Failed to find npm pyodide: {e}")

        return None

    @classmethod
    async def _auto_install_pyodide(cls) -> bool:
        """Auto-install pyodide via npm if possible."""
        if not shutil.which("npm"):
            logger.debug("npm not available for auto-installation")
            return False

        # Try multiple installation strategies
        strategies = [
            cls._npm_install_in_project_root,
            cls._npm_install_global_temp,
        ]

        for strategy in strategies:
            try:
                if await strategy():
                    return True
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue

        return False

    @classmethod
    async def _npm_install_in_project_root(cls) -> bool:
        """Install pyodide in project root if package.json exists."""
        project_root = cls._find_project_root()
        if not project_root or not (project_root / "package.json").exists():
            return False

        logger.info("Installing Pyodide via npm in project root...")
        proc = await asyncio.create_subprocess_exec(
            "npm",
            "install",
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("npm install completed successfully")
            return True
        else:
            logger.warning(f"npm install failed: {stderr.decode()}")
            return False

    @classmethod
    async def _npm_install_global_temp(cls) -> bool:
        """Install pyodide in a temporary directory."""
        temp_dir = Path(tempfile.gettempdir()) / "mip-mcp-npm"
        temp_dir.mkdir(exist_ok=True)

        # Create a minimal package.json for pyodide installation
        package_json = temp_dir / "package.json"
        if not package_json.exists():
            package_json.write_text('{"name": "mip-mcp-pyodide", "version": "1.0.0"}')

        logger.info("Installing Pyodide via npm in temporary directory...")
        proc = await asyncio.create_subprocess_exec(
            "npm",
            "install",
            "pyodide@0.28.0",
            cwd=temp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("Pyodide installed successfully in temporary directory")
            return True
        else:
            logger.warning(f"npm install pyodide failed: {stderr.decode()}")
            return False

    @classmethod
    def _find_project_root(cls) -> Path | None:
        """Find project root by looking for package.json."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "package.json").exists():
                return current
            current = current.parent
        return None

    @classmethod
    async def _download_pyodide(cls) -> str | None:
        """Download Pyodide directly as a fallback."""
        try:
            import aiohttp

            # Create a temporary directory for pyodide
            temp_dir = Path(tempfile.gettempdir()) / "mip-mcp-pyodide"
            temp_dir.mkdir(exist_ok=True)

            pyodide_js = temp_dir / "pyodide.js"
            if pyodide_js.exists():
                # Check if other required files exist
                required_files = [
                    "pyodide.asm.js",
                    "pyodide.asm.wasm",
                    "python_stdlib.zip",
                ]
                if all((temp_dir / f).exists() for f in required_files):
                    return str(pyodide_js)

            # Download Pyodide complete package
            logger.info("Downloading Pyodide complete package...")
            async with aiohttp.ClientSession() as session:
                base_url = "https://cdn.jsdelivr.net/pyodide/v0.28.0/full"

                # List of required Pyodide files
                files_to_download = [
                    "pyodide.js",
                    "pyodide.asm.js",
                    "pyodide.asm.wasm",
                    "python_stdlib.zip",
                    "pyodide_py.tar",  # Python packages
                ]

                # Download all required files
                for filename in files_to_download:
                    file_path = temp_dir / filename
                    if file_path.exists():
                        continue  # Skip if already downloaded

                    url = f"{base_url}/{filename}"
                    logger.debug(f"Downloading {filename}...")

                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                with file_path.open("wb") as f:
                                    f.write(await response.read())
                                logger.debug(f"Downloaded {filename}")
                            else:
                                logger.warning(
                                    f"Failed to download {filename}: HTTP {response.status}"
                                )
                                # Continue with other files
                    except Exception as e:
                        logger.warning(f"Error downloading {filename}: {e}")
                        # Continue with other files

                # Verify essential files are present
                essential_files = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm"]
                if all((temp_dir / f).exists() for f in essential_files):
                    logger.info("Pyodide essential files downloaded successfully")
                    return str(pyodide_js)
                else:
                    missing = [
                        f for f in essential_files if not (temp_dir / f).exists()
                    ]
                    logger.error(
                        f"Failed to download essential Pyodide files: {missing}"
                    )
                    return None

        except ImportError:
            logger.debug("aiohttp not available for download")
        except Exception as e:
            logger.warning(f"Failed to download Pyodide: {e}")

        return None
