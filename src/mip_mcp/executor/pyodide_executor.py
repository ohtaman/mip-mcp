"""Pyodide-based secure executor for MIP code execution."""

import asyncio
import builtins
import contextlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from ..models.responses import SolverProgress
from ..utils.library_detector import MIPLibrary, MIPLibraryDetector
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PyodideExecutor:
    """Secure Pyodide-based executor for MIP code."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Pyodide executor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pyodide_process = None
        self.detector = MIPLibraryDetector()
        self._pyodide_initialized = False
        self._cleanup_started = False
        self.progress_callback: Callable[[SolverProgress], None] | None = None
        self._progress_queue: asyncio.Queue | None = None
        self.start_time: float = 0.0
        self.progress_interval: float = config.get(
            "progress_interval", 10.0
        )  # Progress update interval
        self.execution_timeout: float = config.get(
            "execution_timeout", 60.0
        )  # Execution timeout for MCP

        logger.info("Pyodide executor initialized")

    def set_progress_callback(
        self, callback: Callable[[SolverProgress], None] | None
    ) -> None:
        """Set progress callback function.

        Args:
            callback: Function to call with progress updates
        """
        self.progress_callback = callback

    def _send_progress(self, stage: str, message: str | None = None) -> None:
        """Send progress update if callback is set."""
        if not self.progress_callback:
            return

        try:
            time_elapsed = time.time() - self.start_time

            progress = SolverProgress(
                stage=stage, time_elapsed=time_elapsed, message=message
            )

            # Queue progress for async processing if queue is available
            if self._progress_queue:
                try:
                    self._progress_queue.put_nowait(progress)
                except asyncio.QueueFull:
                    logger.warning("Progress queue is full, skipping update")
            else:
                # Fallback to direct call (may cause RuntimeWarning if async)
                if asyncio.iscoroutinefunction(self.progress_callback):
                    # Create task for async callback
                    task = asyncio.create_task(self.progress_callback(progress))
                    # Add done callback to handle any exceptions
                    task.add_done_callback(
                        lambda t: t.exception() if not t.cancelled() else None
                    )
                else:
                    self.progress_callback(progress)

        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")

    async def _execute_with_periodic_progress(
        self, code_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute Pyodide code with periodic progress updates during long-running operations.

        Args:
            code_data: Code execution data for Pyodide

        Returns:
            Execution result dictionary

        Raises:
            TimeoutError: If execution exceeds timeout limit
        """
        # Start execution in background
        execution_task = asyncio.create_task(self._communicate_with_pyodide(code_data))

        start_time = time.time()
        last_progress_time = start_time
        execution_time = 0.0

        # Monitor execution with periodic progress updates
        while not execution_task.done():
            current_time = time.time()
            execution_time = current_time - start_time

            # Check if Pyodide process is still alive
            if self.pyodide_process and self.pyodide_process.returncode is not None:
                # Process has terminated
                break

            # Check for timeout
            if execution_time > self.execution_timeout:
                # Kill the process if it's still running
                if self.pyodide_process and self.pyodide_process.returncode is None:
                    self.pyodide_process.terminate()
                    try:
                        await asyncio.wait_for(self.pyodide_process.wait(), timeout=5.0)
                    except TimeoutError:
                        self.pyodide_process.kill()

                execution_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await execution_task
                raise TimeoutError(
                    f"Code execution exceeded timeout limit of {self.execution_timeout} seconds"
                )

            # Send periodic progress update (only if process is still alive)
            if current_time - last_progress_time >= self.progress_interval:
                self._send_progress(
                    "modeling",
                    f"Code executing... ({execution_time:.1f}s elapsed, process running)",
                )
                last_progress_time = current_time

            # Wait for progress interval before checking again
            await asyncio.sleep(self.progress_interval)

        # Get the final result
        return await execution_task

    async def _initialize_pyodide(self) -> None:
        """Initialize Pyodide environment (lazy loading)."""
        if self._pyodide_initialized:
            return

        try:
            logger.info("Initializing Pyodide environment...")

            # Find pyodide path first (with automatic download fallback)
            pyodide_path = await self._find_or_download_pyodide_path()
            if not pyodide_path:
                raise RuntimeError(
                    "Pyodide initialization failed: Could not find or download Pyodide. "
                    "Please ensure internet access or manually install: npm install pyodide"
                )

            logger.info(f"Found Pyodide at: {pyodide_path}")

            # Write the Node.js script to a temporary file
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
                f.write(self._get_pyodide_script(pyodide_path))
                script_path = f.name

            try:
                # Create a new Node.js process to run Pyodide
                cmd = ["node", script_path]

                self.pyodide_process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,  # Create new process group
                )

                # Wait for process to signal readiness
                await self._wait_for_process_ready()

                # Wait for initialization
                init_result = await self._communicate_with_pyodide({"action": "init"})

                if not init_result.get("success"):
                    # Try to get stderr for better error info
                    try:
                        stderr_data = await asyncio.wait_for(
                            self.pyodide_process.stderr.read(1000), timeout=1.0
                        )
                        stderr_msg = stderr_data.decode().strip()
                        if stderr_msg:
                            logger.error(f"Pyodide stderr: {stderr_msg}")
                    except Exception:
                        pass

                    raise RuntimeError(
                        f"Pyodide initialization failed: {init_result.get('error', 'unknown error')}"
                    )

                self._pyodide_initialized = True
                logger.info("Pyodide environment initialized successfully")

            finally:
                # Clean up script file
                with contextlib.suppress(OSError, FileNotFoundError):
                    Path(script_path).unlink()

        except Exception as e:
            logger.error(f"Failed to initialize Pyodide: {e}")
            raise RuntimeError(f"Pyodide initialization failed: {e}") from e

    async def _wait_for_process_ready(self) -> None:
        """Wait for Node.js process to signal readiness."""
        try:
            # Read stderr until we see the ready message
            while True:
                stderr_line = await asyncio.wait_for(
                    self.pyodide_process.stderr.readline(), timeout=10.0
                )
                stderr_msg = stderr_line.decode().strip()
                if stderr_msg:
                    logger.debug(f"Pyodide process stderr: {stderr_msg}")
                    if "Node.js process ready for commands" in stderr_msg:
                        logger.info("Pyodide process is ready")
                        return
        except TimeoutError as e:
            raise RuntimeError(
                "Timeout waiting for Pyodide process to become ready"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error waiting for Pyodide process readiness: {e}"
            ) from e

    async def _find_pyodide_path(self) -> str | None:
        """Find pyodide installation path."""
        try:
            # Try to find pyodide using Node.js
            proc = await asyncio.create_subprocess_exec(
                "node",
                "-e",
                """
try {
    console.log(require.resolve('pyodide'));
} catch (e) {
    // Try alternative paths
    const path = require('path');
    const fs = require('fs');

    const searchPaths = [
        path.join(process.cwd(), 'node_modules', 'pyodide'),
        path.join(process.cwd(), '..', 'node_modules', 'pyodide'),
        path.join(process.cwd(), '..', '..', 'node_modules', 'pyodide'),
        path.join(__dirname, '..', '..', 'node_modules', 'pyodide')
    ];

    for (const searchPath of searchPaths) {
        const packagePath = path.join(searchPath, 'package.json');
        if (fs.existsSync(packagePath)) {
            console.log(path.join(searchPath, 'pyodide.js'));
            process.exit(0);
        }
    }

    console.error('PYODIDE_NOT_FOUND');
    process.exit(1);
}
                """,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                path = stdout.decode().strip()
                if path and not path.startswith("PYODIDE_NOT_FOUND"):
                    return path

            return None

        except Exception as e:
            logger.error(f"Failed to find pyodide path: {e}")
            return None

    def _get_pyodide_cache_dir(self) -> Path:
        """Get the pyodide cache directory."""
        # Use user data directory for caching
        if os.name == "nt":  # Windows
            cache_dir = Path(os.environ.get("APPDATA", "")) / "mip-mcp" / "pyodide"
        else:  # Unix-like
            cache_dir = Path.home() / ".cache" / "mip-mcp" / "pyodide"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    async def _download_pyodide(self) -> str | None:
        """Download pyodide from npm registry and cache it."""
        try:
            logger.info("Pyodide not found locally, downloading...")

            cache_dir = self._get_pyodide_cache_dir()
            pyodide_js_path = cache_dir / "pyodide.js"

            # Check if already cached
            if pyodide_js_path.exists():
                logger.info(f"Using cached pyodide at: {pyodide_js_path}")
                return str(pyodide_js_path)

            # Download pyodide package from npm registry
            npm_url = "https://registry.npmjs.org/pyodide/-/pyodide-0.28.0.tgz"
            logger.info("Downloading pyodide package...")

            # Download to temporary file
            with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp_file:
                temp_path = tmp_file.name

            try:
                # Run download in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, urlretrieve, npm_url, temp_path)

                # Extract the package
                logger.info("Extracting pyodide package...")
                await loop.run_in_executor(
                    None, self._extract_pyodide_package, temp_path, cache_dir
                )

                # Verify extraction
                if pyodide_js_path.exists():
                    logger.info(
                        f"Successfully downloaded and cached pyodide at: {pyodide_js_path}"
                    )
                    return str(pyodide_js_path)
                else:
                    logger.error("Failed to extract pyodide.js from downloaded package")
                    return None

            finally:
                # Clean up temporary file
                with contextlib.suppress(OSError):
                    Path(temp_path).unlink()

        except Exception as e:
            logger.error(f"Failed to download pyodide: {e}")
            return None

    def _extract_pyodide_package(self, tar_path: str, extract_dir: Path) -> None:
        """Extract pyodide package from tar.gz file."""
        import tarfile

        with tarfile.open(tar_path, "r:gz") as tar:
            # Extract only the files we need
            needed_files = [
                "package/pyodide.js",
                "package/pyodide.asm.js",
                "package/pyodide.asm.wasm",
                "package/pyodide-lock.json",
                "package/python_stdlib.zip",
            ]

            for member in tar.getmembers():
                if member.name in needed_files:
                    # Extract to cache directory, removing "package/" prefix
                    target_name = member.name.replace("package/", "")
                    member.name = target_name
                    tar.extract(member, extract_dir)

    async def _find_or_download_pyodide_path(self) -> str | None:
        """Find pyodide path, downloading if necessary."""
        # First try to find existing installation
        path = await self._find_pyodide_path()
        if path:
            return path

        # If not found, try to download it
        logger.info("Pyodide not found in local installations, attempting download...")
        downloaded_path = await self._download_pyodide()

        if downloaded_path:
            return downloaded_path

        # If download failed, provide helpful error message
        logger.error(
            "Could not find or download pyodide. Please install manually:\n"
            "  npm install pyodide\n"
            "Or ensure you have internet access for automatic download."
        )
        return None

    def _get_pyodide_script(self, pyodide_path: str) -> str:
        """Get the Node.js script for Pyodide execution."""
        script = """
// Load pyodide from resolved path
const { loadPyodide } = require('PYODIDE_PATH_PLACEHOLDER');"""
        script = script.replace("PYODIDE_PATH_PLACEHOLDER", pyodide_path)
        script += """
const readline = require('readline');

let pyodide = null;

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stderr  // Use stderr to avoid JSON output conflicts
});

async function initPyodide() {
    try {
        // Redirect Pyodide stdout to stderr to avoid JSON conflicts
        pyodide = await loadPyodide({
            stdout: (text) => console.error('PYODIDE_STDOUT:', text),
            stderr: (text) => console.error('PYODIDE_STDERR:', text)
        });
        await pyodide.loadPackage("micropip");

        // Install PuLP
        await pyodide.runPythonAsync(`
            import micropip
            await micropip.install('pulp')
        `);

        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

async function executePython(code) {
    try {
        const result = await pyodide.runPythonAsync(code);

        // Extract globals and generate file content directly
        const globals = pyodide.globals.toJs();
        let lpContent = null;
        let mpsContent = null;

        // Look for PuLP problems in globals and generate content using virtual filesystem
        for (const [name, obj] of Object.entries(globals)) {
            if (obj && typeof obj === 'object' && obj.writeLP && obj.writeMPS) {
                try {
                    // Use Pyodide virtual filesystem to generate LP content
                    await pyodide.runPythonAsync(`${name}.writeLP("/tmp/problem.lp")`);
                    lpContent = pyodide.FS.readFile("/tmp/problem.lp", { encoding: "utf8" });

                    // Use Pyodide virtual filesystem to generate MPS content
                    await pyodide.runPythonAsync(`${name}.writeMPS("/tmp/problem.mps")`);
                    mpsContent = pyodide.FS.readFile("/tmp/problem.mps", { encoding: "utf8" });

                    break; // Use first problem found
                } catch (e) {
                    console.error('Error generating content:', e);
                }
            }
        }

        // Add generated content to globals
        if (lpContent) globals['__lp_content__'] = lpContent;
        if (mpsContent) globals['__mps_content__'] = mpsContent;

        return {
            success: true,
            result: result,
            globals: globals
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            traceback: error.stack
        };
    }
}

rl.on('line', async (input) => {
    try {
        const request = JSON.parse(input);
        let response;

        switch (request.action) {
            case 'init':
                response = await initPyodide();
                break;
            case 'execute':
                response = await executePython(request.code);
                break;
            case 'exit':
                process.exit(0);
                break;
            default:
                response = { success: false, error: 'Unknown action' };
        }

        console.log(JSON.stringify(response));
    } catch (error) {
        console.log(JSON.stringify({ success: false, error: error.message }));
    }
});

// Handle process cleanup
process.on('SIGINT', () => process.exit(0));
process.on('SIGTERM', () => process.exit(0));

// Signal readiness
console.error('DEBUG: Node.js process ready for commands');
"""
        return script

    async def _communicate_with_pyodide(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Send request to Pyodide process and get response."""
        if not self.pyodide_process:
            raise RuntimeError("Pyodide process not initialized")

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.pyodide_process.stdin.write(request_json.encode())
            await self.pyodide_process.stdin.drain()

            # Read response with timeout
            response_line = await asyncio.wait_for(
                self.pyodide_process.stdout.readline(),
                timeout=self.execution_timeout + 10.0,  # Allow for execution + margin
            )
            response_str = response_line.decode().strip()

            if not response_str:
                # Try to read stderr for error info
                try:
                    stderr_data = await asyncio.wait_for(
                        self.pyodide_process.stderr.read(1000), timeout=1.0
                    )
                    stderr_msg = stderr_data.decode().strip()
                    return {
                        "success": False,
                        "error": f"Empty response. Stderr: {stderr_msg}",
                    }
                except Exception:
                    return {"success": False, "error": "Empty response from Pyodide"}

            try:
                return json.loads(response_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {response_str}")
                return {"success": False, "error": f"Invalid JSON response: {e}"}

        except TimeoutError:
            logger.error("Timeout waiting for Pyodide response")
            return {"success": False, "error": "Timeout waiting for Pyodide response"}
        except Exception as e:
            logger.error(f"Communication with Pyodide failed: {e}")
            return {"success": False, "error": str(e)}

    async def execute_mip_code(
        self,
        code: str,
        data: dict[str, Any] | None = None,
    ) -> tuple[str, str, str | None, MIPLibrary]:
        """Execute MIP code in Pyodide environment.

        Args:
            code: PuLP Python code to execute
            data: Optional data dictionary to pass to the code

        Returns:
            Tuple of (stdout, stderr, file_path, detected_library).
            File format is automatically detected (LP preferred, then MPS).
        """
        # Initialize timing
        self.start_time = time.time()

        # Send initial progress
        self._send_progress("modeling", "Initializing Pyodide environment")

        await self._initialize_pyodide()

        # Send progress for library detection
        self._send_progress("modeling", "Detecting MIP library")

        # Detect library (only PuLP supported)
        detected_library = self.detector.detect_library(code)

        if detected_library == MIPLibrary.UNKNOWN:
            return "", "Unknown or unsupported MIP library", None, MIPLibrary.UNKNOWN

        # Currently only support PuLP in Pyodide
        if detected_library != MIPLibrary.PULP:
            return (
                "",
                f"Library {detected_library.value} not supported in Pyodide yet",
                None,
                detected_library,
            )

        try:
            # Send progress for code preparation
            self._send_progress("modeling", "Preparing code for execution")

            # Prepare execution code
            execution_code = self._prepare_execution_code(code, data)

            # Send progress for execution
            self._send_progress(
                "modeling", "Executing PuLP code and generating optimization model"
            )

            # Execute in Pyodide with periodic progress monitoring
            result = await self._execute_with_periodic_progress(
                {"action": "execute", "code": execution_code}
            )

            if not result.get("success"):
                return (
                    "",
                    result.get("error", "Unknown execution error"),
                    None,
                    detected_library,
                )

            # Send progress for result extraction
            self._send_progress("modeling", "Extracting optimization file content")

            # Extract results
            globals_dict = result.get("globals", {})

            # Look for generated content
            lp_content = globals_dict.get("__lp_content__")
            mps_content = globals_dict.get("__mps_content__")

            # Automatic format detection: LP preferred, then MPS
            content = None
            file_format = None
            if lp_content:
                content = lp_content
                file_format = "lp"
            elif mps_content:
                content = mps_content
                file_format = "mps"

            if not content:
                return (
                    "",
                    "No optimization file content generated",
                    None,
                    detected_library,
                )

            # Send progress for file generation
            self._send_progress(
                "modeling", f"Generating {file_format.upper()} optimization file"
            )

            # Write content to temporary file
            temp_file = self._create_temp_file(content, file_format)

            # Send final modeling progress
            self._send_progress(
                "modeling",
                f"Optimization model generated successfully ({file_format.upper()} format)",
            )

            # Get execution output
            stdout = globals_dict.get(
                "__stdout__", "Code executed successfully in Pyodide"
            )

            return stdout, "", temp_file, detected_library

        except TimeoutError as e:
            logger.error(f"Pyodide execution timed out: {e}")
            return "", f"Execution timed out: {e}", None, detected_library

        except Exception as e:
            logger.error(f"Pyodide execution failed: {e}")
            return "", f"Execution failed: {e}", None, detected_library

    def _prepare_execution_code(
        self, user_code: str, data: dict[str, Any] | None
    ) -> str:
        """Prepare code for execution in Pyodide.

        Args:
            user_code: User's MIP code
            data: Optional data dictionary

        Returns:
            Complete code to execute in Pyodide
        """
        data_setup = ""
        if data:
            data_setup = f"__data__ = {json.dumps(data)}\n"

        wrapper_code = f"""
import io
import sys
import tempfile
import pulp

# Capture stdout
__stdout_capture__ = io.StringIO()
sys.stdout = __stdout_capture__

# Setup data if provided
{data_setup}

try:
    # Execute user code
{self._indent_code(user_code, 4)}

    # Reset stdout
    sys.stdout = sys.__stdout__
    __stdout__ = __stdout_capture__.getvalue()

    # Note: LP/MPS content generation is now handled in Node.js side
    # for better Pyodide integration

except Exception as e:
    sys.stdout = sys.__stdout__
    __stdout__ = __stdout_capture__.getvalue()
    __stdout__ += f"\\nError: {{e}}"

    import traceback
    __stdout__ += f"\\nTraceback: {{traceback.format_exc()}}"

# Return results through global variables
globals()['__stdout__'] = __stdout__
"""

        return wrapper_code

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.split("\n"))

    def _create_temp_file(self, content: str, format_type: str) -> str:
        """Create temporary file with the given content.

        Args:
            content: File content
            format_type: File format ("lp" or "mps")

        Returns:
            Path to temporary file
        """
        suffix = f".{format_type.lower()}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name

        logger.info(f"Created temporary {format_type.upper()} file: {temp_path}")
        return temp_path

    async def validate_code(self, code: str) -> dict[str, Any]:
        """Validate PuLP code without executing it.

        Args:
            code: PuLP code to validate

        Returns:
            Validation results
        """
        try:
            # Detect library (only PuLP supported)
            detected_library = self.detector.detect_library(code)

            if detected_library == MIPLibrary.UNKNOWN:
                return {
                    "status": "error",
                    "message": "Could not detect MIP library or unsupported library",
                    "issues": ["Unknown or unsupported MIP library"],
                    "library_detected": "unknown",
                }

            # Basic syntax validation
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                return {
                    "status": "error",
                    "message": f"Syntax error: {e}",
                    "issues": [f"Syntax error at line {e.lineno}: {e.msg}"],
                    "library_detected": detected_library.value,
                }

            # Pyodide-specific validation
            issues = []

            # Check for prohibited operations (should be minimal in Pyodide)
            prohibited_patterns = [
                ("import os", "Direct OS access not allowed"),
                ("import subprocess", "Subprocess not allowed"),
                (
                    "open(",
                    "Direct file operations not recommended (use content variables instead)",
                ),
            ]

            for pattern, message in prohibited_patterns:
                if pattern in code:
                    issues.append(f"Warning: {message}")

            return {
                "status": "success",
                "message": "Code validation passed",
                "issues": issues,
                "library_detected": detected_library.value,
                "secure_execution": True,
                "sandbox": "pyodide",
            }

        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return {
                "status": "error",
                "message": f"Validation failed: {e}",
                "issues": [str(e)],
                "library_detected": "unknown",
            }

    async def cleanup(self):
        """Clean up Pyodide process with proper timeout handling."""
        if self._cleanup_started:
            logger.debug("Cleanup already in progress")
            return

        self._cleanup_started = True

        if self.pyodide_process:
            logger.debug(
                f"Starting cleanup of Pyodide process (PID: {self.pyodide_process.pid})"
            )

            try:
                # First, try graceful shutdown with exit command
                try:
                    await asyncio.wait_for(
                        self._communicate_with_pyodide({"action": "exit"}), timeout=2.0
                    )
                    logger.debug("Sent exit command to Pyodide process")
                except (TimeoutError, Exception) as e:
                    logger.debug(f"Failed to send exit command: {e}")

                # Give process a moment to exit gracefully
                try:
                    await asyncio.wait_for(self.pyodide_process.wait(), timeout=1.0)
                    logger.debug("Process exited gracefully")
                except TimeoutError:
                    # Process didn't exit gracefully, try SIGTERM
                    logger.debug("Process didn't exit gracefully, sending SIGTERM")
                    try:
                        self.pyodide_process.terminate()
                        await asyncio.wait_for(self.pyodide_process.wait(), timeout=5.0)
                        logger.debug("Process terminated with SIGTERM")
                    except TimeoutError:
                        # Last resort: SIGKILL
                        logger.warning(
                            "Process didn't respond to SIGTERM, sending SIGKILL"
                        )
                        try:
                            self.pyodide_process.kill()
                            await asyncio.wait_for(
                                self.pyodide_process.wait(), timeout=2.0
                            )
                            logger.debug("Process killed with SIGKILL")
                        except TimeoutError:
                            logger.error(
                                "Process failed to terminate even after SIGKILL"
                            )

            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            finally:
                self.pyodide_process = None
                self._pyodide_initialized = False
                logger.info("Pyodide process cleanup completed")

    def __del__(self):
        """Cleanup on destruction."""
        if self.pyodide_process and not self._cleanup_started:
            # Note: Can't use await in __del__, so we'll just terminate synchronously
            logger.warning(
                "PyodideExecutor finalized without cleanup - terminating process"
            )
            with contextlib.suppress(builtins.BaseException):
                self.pyodide_process.terminate()
                # Try to kill if terminate doesn't work quickly
                try:
                    # Non-blocking check if process is still alive
                    if self.pyodide_process.poll() is None:
                        self.pyodide_process.kill()
                except (OSError, AttributeError):
                    pass  # Process might already be gone
