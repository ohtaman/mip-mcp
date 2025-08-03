"""Pyodide-based secure executor for MIP code execution."""

import asyncio
import builtins
import contextlib
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..models.responses import SolverProgress
from ..utils.library_detector import MIPLibrary, MIPLibraryDetector
from ..utils.logger import get_logger
from ..utils.pyodide_manager import PyodideManager

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

            # Get pyodide path from manager (should be available from server startup)
            pyodide_path = PyodideManager.get_pyodide_path()
            if not pyodide_path:
                # Fallback: try to ensure pyodide is available
                logger.info("Pyodide not ready, attempting to initialize...")
                if await PyodideManager.ensure_pyodide_available():
                    pyodide_path = PyodideManager.get_pyodide_path()

                if not pyodide_path:
                    raise RuntimeError(
                        "Pyodide module not found. Server may not have initialized properly."
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

    def _check_bundled_pyodide(self) -> str | None:
        """Check for bundled pyodide installation (from wheel)."""
        try:
            # Check if bundled pyodide files exist (installed from wheel)
            import pkg_resources

            try:
                pyodide_js_path = pkg_resources.resource_filename(
                    "mip_mcp", "pyodide/pyodide.js"
                )
                if Path(pyodide_js_path).exists():
                    return pyodide_js_path
            except (ImportError, FileNotFoundError):
                pass

            return None

        except Exception as e:
            logger.debug(f"Error checking bundled pyodide: {e}")
            return None

    async def _find_pyodide_path(self) -> str | None:
        """Find pyodide installation path."""
        try:
            # First check for bundled pyodide (from wheel installation)
            bundled_path = self._check_bundled_pyodide()
            if bundled_path:
                logger.info(f"Using bundled pyodide at: {bundled_path}")
                return bundled_path

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

        // Extract JSON result string (avoids ConversionError with tuple keys)
        let jsonResultString = null;
        try {
            jsonResultString = pyodide.globals.get('__json_result__');
        } catch (e) {
            console.error('Failed to extract JSON result:', e);
            return {
                success: false,
                error: `Failed to extract execution results: ${e.message}`,
                traceback: e.stack
            };
        }

        if (!jsonResultString) {
            return {
                success: false,
                error: 'No JSON result found in Python execution',
                traceback: null
            };
        }

        // Parse JSON result
        let parsedResult = null;
        try {
            parsedResult = JSON.parse(jsonResultString);
        } catch (e) {
            console.error('Failed to parse JSON result:', e);
            return {
                success: false,
                error: `Failed to parse execution results: ${e.message}`,
                json_raw: jsonResultString.substring(0, 500) + '...'
            };
        }

        return {
            success: true,
            result: result,
            json_data: parsedResult
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

            # Check for tuple key patterns and provide warnings
            tuple_warnings = self._detect_tuple_key_patterns(code)
            if tuple_warnings:
                logger.info(
                    f"Detected potential tuple key patterns: {len(tuple_warnings)} warnings"
                )
                for warning in tuple_warnings:
                    logger.warning(f"Tuple key pattern: {warning}")

            # Prepare execution code with tuple key conversion
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

            # Extract JSON results (new approach - no more ConversionError!)
            json_data = result.get("json_data", {})

            if not json_data:
                return (
                    "",
                    "No JSON data returned from Pyodide execution",
                    None,
                    detected_library,
                )

            # Check execution status
            execution_status = json_data.get("execution_status", "unknown")
            if execution_status == "error":
                error_msg = json_data.get("error_message", "Unknown error")
                traceback_info = json_data.get("traceback", "")
                return (
                    json_data.get("stdout", ""),
                    f"Execution error: {error_msg}\n{traceback_info}",
                    None,
                    detected_library,
                )

            # Extract content (LP preferred, then MPS)
            lp_content = json_data.get("lp_content")
            mps_content = json_data.get("mps_content")

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
                # Check if there were problems but failed to generate content
                problems_info = json_data.get("problems_info", [])
                if problems_info:
                    problem_errors = [
                        p.get("error") for p in problems_info if "error" in p
                    ]
                    if problem_errors:
                        error_details = "; ".join(problem_errors)
                        return (
                            json_data.get("stdout", ""),
                            f"Problem found but failed to generate content: {error_details}",
                            None,
                            detected_library,
                        )

                return (
                    json_data.get("stdout", ""),
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

            # Get execution output with tuple key conversion info
            stdout = json_data.get("stdout", "Code executed successfully in Pyodide")

            # Add information about variables that had tuple keys converted
            variables_info = json_data.get("variables_info", {})
            converted_vars = [
                name
                for name in variables_info
                if "_" in name and any(char.isdigit() for char in name)
            ]
            if converted_vars:
                stdout += f"\n\nNote: Successfully handled tuple keys in variables: {', '.join(converted_vars[:5])}"
                if len(converted_vars) > 5:
                    stdout += f" and {len(converted_vars) - 5} more"

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
import json
import pulp

# Capture stdout
__stdout_capture__ = io.StringIO()
sys.stdout = __stdout_capture__

# Setup data if provided
{data_setup}

def __convert_to_json_safe(obj, visited=None):
    '''Convert objects to JSON-safe format, handling tuple keys and complex objects.'''
    if visited is None:
        visited = set()

    # Avoid infinite recursion
    obj_id = id(obj)
    if obj_id in visited:
        return f"<circular_reference_to_{{type(obj).__name__}}>"
    visited.add(obj_id)

    try:
        if isinstance(obj, dict):
            converted = {{}}
            for key, value in obj.items():
                # Convert tuple keys to string representation
                if isinstance(key, tuple):
                    str_key = "_".join(str(k) for k in key)
                    converted[str_key] = __convert_to_json_safe(value, visited)
                elif isinstance(key, (str, int, float, bool)) or key is None:
                    converted[str(key)] = __convert_to_json_safe(value, visited)
                else:
                    # Convert other key types to string
                    converted[str(key)] = __convert_to_json_safe(value, visited)
            return converted
        elif isinstance(obj, (list, tuple)):
            return [__convert_to_json_safe(item, visited) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif hasattr(obj, '__dict__'):
            # Handle objects with attributes (like PuLP variables)
            return {{
                '__type__': type(obj).__name__,
                '__module__': getattr(type(obj), '__module__', 'unknown'),
                '__str__': str(obj),
                '__repr__': repr(obj)
            }}
        else:
            return str(obj)
    except Exception:
        return f"<unconvertible_{{type(obj).__name__}}>"
    finally:
        visited.discard(obj_id)

def __extract_problem_info(globals_dict):
    '''Extract PuLP problem information and generate LP/MPS content.'''
    problems_info = []
    lp_content = None
    mps_content = None

    for name, obj in globals_dict.items():
        if hasattr(obj, 'writeLP') and hasattr(obj, 'writeMPS'):
            try:
                # Generate LP content
                import tempfile
                import os

                # Create temporary files for LP and MPS
                with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as lp_file:
                    lp_path = lp_file.name

                with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as mps_file:
                    mps_path = mps_file.name

                # Write LP and MPS files
                obj.writeLP(lp_path)
                obj.writeMPS(mps_path)

                # Read content
                with open(lp_path, 'r') as f:
                    lp_content = f.read()

                with open(mps_path, 'r') as f:
                    mps_content = f.read()

                # Clean up temporary files
                os.unlink(lp_path)
                os.unlink(mps_path)

                problem_info = {{
                    'name': name,
                    'sense': str(getattr(obj, 'sense', 'unknown')),
                    'status': str(getattr(obj, 'status', 'unknown')),
                    'num_variables': len(getattr(obj, 'variables', [])),
                    'num_constraints': len(getattr(obj, 'constraints', [])),
                    'objective': str(getattr(obj, 'objective', 'none'))
                }}
                problems_info.append(problem_info)

                # Use first problem found
                break

            except Exception as e:
                problems_info.append({{
                    'name': name,
                    'error': f"Failed to extract problem info: {{e}}"
                }})

    return problems_info, lp_content, mps_content

try:
    # Execute user code
{self._indent_code(user_code, 4)}

    # Reset stdout
    sys.stdout = sys.__stdout__
    __stdout__ = __stdout_capture__.getvalue()

    # Extract and serialize all relevant information as JSON
    __globals_copy = dict(globals())

    # Find and extract PuLP problem information
    __problems_info, __lp_content, __mps_content = __extract_problem_info(__globals_copy)

    # Convert user variables to JSON-safe format
    __variables_info = {{}}
    for __var_name in list(__globals_copy.keys()):
        if (not __var_name.startswith('_') and
            __var_name not in ['pulp', 'io', 'sys', 'tempfile', 'json'] and
            not callable(__globals_copy[__var_name])):
            try:
                __variables_info[__var_name] = __convert_to_json_safe(__globals_copy[__var_name])
            except Exception as __e:
                __variables_info[__var_name] = f"<conversion_error: {{__e}}>"

    # Create comprehensive result data
    __result_data = {{
        'stdout': __stdout__,
        'lp_content': __lp_content,
        'mps_content': __mps_content,
        'problems_info': __problems_info,
        'variables_info': __variables_info,
        'execution_status': 'success'
    }}

    # Serialize to JSON string
    __json_result__ = json.dumps(__result_data, ensure_ascii=False, indent=None)

except Exception as e:
    sys.stdout = sys.__stdout__
    __stdout__ = __stdout_capture__.getvalue()
    __stdout__ += f"\\nError: {{e}}"

    import traceback
    __stdout__ += f"\\nTraceback: {{traceback.format_exc()}}"

    # Create error result data
    __result_data = {{
        'stdout': __stdout__,
        'lp_content': None,
        'mps_content': None,
        'problems_info': [],
        'variables_info': {{}},
        'execution_status': 'error',
        'error_message': str(e),
        'traceback': traceback.format_exc()
    }}

    # Serialize error data to JSON
    __json_result__ = json.dumps(__result_data, ensure_ascii=False, indent=None)

# Store JSON result for extraction
globals()['__json_result__'] = __json_result__
"""

        return wrapper_code

    def _detect_tuple_key_patterns(self, code: str) -> list[str]:
        """Detect potential tuple key patterns in user code that might cause ConversionError.

        Args:
            code: User's Python code

        Returns:
            List of warning messages about potential tuple key issues
        """
        warnings = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            line = line.strip()

            # Look for dictionary assignments with tuple keys
            if "[(" in line and ")]" in line and "=" in line:
                warnings.append(
                    f"Line {i}: Potential tuple key assignment detected: {line[:50]}... "
                    "Consider using string keys instead: dict[f'{key1}_{key2}_{key3}'] = value"
                )

            # Look for tuple construction patterns
            if (
                "for " in line
                and " in " in line
                and "(" in line
                and ")" in line
                and any(keyword in line for keyword in ["range", "enumerate", "zip"])
            ):
                # This might be creating tuples for dictionary keys
                next_lines = lines[i : i + 3] if i < len(lines) else []
                for next_line in next_lines:
                    if "[(" in next_line and ")]" in next_line:
                        warnings.append(
                            f"Lines {i}-{i+len(next_lines)}: Potential tuple key creation in loop. "
                            "Consider using string keys for Pyodide compatibility."
                        )
                        break

        return warnings

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
