"""Pyodide-based secure executor for MIP code execution."""

import asyncio
import json
import tempfile
import os
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

from ..utils.library_detector import MIPLibraryDetector, MIPLibrary
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PyodideExecutor:
    """Secure Pyodide-based executor for MIP code."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Pyodide executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pyodide_process = None
        self.detector = MIPLibraryDetector()
        self._pyodide_initialized = False
        
        logger.info("Pyodide executor initialized")
    
    async def _initialize_pyodide(self) -> None:
        """Initialize Pyodide environment (lazy loading)."""
        if self._pyodide_initialized:
            return
        
        try:
            logger.info("Initializing Pyodide environment...")
            
            # Find pyodide path first
            pyodide_path = await self._find_pyodide_path()
            if not pyodide_path:
                raise RuntimeError("Pyodide module not found. Please install: npm install pyodide")
            
            logger.info(f"Found Pyodide at: {pyodide_path}")
            
            # Write the Node.js script to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(self._get_pyodide_script(pyodide_path))
                script_path = f.name
            
            try:
                # Create a new Node.js process to run Pyodide
                cmd = ["node", script_path]
                
                self.pyodide_process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for process to signal readiness
                await self._wait_for_process_ready()
                
                # Wait for initialization
                init_result = await self._communicate_with_pyodide({"action": "init"})
                
                if not init_result.get("success"):
                    # Try to get stderr for better error info
                    try:
                        stderr_data = await asyncio.wait_for(
                            self.pyodide_process.stderr.read(1000), 
                            timeout=1.0
                        )
                        stderr_msg = stderr_data.decode().strip()
                        if stderr_msg:
                            logger.error(f"Pyodide stderr: {stderr_msg}")
                    except:
                        pass
                    
                    raise RuntimeError(f"Pyodide initialization failed: {init_result.get('error', 'unknown error')}")
                
                self._pyodide_initialized = True
                logger.info("Pyodide environment initialized successfully")
                
            finally:
                # Clean up script file
                try:
                    import os
                    os.unlink(script_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to initialize Pyodide: {e}")
            raise RuntimeError(f"Pyodide initialization failed: {e}")
    
    async def _wait_for_process_ready(self) -> None:
        """Wait for Node.js process to signal readiness."""
        try:
            # Read stderr until we see the ready message
            while True:
                stderr_line = await asyncio.wait_for(
                    self.pyodide_process.stderr.readline(),
                    timeout=10.0
                )
                stderr_msg = stderr_line.decode().strip()
                if stderr_msg:
                    logger.debug(f"Pyodide process stderr: {stderr_msg}")
                    if "Node.js process ready for commands" in stderr_msg:
                        logger.info("Pyodide process is ready")
                        return
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for Pyodide process to become ready")
        except Exception as e:
            raise RuntimeError(f"Error waiting for Pyodide process readiness: {e}")
    
    async def _find_pyodide_path(self) -> Optional[str]:
        """Find pyodide installation path."""
        try:
            # Try to find pyodide using Node.js
            proc = await asyncio.create_subprocess_exec(
                'node', '-e', '''
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
                ''',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                path = stdout.decode().strip()
                if path and not path.startswith('PYODIDE_NOT_FOUND'):
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
        script = script.replace('PYODIDE_PATH_PLACEHOLDER', pyodide_path)
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
    
    async def _communicate_with_pyodide(self, request: Dict[str, Any]) -> Dict[str, Any]:
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
                timeout=30.0  # 30 second timeout
            )
            response_str = response_line.decode().strip()
            
            if not response_str:
                # Try to read stderr for error info
                try:
                    stderr_data = await asyncio.wait_for(
                        self.pyodide_process.stderr.read(1000),
                        timeout=1.0
                    )
                    stderr_msg = stderr_data.decode().strip()
                    return {"success": False, "error": f"Empty response. Stderr: {stderr_msg}"}
                except:
                    return {"success": False, "error": "Empty response from Pyodide"}
            
            try:
                return json.loads(response_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response: {response_str}")
                return {"success": False, "error": f"Invalid JSON response: {e}"}
            
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for Pyodide response")
            return {"success": False, "error": "Timeout waiting for Pyodide response"}
        except Exception as e:
            logger.error(f"Communication with Pyodide failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_mip_code(
        self,
        code: str,
        data: Optional[Dict[str, Any]] = None,
        output_format: str = "mps",
        library: str = "auto"
    ) -> Tuple[str, str, Optional[str], MIPLibrary]:
        """Execute MIP code in Pyodide environment.
        
        Args:
            code: MIP Python code to execute
            data: Optional data dictionary to pass to the code  
            output_format: Output format ("mps" or "lp")
            library: Library to use ("auto", "pulp", "python-mip")
            
        Returns:
            Tuple of (stdout, stderr, file_path, detected_library)
        """
        await self._initialize_pyodide()
        
        # Detect library
        if library == "auto":
            detected_library = self.detector.detect_library(code)
        else:
            detected_library = self.detector.validate_library_choice(library)
        
        if detected_library == MIPLibrary.UNKNOWN:
            return "", "Unknown or unsupported MIP library", None, MIPLibrary.UNKNOWN
        
        # Currently only support PuLP in Pyodide
        if detected_library != MIPLibrary.PULP:
            return "", f"Library {detected_library.value} not supported in Pyodide yet", None, detected_library
        
        try:
            # Prepare execution code
            execution_code = self._prepare_execution_code(code, data, output_format)
            
            # Execute in Pyodide
            result = await self._communicate_with_pyodide({
                "action": "execute",
                "code": execution_code
            })
            
            if not result.get("success"):
                return "", result.get("error", "Unknown execution error"), None, detected_library
            
            # Extract results
            globals_dict = result.get("globals", {})
            
            # Look for generated content
            lp_content = globals_dict.get("__lp_content__")
            mps_content = globals_dict.get("__mps_content__")
            
            content = None
            if output_format == "lp" and lp_content:
                content = lp_content
            elif output_format == "mps" and mps_content:
                content = mps_content
            elif lp_content:  # fallback to LP if available
                content = lp_content
            elif mps_content:  # fallback to MPS if available
                content = mps_content
            
            if not content:
                return "", "No optimization file content generated", None, detected_library
            
            # Write content to temporary file
            temp_file = self._create_temp_file(content, output_format)
            
            # Get execution output
            stdout = globals_dict.get("__stdout__", "Code executed successfully in Pyodide")
            
            return stdout, "", temp_file, detected_library
            
        except Exception as e:
            logger.error(f"Pyodide execution failed: {e}")
            return "", f"Execution failed: {e}", None, detected_library
    
    def _prepare_execution_code(self, user_code: str, data: Optional[Dict[str, Any]], output_format: str) -> str:
        """Prepare code for execution in Pyodide.
        
        Args:
            user_code: User's MIP code
            data: Optional data dictionary
            output_format: Desired output format
            
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
        
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=suffix, 
            delete=False
        ) as f:
            f.write(content)
            temp_path = f.name
        
        logger.info(f"Created temporary {format_type.upper()} file: {temp_path}")
        return temp_path
    
    async def validate_code(self, code: str, library: str = "auto") -> Dict[str, Any]:
        """Validate MIP code without executing it.
        
        Args:
            code: MIP code to validate
            library: Library to validate for
            
        Returns:
            Validation results
        """
        try:
            # Detect library
            if library == "auto":
                detected_library = self.detector.detect_library(code)
            else:
                detected_library = self.detector.validate_library_choice(library)
            
            if detected_library == MIPLibrary.UNKNOWN:
                return {
                    "status": "error",
                    "message": "Could not detect MIP library or unsupported library",
                    "issues": ["Unknown or unsupported MIP library"],
                    "library_detected": "unknown"
                }
            
            # Basic syntax validation
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                return {
                    "status": "error", 
                    "message": f"Syntax error: {e}",
                    "issues": [f"Syntax error at line {e.lineno}: {e.msg}"],
                    "library_detected": detected_library.value
                }
            
            # Pyodide-specific validation
            issues = []
            
            # Check for prohibited operations (should be minimal in Pyodide)
            prohibited_patterns = [
                ("import os", "Direct OS access not allowed"),
                ("import subprocess", "Subprocess not allowed"),
                ("open(", "Direct file operations not recommended (use content variables instead)")
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
                "sandbox": "pyodide"
            }
            
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return {
                "status": "error",
                "message": f"Validation failed: {e}",
                "issues": [str(e)],
                "library_detected": "unknown"
            }
    
    async def cleanup(self):
        """Clean up Pyodide process."""
        if self.pyodide_process:
            try:
                # Send exit command
                await self._communicate_with_pyodide({"action": "exit"})
                await asyncio.sleep(0.1)  # Give it time to exit gracefully
            except:
                pass  # Process might already be dead
            
            try:
                self.pyodide_process.terminate()
                await self.pyodide_process.wait()
            except:
                pass
            
            self.pyodide_process = None
            self._pyodide_initialized = False
            logger.info("Pyodide process cleaned up")
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.pyodide_process:
            # Note: Can't use await in __del__, so we'll just terminate
            try:
                self.pyodide_process.terminate()
            except:
                pass