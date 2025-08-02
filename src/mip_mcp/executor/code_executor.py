"""Python code executor for PuLP optimization."""

import sys
import io
import contextlib
import tempfile
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .pyodide_executor import SecurityError
from ..models.solution import OptimizationSolution
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CodeExecutionError(Exception):
    """Error during code execution."""
    pass


class PuLPCodeExecutor:
    """Executes PuLP code safely and extracts optimization results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        # NOTE: Security checking is now handled by Pyodide WebAssembly sandbox
        # This executor is legacy and should be phased out in favor of PyodideExecutor
        self.timeout = config.get("executor", {}).get("timeout", 300)
    
    async def execute_pulp_code(
        self, 
        code: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Optional[str]]:
        """Execute PuLP code and return generated files.
        
        Args:
            code: PuLP Python code to execute
            data: Optional data dictionary to make available to the code
            
        Returns:
            Tuple of (stdout_output, stderr_output, generated_file_path)
            
        Raises:
            SecurityError: If code fails security validation
            CodeExecutionError: If code execution fails
        """
        # NOTE: Security validation is now handled by Pyodide WebAssembly sandbox
        # This legacy executor should be replaced by PyodideExecutor for production use
        
        # Prepare execution environment
        namespace = self._prepare_namespace(data)
        
        # Add output variables for MPS/LP content
        namespace['__mps_content__'] = None
        namespace['__lp_content__'] = None
            
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    # Execute the code
                    exec(code, namespace)
            
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            logger.info(f"Code executed successfully. Output length: {len(stdout_output)}")
            
            # Check for output content in variables
            generated_file = self._save_output_content(namespace)
            
            return stdout_output, stderr_output, generated_file
            
        except Exception as e:
            stderr_output = stderr_capture.getvalue()
            logger.error(f"Code execution failed: {e}")
            raise CodeExecutionError(f"Execution failed: {e}") from e
    
    def _prepare_namespace(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare execution namespace with safe builtins and PuLP."""
        namespace = {
            '__builtins__': self._get_safe_builtins(),
            'data': data or {},
        }
        
        # Import PuLP safely
        try:
            import pulp
            namespace['pulp'] = pulp
            
            # Also make common PuLP classes available directly
            namespace['LpProblem'] = pulp.LpProblem
            namespace['LpVariable'] = pulp.LpVariable
            namespace['LpMaximize'] = pulp.LpMaximize
            namespace['LpMinimize'] = pulp.LpMinimize
            namespace['LpStatus'] = pulp.LpStatus
            namespace['value'] = pulp.value
            
        except ImportError:
            raise CodeExecutionError("PuLP library not available")
        
        # Add safe math libraries
        try:
            import math
            namespace['math'] = math
        except ImportError:
            pass
        
        return namespace
    
    def _get_safe_builtins(self) -> Dict[str, Any]:
        """Get a safe subset of builtin functions."""
        safe_builtins = {
            # Basic types
            'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set',
            # Utility functions
            'len', 'range', 'enumerate', 'zip', 'sum', 'min', 'max',
            'abs', 'round', 'sorted', 'reversed',
            # I/O (limited)
            'print',
            # Type checking
            'isinstance', 'type',
            # Iteration
            'iter', 'next', 'all', 'any',
            # Import (controlled)
            '__import__',
        }
        
        builtin_dict = {}
        builtins_source = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        
        for name in safe_builtins:
            if name in builtins_source:
                builtin_dict[name] = builtins_source[name]
        
        return builtin_dict
    
    def _save_output_content(self, namespace: Dict[str, Any]) -> Optional[str]:
        """Save output content from variables to temporary file.
        
        Args:
            namespace: Execution namespace
            
        Returns:
            Path to saved file if content exists, None otherwise
        """
        # Check for MPS content first
        if namespace.get('__mps_content__'):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as f:
                f.write(namespace['__mps_content__'])
                logger.info(f"Saved MPS content to: {f.name}")
                return f.name
        
        # Check for LP content
        if namespace.get('__lp_content__'):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
                f.write(namespace['__lp_content__'])
                logger.info(f"Saved LP content to: {f.name}")
                return f.name
        
        return None
    
    def extract_problem_from_pulp(self, namespace: Dict[str, Any]) -> Optional[Any]:
        """Extract PuLP problem object from execution namespace.
        
        Args:
            namespace: Execution namespace after code execution
            
        Returns:
            PuLP problem object if found, None otherwise
        """
        import pulp
        
        # Look for PuLP problem objects in namespace
        for var_name, obj in namespace.items():
            if isinstance(obj, pulp.LpProblem):
                logger.info(f"Found PuLP problem: {var_name}")
                return obj
        
        return None
    
    def generate_mps_file(self, problem: Any, output_path: str) -> str:
        """Generate MPS file from PuLP problem.
        
        Args:
            problem: PuLP problem object
            output_path: Path where to save the MPS file
            
        Returns:
            Path to generated MPS file
        """
        try:
            problem.writeMPS(output_path)
            logger.info(f"Generated MPS file: {output_path}")
            return output_path
        except Exception as e:
            raise CodeExecutionError(f"Failed to generate MPS file: {e}")
    
    def generate_lp_file(self, problem: Any, output_path: str) -> str:
        """Generate LP file from PuLP problem.
        
        Args:
            problem: PuLP problem object
            output_path: Path where to save the LP file
            
        Returns:
            Path to generated LP file
        """
        try:
            problem.writeLP(output_path)
            logger.info(f"Generated LP file: {output_path}")
            return output_path
        except Exception as e:
            raise CodeExecutionError(f"Failed to generate LP file: {e}")
    
    async def execute_and_generate_files(
        self, 
        code: str, 
        data: Optional[Dict[str, Any]] = None,
        format: str = "mps"
    ) -> Tuple[str, str, Optional[str]]:
        """Execute PuLP code and generate optimization files.
        
        Args:
            code: PuLP Python code
            data: Optional data dictionary
            format: Output format ("mps" or "lp")
            
        Returns:
            Tuple of (stdout, stderr, file_path)
        """
        # Execute the code first
        stdout, stderr, generated_file = await self.execute_pulp_code(code, data)
        
        # If no content was provided in variables, try to extract PuLP problem and generate content
        if not generated_file:
            namespace = self._prepare_namespace(data)
            
            try:
                exec(code, namespace)
                problem = self.extract_problem_from_pulp(namespace)
                
                if problem:
                    # Generate content and save to temporary file
                    if format.lower() == "mps":
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as f:
                            problem.writeMPS(f.name)
                            generated_file = f.name
                    else:  # lp format
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
                            problem.writeLP(f.name)
                            generated_file = f.name
                
            except Exception as e:
                logger.error(f"Failed to extract problem and generate file: {e}")
                stderr += f"\nFile generation error: {e}"
        
        return stdout, stderr, generated_file