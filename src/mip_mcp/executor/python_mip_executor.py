"""Python-MIP code executor for optimization problems."""

import sys
import io
import contextlib
import tempfile
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .sandbox import SecurityChecker, SecurityError
from ..models.solution import OptimizationSolution
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PythonMIPCodeExecutor:
    """Executes Python-MIP code safely and extracts optimization results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.security_checker = SecurityChecker()
        self.timeout = config.get("executor", {}).get("timeout", 300)
    
    async def execute_mip_code(
        self, 
        code: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Optional[str]]:
        """Execute Python-MIP code and return generated files.
        
        Args:
            code: Python-MIP code to execute
            data: Optional data dictionary to make available to the code
            
        Returns:
            Tuple of (stdout_output, stderr_output, generated_file_path)
            
        Raises:
            SecurityError: If code fails security validation
            CodeExecutionError: If code execution fails
        """
        # Validate code security
        self.security_checker.validate_code(code)
        
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
            
            logger.info(f"Python-MIP code executed successfully. Output length: {len(stdout_output)}")
            
            # Check for output content in variables
            generated_file = self._save_output_content(namespace)
            
            return stdout_output, stderr_output, generated_file
            
        except Exception as e:
            stderr_output = stderr_capture.getvalue()
            logger.error(f"Python-MIP code execution failed: {e}")
            raise CodeExecutionError(f"Execution failed: {e}") from e
    
    def _prepare_namespace(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare execution namespace with safe builtins and Python-MIP."""
        namespace = {
            '__builtins__': self._get_safe_builtins(),
            'data': data or {},
        }
        
        # Import Python-MIP safely
        try:
            import mip
            namespace['mip'] = mip
            
            # Also make common Python-MIP classes available directly
            namespace['Model'] = mip.Model
            namespace['MAXIMIZE'] = mip.MAXIMIZE
            namespace['MINIMIZE'] = mip.MINIMIZE
            namespace['BINARY'] = mip.BINARY
            namespace['INTEGER'] = mip.INTEGER
            namespace['CONTINUOUS'] = mip.CONTINUOUS
            
        except ImportError:
            raise CodeExecutionError("Python-MIP library not available")
        
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
    
    def extract_model_from_namespace(self, namespace: Dict[str, Any]) -> Optional[Any]:
        """Extract Python-MIP model object from execution namespace.
        
        Args:
            namespace: Execution namespace after code execution
            
        Returns:
            Python-MIP model object if found, None otherwise
        """
        try:
            import mip
            
            # Look for Python-MIP model objects in namespace
            for var_name, obj in namespace.items():
                if isinstance(obj, mip.Model):
                    logger.info(f"Found Python-MIP model: {var_name}")
                    return obj
            
            return None
            
        except ImportError:
            logger.error("Python-MIP not available for model extraction")
            return None
    
    def generate_mps_file(self, model: Any, output_path: str) -> str:
        """Generate MPS file from Python-MIP model.
        
        Args:
            model: Python-MIP model object
            output_path: Path where to save the MPS file
            
        Returns:
            Path to generated MPS file
        """
        try:
            model.write(output_path)
            logger.info(f"Generated MPS file: {output_path}")
            return output_path
        except Exception as e:
            raise CodeExecutionError(f"Failed to generate MPS file: {e}")
    
    def generate_lp_file(self, model: Any, output_path: str) -> str:
        """Generate LP file from Python-MIP model.
        
        Args:
            model: Python-MIP model object
            output_path: Path where to save the LP file
            
        Returns:
            Path to generated LP file
        """
        try:
            # Python-MIP uses .write() for both formats, determined by extension
            if not output_path.endswith('.lp'):
                output_path = output_path.replace('.mps', '.lp')
            
            model.write(output_path)
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
        """Execute Python-MIP code and generate optimization files.
        
        Args:
            code: Python-MIP code
            data: Optional data dictionary
            format: Output format ("mps" or "lp")
            
        Returns:
            Tuple of (stdout, stderr, file_path)
        """
        # Execute the code first
        stdout, stderr, generated_file = await self.execute_mip_code(code, data)
        
        # If no content was provided in variables, try to extract Python-MIP model
        if not generated_file:
            namespace = self._prepare_namespace(data)
            
            try:
                exec(code, namespace)
                model = self.extract_model_from_namespace(namespace)
                
                if model:
                    # Generate content and save to temporary file
                    with tempfile.NamedTemporaryFile(
                        mode='w', 
                        suffix=f'.{format}', 
                        delete=False
                    ) as f:
                        if format.lower() == "mps":
                            generated_file = self.generate_mps_file(model, f.name)
                        else:  # lp format
                            generated_file = self.generate_lp_file(model, f.name)
                
            except Exception as e:
                logger.error(f"Failed to extract model and generate file: {e}")
                stderr += f"\nFile generation error: {e}"
        
        return stdout, stderr, generated_file


# Import the CodeExecutionError from the existing module
from .code_executor import CodeExecutionError