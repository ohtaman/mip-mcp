"""Universal MIP code executor supporting multiple libraries."""

from typing import Dict, Any, Optional, Tuple
from ..utils.library_detector import MIPLibraryDetector, MIPLibrary
from .code_executor import PuLPCodeExecutor, CodeExecutionError
from .python_mip_executor import PythonMIPCodeExecutor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UniversalMIPExecutor:
    """Universal executor that supports multiple MIP libraries."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the universal executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.detector = MIPLibraryDetector()
        
        # Initialize library-specific executors
        self.executors = {
            MIPLibrary.PULP: PuLPCodeExecutor(config),
            MIPLibrary.PYTHON_MIP: PythonMIPCodeExecutor(config)
        }
        
        logger.info("Universal MIP executor initialized")
    
    async def execute_and_generate_files(
        self,
        code: str,
        data: Optional[Dict[str, Any]] = None,
        library: str = "auto"
    ) -> Tuple[str, str, Optional[str], MIPLibrary]:
        """Execute MIP code using appropriate library executor.
        
        Args:
            code: MIP code to execute
            data: Optional data dictionary
            library: Library to use ("auto", "pulp", "python-mip")
            
        Returns:
            Tuple of (stdout, stderr, file_path, detected_library).
            File format is automatically determined by the executor.
        """
        # Determine which library to use
        detected_library = self._determine_library(code, library)
        
        if detected_library == MIPLibrary.UNKNOWN:
            raise CodeExecutionError(
                f"Unable to determine MIP library. Supported libraries: {self.detector.get_supported_libraries()}"
            )
        
        # Get appropriate executor
        executor = self.executors.get(detected_library)
        if not executor:
            raise CodeExecutionError(f"No executor available for library: {detected_library.value}")
        
        logger.info(f"Executing code with {detected_library.value} executor")
        
        try:
            # Execute using library-specific method (let executors determine format)
            if detected_library == MIPLibrary.PULP:
                stdout, stderr, file_path = await executor.execute_and_generate_files(
                    code, data
                )
            elif detected_library == MIPLibrary.PYTHON_MIP:
                stdout, stderr, file_path = await executor.execute_and_generate_files(
                    code, data
                )
            
            return stdout, stderr, file_path, detected_library
            
        except Exception as e:
            logger.error(f"Execution failed with {detected_library.value}: {e}")
            raise CodeExecutionError(f"Execution failed: {e}") from e
    
    def _determine_library(self, code: str, library_preference: str) -> MIPLibrary:
        """Determine which library to use for execution.
        
        Args:
            code: Code to analyze
            library_preference: User preference ("auto", "pulp", "python-mip")
            
        Returns:
            MIPLibrary to use
        """
        # If user specified a library, validate and use it
        if library_preference.lower() != "auto":
            preferred_library = self.detector.validate_library_choice(library_preference)
            if preferred_library != MIPLibrary.UNKNOWN:
                # Verify the code is compatible with the chosen library
                detected = self.detector.detect_library(code)
                if detected == preferred_library or detected == MIPLibrary.UNKNOWN:
                    logger.info(f"Using user-specified library: {preferred_library.value}")
                    return preferred_library
                else:
                    logger.warning(
                        f"Code appears to use {detected.value} but user specified {preferred_library.value}"
                    )
                    # Use user preference anyway, let execution fail if incompatible
                    return preferred_library
        
        # Auto-detect library from code
        detected_library = self.detector.detect_library(code)
        logger.info(f"Auto-detected library: {detected_library.value}")
        return detected_library
    
    def extract_problem_from_namespace(
        self, 
        namespace: Dict[str, Any], 
        library: MIPLibrary
    ) -> Optional[Any]:
        """Extract problem object from execution namespace.
        
        Args:
            namespace: Execution namespace
            library: Library type
            
        Returns:
            Problem object if found, None otherwise
        """
        executor = self.executors.get(library)
        if not executor:
            return None
        
        if library == MIPLibrary.PULP:
            return executor.extract_problem_from_pulp(namespace)
        elif library == MIPLibrary.PYTHON_MIP:
            return executor.extract_model_from_namespace(namespace)
        
        return None
    
    def get_supported_libraries(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported libraries.
        
        Returns:
            Dictionary with library information
        """
        return {
            "pulp": {
                "name": "PuLP",
                "description": "Linear programming library for Python",
                "executor_available": MIPLibrary.PULP in self.executors,
                "import_example": "import pulp"
            },
            "python-mip": {
                "name": "Python-MIP",
                "description": "Mixed-Integer Linear Programming library for Python",
                "executor_available": MIPLibrary.PYTHON_MIP in self.executors,
                "import_example": "import mip"
            }
        }
    
    async def validate_code(self, code: str, library: str = "auto") -> Dict[str, Any]:
        """Validate code for security and library compatibility.
        
        Args:
            code: Code to validate
            library: Library preference
            
        Returns:
            Validation results
        """
        try:
            # Determine library
            detected_library = self._determine_library(code, library)
            
            if detected_library == MIPLibrary.UNKNOWN:
                return {
                    "status": "invalid",
                    "message": "Unable to determine MIP library",
                    "issues": ["No supported MIP library detected"],
                    "supported_libraries": self.detector.get_supported_libraries()
                }
            
            # Get appropriate executor for validation
            executor = self.executors.get(detected_library)
            if not executor:
                return {
                    "status": "invalid",
                    "message": f"No executor available for {detected_library.value}",
                    "issues": [f"Library {detected_library.value} not supported"]
                }
            
            # Use security checker from the executor
            executor.security_checker.validate_code(code)
            
            return {
                "status": "valid",
                "message": "Code passed security validation",
                "library": detected_library.value,
                "issues": []
            }
            
        except Exception as e:
            return {
                "status": "invalid",
                "message": f"Validation failed: {e}",
                "issues": [str(e)]
            }