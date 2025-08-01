"""Base solver abstraction layer."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from ..models.solution import OptimizationSolution


class SolverError(Exception):
    """Base class for solver-related errors."""
    pass


class BaseSolver(ABC):
    """Abstract base class for optimization solvers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the solver.
        
        Args:
            config: Solver configuration dictionary
        """
        self.config = config
        self.timeout = config.get("timeout", 3600)
        self.parameters = config.get("parameters", {})
    
    @abstractmethod
    async def solve_from_file(self, file_path: str) -> OptimizationSolution:
        """Solve optimization problem from file.
        
        Args:
            file_path: Path to MPS or LP file
            
        Returns:
            Optimization solution
            
        Raises:
            SolverError: If solving fails
        """
        pass
    
    @abstractmethod
    def get_solver_info(self) -> Dict[str, Any]:
        """Get solver information.
        
        Returns:
            Dictionary with solver name, version, capabilities
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set solver parameters.
        
        Args:
            params: Parameter dictionary
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """Validate optimization file format.
        
        Args:
            file_path: Path to optimization file
            
        Returns:
            True if file is valid
            
        Raises:
            SolverError: If file is invalid
        """
        path = Path(file_path)
        
        if not path.exists():
            raise SolverError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise SolverError(f"Path is not a file: {file_path}")
        
        # Check file extension
        suffix = path.suffix.lower()
        if suffix not in ['.mps', '.lp']:
            raise SolverError(f"Unsupported file format: {suffix}")
        
        # Basic file size check (prevent extremely large files)
        file_size = path.stat().st_size
        max_size = self.config.get("max_file_size", 100 * 1024 * 1024)  # 100MB default
        
        if file_size > max_size:
            raise SolverError(f"File too large: {file_size} bytes (max: {max_size})")
        
        return True
    
    def _extract_solution_status(self, solver_status: Any) -> str:
        """Extract standardized status from solver-specific status.
        
        Args:
            solver_status: Solver-specific status object
            
        Returns:
            Standardized status string
        """
        # This will be implemented by specific solvers
        return "unknown"
    
    def _is_optimal(self, status: str) -> bool:
        """Check if status indicates optimal solution."""
        return status.lower() in ["optimal", "opt", "optimum"]
    
    def _is_feasible(self, status: str) -> bool:
        """Check if status indicates feasible solution."""
        return status.lower() not in ["infeasible", "infeas", "unbounded"]