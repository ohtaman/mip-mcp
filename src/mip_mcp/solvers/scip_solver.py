"""SCIP solver implementation using pyscipopt."""

import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import pyscipopt
except ImportError:
    pyscipopt = None

from .base import BaseSolver, SolverError
from ..models.solution import OptimizationSolution
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SCIPSolver(BaseSolver):
    """SCIP solver implementation using pyscipopt."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SCIP solver.
        
        Args:
            config: Solver configuration
        """
        super().__init__(config)
        
        if pyscipopt is None:
            raise SolverError("pyscipopt is not available")
        
        self.model = None
    
    async def solve_from_file(self, file_path: str) -> OptimizationSolution:
        """Solve optimization problem from MPS or LP file.
        
        Args:
            file_path: Path to MPS or LP file
            
        Returns:
            Optimization solution
        """
        self.validate_file(file_path)
        
        try:
            # Create SCIP model with quiet output for MCP compatibility
            self.model = pyscipopt.Model("MIP_MCP_Problem")
            
            # Suppress SCIP output to avoid MCP protocol pollution
            self.model.hideOutput()
            
            # Set parameters
            self._apply_parameters()
            
            # Read problem file
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == '.mps':
                self.model.readProblem(file_path)
            elif file_path_obj.suffix.lower() == '.lp':
                self.model.readProblem(file_path)
            else:
                raise SolverError(f"Unsupported file format: {file_path_obj.suffix}")
            
            logger.info(f"Problem loaded: {self.model.getNVars()} variables, {self.model.getNConss()} constraints")
            
            # Solve the problem
            self.model.optimize()
            
            # Extract solution
            solution = self._extract_solution()
            
            logger.info(f"Optimization completed with status: {solution.status}")
            return solution
            
        except Exception as e:
            logger.error(f"SCIP solver failed: {e}")
            raise SolverError(f"SCIP solving failed: {e}") from e
        
        finally:
            # Clean up
            if self.model:
                # SCIP model cleanup is handled automatically
                self.model = None
    
    def _apply_parameters(self) -> None:
        """Apply solver parameters to SCIP model."""
        for param_name, param_value in self.parameters.items():
            try:
                self.model.setParam(param_name, param_value)
                logger.debug(f"Set SCIP parameter {param_name} = {param_value}")
            except Exception as e:
                logger.warning(f"Failed to set SCIP parameter {param_name}: {e}")
        
        # Set timeout
        if self.timeout > 0:
            self.model.setParam("limits/time", self.timeout)
    
    def _extract_solution(self) -> OptimizationSolution:
        """Extract solution from SCIP model.
        
        Returns:
            Optimization solution
        """
        if not self.model:
            raise SolverError("No model available for solution extraction")
        
        try:
            # Get solution status
            status = self._extract_solution_status(self.model.getStatus())
            
            # Get objective value
            objective_value = None
            if self._is_optimal(status) or self._is_feasible(status):
                try:
                    objective_value = self.model.getObjVal()
                except:
                    objective_value = None
            
            # Get variable values
            variables = {}
            if self._is_optimal(status) or self._is_feasible(status):
                try:
                    for var in self.model.getVars():
                        var_name = var.name
                        var_value = self.model.getVal(var)
                        variables[var_name] = var_value
                except Exception as e:
                    logger.warning(f"Failed to extract variable values: {e}")
            
            # Get solving statistics
            solve_time = self.model.getSolvingTime()
            
            # Get solver info
            solver_info = {
                "solver_name": "SCIP",
                "solver_version": self.model.version(),
                "nodes": self.model.getNNodes(),
                "gap": self.model.getGap(),
            }
            
            return OptimizationSolution(
                status=status,
                objective_value=objective_value,
                variables=variables,
                solve_time=solve_time,
                solver_info=solver_info
            )
            
        except Exception as e:
            logger.error(f"Failed to extract solution: {e}")
            return OptimizationSolution(
                status="error",
                message=f"Solution extraction failed: {e}"
            )
    
    def _extract_solution_status(self, scip_status) -> str:
        """Convert SCIP status to standardized status.
        
        Args:
            scip_status: SCIP status string
            
        Returns:
            Standardized status string
        """
        status_map = {
            "optimal": "optimal",
            "infeasible": "infeasible",
            "unbounded": "unbounded",
            "inforunbd": "infeasible_or_unbounded",
            "nodelimit": "node_limit",
            "timelimit": "time_limit",
            "memlimit": "memory_limit",
            "gaplimit": "gap_limit",
            "sollimit": "solution_limit",
            "bestsollimit": "best_solution_limit",
            "userinterrupt": "user_interrupt",
            "terminate": "terminated",
            "unknown": "unknown"
        }
        
        scip_status_str = str(scip_status).lower()
        return status_map.get(scip_status_str, "unknown")
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get SCIP solver information.
        
        Returns:
            Solver information dictionary
        """
        if pyscipopt is None:
            return {
                "name": "SCIP",
                "available": False,
                "error": "pyscipopt not installed"
            }
        
        try:
            # Create temporary model to get version info
            temp_model = pyscipopt.Model("temp")
            version = temp_model.version()
            
            return {
                "name": "SCIP",
                "available": True,
                "version": version,
                "description": "Solving Constraint Integer Programs",
                "capabilities": ["LP", "MIP", "MINLP"],
                "file_formats": ["mps", "lp"]
            }
        except Exception as e:
            return {
                "name": "SCIP",
                "available": False,
                "error": str(e)
            }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set SCIP parameters.
        
        Args:
            params: Parameter dictionary
        """
        self.parameters.update(params)
        
        # If model exists, apply parameters immediately
        if self.model:
            self._apply_parameters()