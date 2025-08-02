"""SCIP solver implementation using pyscipopt."""

import os
import tempfile
import time
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, Callable
from pathlib import Path

try:
    import pyscipopt
except ImportError:
    pyscipopt = None

from .base import BaseSolver, SolverError
from ..models.solution import OptimizationSolution
from ..models.responses import SolverProgress, ProgressResponse
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
        self.progress_callback: Optional[Callable[[SolverProgress], None]] = None
        self.last_progress_time: float = 0.0
        self.progress_interval: float = 2.0  # Minimum 2 seconds between progress updates
        self.start_time: float = 0.0
    
    def set_progress_callback(self, callback: Optional[Callable[[SolverProgress], None]]) -> None:
        """Set progress callback function.
        
        Args:
            callback: Function to call with progress updates
        """
        self.progress_callback = callback
    
    def _should_send_progress(self) -> bool:
        """Check if enough time has passed to send progress update."""
        current_time = time.time()
        if current_time - self.last_progress_time >= self.progress_interval:
            self.last_progress_time = current_time
            return True
        return False
    
    def _send_progress(self, stage: str, message: Optional[str] = None) -> None:
        """Send progress update if callback is set and enough time has passed."""
        if not self.progress_callback or not self._should_send_progress():
            return
        
        try:
            time_elapsed = time.time() - self.start_time
            
            # Get solver statistics if model is available and solving
            nodes_processed = None
            best_solution = None
            best_bound = None
            gap = None
            gap_percentage = None
            
            if self.model and stage == "solving":
                try:
                    nodes_processed = self.model.getNNodes()
                    
                    # Try to get bounds
                    try:
                        best_solution = self.model.getPrimalbound()
                        if best_solution == float('inf') or best_solution == -float('inf'):
                            best_solution = None
                    except:
                        best_solution = None
                    
                    try:
                        best_bound = self.model.getDualbound()
                        if best_bound == float('inf') or best_bound == -float('inf'):
                            best_bound = None
                    except:
                        best_bound = None
                    
                    try:
                        gap = self.model.getGap()
                        if gap != float('inf') and gap >= 0:
                            gap_percentage = gap * 100.0
                        else:
                            gap = None
                    except:
                        gap = None
                        
                except Exception as e:
                    logger.debug(f"Could not get solver statistics: {e}")
            
            progress = SolverProgress(
                stage=stage,
                time_elapsed=time_elapsed,
                nodes_processed=nodes_processed,
                best_solution=best_solution,
                best_bound=best_bound,
                gap=gap,
                gap_percentage=gap_percentage,
                message=message
            )
            
            # Handle async callback properly
            if asyncio.iscoroutinefunction(self.progress_callback):
                # Create task for async callback
                asyncio.create_task(self.progress_callback(progress))
            else:
                self.progress_callback(progress)
            
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")
    
    async def _solve_with_progress_monitoring(self) -> None:
        """Solve the problem with progress monitoring."""
        def solve_in_executor():
            """Run the actual solving in executor."""
            return self.model.optimize()
        
        # Create a task for the actual solving
        loop = asyncio.get_event_loop()
        solving_task = loop.run_in_executor(None, solve_in_executor)
        
        # Monitor progress while solving
        while not solving_task.done():
            self._send_progress("solving", "Optimizing...")
            await asyncio.sleep(self.progress_interval)
        
        # Wait for solving to complete
        await solving_task
        
        # Send final progress
        self._send_progress("finished", "Optimization completed")
    
    async def solve_from_file(self, file_path: str, capture_output: bool = False) -> OptimizationSolution:
        """Solve optimization problem from MPS or LP file.
        
        Args:
            file_path: Path to MPS or LP file
            capture_output: If True, allow output for capturing; if False, suppress output
            
        Returns:
            Optimization solution
        """
        self.validate_file(file_path)
        
        try:
            # Initialize timing
            self.start_time = time.time()
            self.last_progress_time = 0.0
            
            # Send initial progress
            self._send_progress("initializing", "Setting up SCIP model")
            
            # Create SCIP model with conditional output suppression
            self.model = pyscipopt.Model("MIP_MCP_Problem")
            
            # Always suppress SCIP output to avoid MCP protocol pollution
            # Solver output will be captured via log file when needed
            self.model.hideOutput()
            
            # Note: When capture_output=True, we generate detailed output from solver statistics
            # No need for verbose solver parameters that can slow down solving
            
            # Set parameters
            self._apply_parameters()
            
            # Send progress update
            self._send_progress("initializing", "Loading problem file")
            
            # Read problem file
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == '.mps':
                self.model.readProblem(file_path)
            elif file_path_obj.suffix.lower() == '.lp':
                self.model.readProblem(file_path)
            else:
                raise SolverError(f"Unsupported file format: {file_path_obj.suffix}")
            
            logger.info(f"Problem loaded: {self.model.getNVars()} variables, {self.model.getNConss()} constraints")
            
            # Send presolving progress
            self._send_progress("presolving", f"Problem loaded: {self.model.getNVars()} variables, {self.model.getNConss()} constraints")
            
            # Start solving with progress monitoring
            self._send_progress("solving", "Starting optimization")
            
            # Setup a simple progress monitoring using asyncio
            solving_task = asyncio.create_task(self._solve_with_progress_monitoring())
            await solving_task
            
            # Extract solution
            solution = self._extract_solution(capture_output)
            
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
    
    def _extract_solution(self, capture_output: bool = False) -> OptimizationSolution:
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
            
            # Generate detailed solver output if requested
            solver_output = None
            if capture_output:
                solver_output = self._generate_solver_output_summary()
                solver_info["detailed_output"] = solver_output
            
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
    
    def _generate_solver_output_summary(self) -> str:
        """Generate a summary of solver output when detailed output is requested.
        
        Returns:
            Formatted string with solver statistics and information
        """
        if not self.model:
            return "No model available for output summary"
        
        try:
            output_lines = []
            output_lines.append("SCIP Optimization Summary")
            output_lines.append("=" * 50)
            output_lines.append(f"Problem Name: {self.model.getProbName()}")
            output_lines.append(f"Variables: {self.model.getNVars()}")
            output_lines.append(f"Constraints: {self.model.getNConss()}")
            output_lines.append(f"Status: {self.model.getStatus()}")
            
            # Solving statistics
            output_lines.append(f"Solving Time: {self.model.getSolvingTime():.3f} seconds")
            output_lines.append(f"Nodes Processed: {self.model.getNNodes()}")
            output_lines.append(f"Gap: {self.model.getGap():.6f}")
            
            # Objective value if available
            try:
                obj_val = self.model.getObjVal()
                output_lines.append(f"Objective Value: {obj_val}")
            except:
                output_lines.append("Objective Value: Not available")
            
            # Bounds information
            try:
                primal_bound = self.model.getPrimalbound()
                dual_bound = self.model.getDualbound()
                output_lines.append(f"Primal Bound: {primal_bound}")
                output_lines.append(f"Dual Bound: {dual_bound}")
            except:
                output_lines.append("Bound information: Not available")
            
            output_lines.append("=" * 50)
            
            return "\n".join(output_lines)
            
        except Exception as e:
            return f"Error generating solver output summary: {e}"