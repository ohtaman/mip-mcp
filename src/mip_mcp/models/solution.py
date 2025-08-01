"""Data models for optimization solutions."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class SolutionVariable(BaseModel):
    """Represents a variable in the optimization solution."""
    name: str
    value: float
    reduced_cost: Optional[float] = None


class SolutionConstraint(BaseModel):
    """Represents a constraint in the optimization solution."""
    name: str
    slack: Optional[float] = None
    dual_value: Optional[float] = None


class OptimizationSolution(BaseModel):
    """Represents the complete optimization solution."""
    status: str = Field(description="Solution status (optimal, infeasible, unbounded, error, etc.)")
    objective_value: Optional[float] = Field(None, description="Optimal objective function value")
    variables: Dict[str, float] = Field(default_factory=dict, description="Variable values by name")
    constraints: List[SolutionConstraint] = Field(default_factory=list, description="Constraint information")
    solve_time: Optional[float] = Field(None, description="Time taken to solve in seconds")
    iterations: Optional[int] = Field(None, description="Number of solver iterations")
    message: Optional[str] = Field(None, description="Additional solver message")
    solver_info: Optional[Dict[str, Any]] = Field(None, description="Solver-specific information")
    
    @property
    def is_optimal(self) -> bool:
        """Check if the solution is optimal."""
        return self.status.lower() in ["optimal", "opt", "optimum"]
    
    @property
    def is_feasible(self) -> bool:
        """Check if the solution is feasible."""
        return self.status.lower() not in ["infeasible", "infeas", "unbounded"]