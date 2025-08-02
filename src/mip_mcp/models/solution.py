"""Data models for optimization solutions."""

from typing import Dict, Any, Optional, List, Union
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


class ValidationViolation(BaseModel):
    """Represents a constraint or bound violation in solution validation."""
    type: str = Field(description="Type of violation (constraint, bound, integer)")
    description: str = Field(description="Human-readable description of the violation")
    severity: str = Field(description="Severity level (error, warning)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional violation details")


class SolutionValidation(BaseModel):
    """Represents the validation results for an optimization solution."""
    is_valid: bool = Field(description="Whether the solution satisfies all constraints")
    tolerance_used: float = Field(description="Numerical tolerance used for validation")
    constraint_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Linear constraint violations")
    bound_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Variable bound violations")
    integer_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Integer constraint violations")
    summary: Dict[str, int] = Field(default_factory=dict, description="Validation summary statistics")
    error: Optional[str] = Field(None, description="Validation error message if validation failed")


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
    validation: Optional[SolutionValidation] = Field(None, description="Solution validation results")
    
    @property
    def is_optimal(self) -> bool:
        """Check if the solution is optimal."""
        return self.status.lower() in ["optimal", "opt", "optimum"]
    
    @property
    def is_feasible(self) -> bool:
        """Check if the solution is feasible."""
        return self.status.lower() not in ["infeasible", "infeas", "unbounded"]