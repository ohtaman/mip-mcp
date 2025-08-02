"""Data models for optimization solutions."""

from typing import Any

from pydantic import BaseModel, Field


class SolutionVariable(BaseModel):
    """Represents a variable in the optimization solution."""

    name: str
    value: float
    reduced_cost: float | None = None


class SolutionConstraint(BaseModel):
    """Represents a constraint in the optimization solution."""

    name: str
    slack: float | None = None
    dual_value: float | None = None


class ValidationViolation(BaseModel):
    """Represents a constraint or bound violation in solution validation."""

    type: str = Field(description="Type of violation (constraint, bound, integer)")
    description: str = Field(description="Human-readable description of the violation")
    severity: str = Field(description="Severity level (error, warning)")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional violation details"
    )


class SolutionValidation(BaseModel):
    """Represents the validation results for an optimization solution."""

    is_valid: bool = Field(description="Whether the solution satisfies all constraints")
    tolerance_used: float = Field(description="Numerical tolerance used for validation")
    constraint_violations: list[dict[str, Any]] = Field(
        default_factory=list, description="Linear constraint violations"
    )
    bound_violations: list[dict[str, Any]] = Field(
        default_factory=list, description="Variable bound violations"
    )
    integer_violations: list[dict[str, Any]] = Field(
        default_factory=list, description="Integer constraint violations"
    )
    summary: dict[str, int] = Field(
        default_factory=dict, description="Validation summary statistics"
    )
    error: str | None = Field(
        None, description="Validation error message if validation failed"
    )


class OptimizationSolution(BaseModel):
    """Represents the complete optimization solution."""

    status: str = Field(
        description="Solution status (optimal, infeasible, unbounded, error, etc.)"
    )
    objective_value: float | None = Field(
        None, description="Optimal objective function value"
    )
    variables: dict[str, float] = Field(
        default_factory=dict, description="Variable values by name"
    )
    constraints: list[SolutionConstraint] = Field(
        default_factory=list, description="Constraint information"
    )
    solve_time: float | None = Field(None, description="Time taken to solve in seconds")
    iterations: int | None = Field(None, description="Number of solver iterations")
    message: str | None = Field(None, description="Additional solver message")
    solver_info: dict[str, Any] | None = Field(
        None, description="Solver-specific information"
    )
    validation: SolutionValidation | None = Field(
        None, description="Solution validation results"
    )

    @property
    def is_optimal(self) -> bool:
        """Check if the solution is optimal."""
        return self.status.lower() in ["optimal", "opt", "optimum"]

    @property
    def is_feasible(self) -> bool:
        """Check if the solution is feasible."""
        return self.status.lower() not in ["infeasible", "infeas", "unbounded"]
