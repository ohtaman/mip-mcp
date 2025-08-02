"""Response models for MCP tools."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExecutionResponse(BaseModel):
    """Response from code execution."""

    status: str = Field(
        description="Execution status: 'success', 'error', 'security_error', or 'execution_error'"
    )
    message: str = Field(
        description="Human-readable message about the execution result"
    )
    stdout: str = Field(description="Standard output from code execution")
    stderr: str = Field(description="Standard error output from code execution")
    solution: dict[str, Any] | None = Field(
        default=None, description="Optimization solution details if successful"
    )
    file_format: str | None = Field(
        default=None, description="File format used (auto, mps, or lp)"
    )
    library_used: str | None = Field(
        default=None, description="MIP library used (pulp)"
    )
    executor_used: str | None = Field(
        default=None, description="Executor used (pyodide for security)"
    )
    solver_info: dict[str, Any] | None = Field(
        default=None, description="Information about the solver used"
    )
    solver_output: str | None = Field(
        default=None, description="Detailed solver output (only included if requested)"
    )


class SolverInfo(BaseModel):
    """Information about a solver."""

    name: str = Field(description="Solver name")
    available: bool = Field(description="Whether the solver is available")
    version: str | None = Field(default=None, description="Solver version")
    description: str | None = Field(default=None, description="Solver description")
    capabilities: list[str] | None = Field(
        default=None, description="List of supported problem types"
    )
    file_formats: list[str] | None = Field(
        default=None, description="Supported file formats"
    )
    error: str | None = Field(
        default=None, description="Error message if solver not available"
    )

    @field_validator("version", mode="before")
    @classmethod
    def convert_version_to_string(cls, v):
        """Convert numeric version to string."""
        if v is None:
            return None
        return str(v)

    @property
    def supported_problem_types(self) -> list[str]:
        """Alias for capabilities for backward compatibility."""
        return self.capabilities or []

    @property
    def parameters(self) -> dict[str, Any]:
        """Parameters placeholder for backward compatibility."""
        return {}


class SolverInfoResponse(BaseModel):
    """Response from get_solver_info."""

    status: str = Field(description="Response status: 'success' or 'error'")
    solvers: dict[str, SolverInfo] = Field(description="Available solvers information")
    default_solver: str | None = Field(default=None, description="Default solver name")
    message: str | None = Field(
        default=None, description="Error message if status is 'error'"
    )


class ValidationIssue(BaseModel):
    """Code validation issue."""

    type: str = Field(description="Issue type: 'error', 'warning', 'security'")
    message: str = Field(description="Description of the issue")
    line: int | None = Field(description="Line number where issue occurs")
    column: int | None = Field(description="Column number where issue occurs")


class ValidationResponse(BaseModel):
    """Response from code validation."""

    status: str = Field(
        description="Validation status: 'success', 'error', or 'security_error'"
    )
    message: str = Field(description="Validation result message")
    issues: list[ValidationIssue] = Field(description="List of validation issues found")
    is_valid: bool | None = Field(description="Whether the code is valid")


class ExampleCode(BaseModel):
    """Example code snippet."""

    name: str = Field(description="Example name")
    description: str = Field(description="Description of what the example demonstrates")
    code: str = Field(description="The example code")
    library: str | None = Field(description="Library used in the example (pulp)")


class SolverProgress(BaseModel):
    """Progress information during optimization solving."""

    stage: str = Field(
        description="Current stage: 'modeling', 'initializing', 'presolving', 'solving', 'finished'"
    )
    time_elapsed: float = Field(description="Time elapsed in seconds")
    nodes_processed: int | None = Field(
        default=None, description="Number of branch-and-bound nodes processed"
    )
    best_solution: float | None = Field(
        default=None, description="Best primal bound (solution value) found so far"
    )
    best_bound: float | None = Field(
        default=None, description="Best dual bound found so far"
    )
    gap: float | None = Field(default=None, description="Optimality gap (0.0 to 1.0)")
    gap_percentage: float | None = Field(
        default=None, description="Optimality gap as percentage"
    )
    message: str | None = Field(default=None, description="Current status message")


class ProgressResponse(BaseModel):
    """Response containing progress information."""

    type: str = Field(
        default="progress", description="Response type for identification"
    )
    progress: SolverProgress = Field(description="Current solver progress information")
    is_final: bool = Field(
        default=False, description="Whether this is the final progress update"
    )


class ExamplesResponse(BaseModel):
    """Response from get_mip_examples."""

    status: str = Field(description="Response status: 'success' or 'error'")
    examples: dict[str, ExampleCode] = Field(
        description="Dictionary of example code snippets"
    )
    total_examples: int = Field(description="Total number of examples provided")
    message: str | None = Field(description="Error message if status is 'error'")
