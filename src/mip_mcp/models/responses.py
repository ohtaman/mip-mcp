"""Response models for MCP tools."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class ExecutionResponse(BaseModel):
    """Response from code execution."""
    
    status: str = Field(description="Execution status: 'success', 'error', 'security_error', or 'execution_error'")
    message: str = Field(description="Human-readable message about the execution result")
    stdout: str = Field(description="Standard output from code execution")
    stderr: str = Field(description="Standard error output from code execution")
    solution: Optional[Dict[str, Any]] = Field(default=None, description="Optimization solution details if successful")
    file_format: Optional[str] = Field(default=None, description="File format used (auto, mps, or lp)")
    library_used: Optional[str] = Field(default=None, description="MIP library used (pulp)")
    executor_used: Optional[str] = Field(default=None, description="Executor used (pyodide for security)")
    solver_info: Optional[Dict[str, Any]] = Field(default=None, description="Information about the solver used")
    solver_output: Optional[str] = Field(default=None, description="Detailed solver output (only included if requested)")


class SolverInfo(BaseModel):
    """Information about a solver."""
    
    name: str = Field(description="Solver name")
    available: bool = Field(description="Whether the solver is available")
    version: Optional[str] = Field(default=None, description="Solver version")
    description: Optional[str] = Field(default=None, description="Solver description")
    capabilities: Optional[List[str]] = Field(default=None, description="List of supported problem types")
    file_formats: Optional[List[str]] = Field(default=None, description="Supported file formats")
    error: Optional[str] = Field(default=None, description="Error message if solver not available")
    
    @field_validator('version', mode='before')
    @classmethod
    def convert_version_to_string(cls, v):
        """Convert numeric version to string."""
        if v is None:
            return None
        return str(v)
    
    @property
    def supported_problem_types(self) -> List[str]:
        """Alias for capabilities for backward compatibility."""
        return self.capabilities or []
    
    @property 
    def parameters(self) -> Dict[str, Any]:
        """Parameters placeholder for backward compatibility."""
        return {}


class SolverInfoResponse(BaseModel):
    """Response from get_solver_info."""
    
    status: str = Field(description="Response status: 'success' or 'error'")
    solvers: Dict[str, SolverInfo] = Field(description="Available solvers information")
    default_solver: Optional[str] = Field(default=None, description="Default solver name")
    message: Optional[str] = Field(default=None, description="Error message if status is 'error'")


class ValidationIssue(BaseModel):
    """Code validation issue."""
    
    type: str = Field(description="Issue type: 'error', 'warning', 'security'")
    message: str = Field(description="Description of the issue")
    line: Optional[int] = Field(description="Line number where issue occurs")
    column: Optional[int] = Field(description="Column number where issue occurs")


class ValidationResponse(BaseModel):
    """Response from code validation."""
    
    status: str = Field(description="Validation status: 'success', 'error', or 'security_error'")
    message: str = Field(description="Validation result message")
    issues: List[ValidationIssue] = Field(description="List of validation issues found")
    is_valid: Optional[bool] = Field(description="Whether the code is valid")


class ExampleCode(BaseModel):
    """Example code snippet."""
    
    name: str = Field(description="Example name")
    description: str = Field(description="Description of what the example demonstrates")
    code: str = Field(description="The example code")
    library: Optional[str] = Field(description="Library used in the example (pulp)")


class SolverProgress(BaseModel):
    """Progress information during optimization solving."""
    
    stage: str = Field(description="Current stage: 'modeling', 'initializing', 'presolving', 'solving', 'finished'")
    time_elapsed: float = Field(description="Time elapsed in seconds")
    nodes_processed: Optional[int] = Field(default=None, description="Number of branch-and-bound nodes processed")
    best_solution: Optional[float] = Field(default=None, description="Best primal bound (solution value) found so far")
    best_bound: Optional[float] = Field(default=None, description="Best dual bound found so far")
    gap: Optional[float] = Field(default=None, description="Optimality gap (0.0 to 1.0)")
    gap_percentage: Optional[float] = Field(default=None, description="Optimality gap as percentage")
    message: Optional[str] = Field(default=None, description="Current status message")


class ProgressResponse(BaseModel):
    """Response containing progress information."""
    
    type: str = Field(default="progress", description="Response type for identification")
    progress: SolverProgress = Field(description="Current solver progress information")
    is_final: bool = Field(default=False, description="Whether this is the final progress update")


class ExamplesResponse(BaseModel):
    """Response from get_mip_examples."""
    
    status: str = Field(description="Response status: 'success' or 'error'")
    examples: Dict[str, ExampleCode] = Field(description="Dictionary of example code snippets")
    total_examples: int = Field(description="Total number of examples provided")
    message: Optional[str] = Field(description="Error message if status is 'error'")