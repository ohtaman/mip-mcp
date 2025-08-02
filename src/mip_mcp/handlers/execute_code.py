"""MCP handler for executing PuLP code and solving optimization problems."""

import tempfile
import os
import sys
import contextlib
from typing import Dict, Any, Optional
from pathlib import Path

from ..executor.pyodide_executor import PyodideExecutor
from ..exceptions import CodeExecutionError, SecurityError
from ..solvers.scip_solver import SCIPSolver
from ..models.solution import OptimizationSolution, SolutionValidation
from ..models.responses import ExecutionResponse, SolverInfoResponse, ValidationResponse, ExamplesResponse, SolverInfo, ValidationIssue, ExampleCode
from ..utils.solution_validator import SolutionValidator
from ..utils.library_detector import MIPLibrary
from ..utils.logger import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def suppress_stdout_for_mcp():
    """Suppress stdout in MCP mode to prevent protocol pollution."""
    # Check if we're in MCP mode by looking for CLI indicators
    is_mcp_mode = (
        os.environ.get('MCP_MODE') == '1' or
        '--mcp' in sys.argv or
        'mcp-server' in ' '.join(sys.argv)
    )
    
    if is_mcp_mode:
        # Redirect stdout to stderr to avoid JSON protocol pollution
        original_stdout = sys.stdout
        try:
            sys.stdout = sys.stderr
            yield
        finally:
            sys.stdout = original_stdout
    else:
        yield


async def execute_mip_code_handler(
    code: str,
    data: Optional[Dict[str, Any]] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    validate_solution: bool = True,
    validation_tolerance: float = 1e-6,
    config: Optional[Dict[str, Any]] = None
) -> ExecutionResponse:
    """Execute PuLP code and solve the optimization problem.
    
    Args:
        code: PuLP Python code to execute
        data: Optional data dictionary to pass to the code
        solver_params: Optional solver parameters
        validate_solution: Whether to validate solution against constraints
        validation_tolerance: Numerical tolerance for constraint validation
        config: Configuration dictionary
        
    Returns:
        Dictionary containing execution results and optimization solution.
        File format is automatically detected (LP preferred, then MPS).
        Always uses Pyodide WebAssembly sandbox for security.
    """
    config = config or {}
    
    try:
        # Always use Pyodide executor for security
        executor = PyodideExecutor(config)
        logger.info("Using Pyodide executor for secure execution")
        
        solver = SCIPSolver(config.get("solvers", {}))
        
        # Apply solver parameters if provided
        if solver_params:
            solver.set_parameters(solver_params)
        
        logger.info("Executing PuLP code in Pyodide sandbox")
        
        # Execute PuLP code and generate optimization file
        stdout, stderr, file_path, detected_library = await executor.execute_mip_code(
            code, data
        )
        
        if not file_path:
            return ExecutionResponse(
                status="error",
                message="No optimization file was generated",
                stdout=stdout,
                stderr=stderr,
                solution=None
            )
        
        logger.info(f"Generated optimization file: {file_path}")
        
        # Solve the optimization problem with stdout suppression for MCP
        with suppress_stdout_for_mcp():
            solution = await solver.solve_from_file(file_path)
        
        # Validate solution if requested and solution is optimal
        if validate_solution and solution.is_optimal:
            try:
                # For Pyodide, validation is not yet implemented
                # TODO: implement solution validation for Pyodide environment
                logger.info("Solution validation not yet implemented for Pyodide executor")
                problem_obj = None
                
                if problem_obj:
                    validator = SolutionValidator(validation_tolerance)
                    validation_result = validator.validate_solution(
                        problem_obj, 
                        solution.model_dump()
                    )
                    
                    # Create validation model and add to solution
                    solution.validation = SolutionValidation(**validation_result)
                    logger.info(f"Solution validation completed: valid={validation_result['is_valid']}")
                else:
                    logger.info("Solution validation skipped - not implemented for Pyodide yet")
                    
            except Exception as e:
                logger.error(f"Solution validation failed: {e}")
                solution.validation = SolutionValidation(
                    is_valid=False,
                    tolerance_used=validation_tolerance,
                    error=f"Validation failed: {e}"
                )
        
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {e}")
        
        # Prepare response
        response = ExecutionResponse(
            status="success",
            message="Code executed and problem solved successfully",
            stdout=stdout,
            stderr=stderr,
            solution=solution.model_dump(),
            file_format="auto",
            library_used=detected_library.value,
            executor_used="pyodide",
            solver_info=solver.get_solver_info()
        )
        
        logger.info(f"Optimization completed: {solution.status}")
        return response
        
    except Exception as e:
        # Clean up Pyodide if needed
        if isinstance(executor, PyodideExecutor):
            try:
                await executor.cleanup()
            except:
                pass
        
        if "security" in str(e).lower():
            logger.error(f"Security error: {e}")
            return ExecutionResponse(
                status="security_error",
                message=f"Code contains security violations: {e}",
                stdout="",
                stderr="",
                solution=None
            )
        raise
    
    except SecurityError as e:
        logger.error(f"Security error: {e}")
        return ExecutionResponse(
            status="security_error",
            message=f"Code contains security violations: {e}",
            stdout="",
            stderr="",
            solution=None
        )
    
    except CodeExecutionError as e:
        logger.error(f"Code execution error: {e}")
        return ExecutionResponse(
            status="execution_error",
            message=f"Code execution failed: {e}",
            stdout="",
            stderr="",
            solution=None
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ExecutionResponse(
            status="error",
            message=f"An unexpected error occurred: {e}",
            stdout="",
            stderr="",
            solution=None
        )


async def get_solver_info_handler(config: Optional[Dict[str, Any]] = None) -> SolverInfoResponse:
    """Get information about available solvers.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with solver information
    """
    try:
        solver = SCIPSolver(config.get("solvers", {}) if config else {})
        solver_info = solver.get_solver_info()
        
        return SolverInfoResponse(
            status="success",
            solvers={
                "scip": SolverInfo(**solver_info)
            },
            default_solver="scip"
        )
    
    except Exception as e:
        logger.error(f"Failed to get solver info: {e}")
        return SolverInfoResponse(
            status="error",
            message=f"Failed to get solver information: {e}",
            solvers={},
            default_solver=None
        )


async def validate_mip_code_handler(
    code: str,
    config: Optional[Dict[str, Any]] = None
) -> ValidationResponse:
    """Validate PuLP code without executing it.
    
    Args:
        code: PuLP Python code to validate
        config: Configuration dictionary
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Always use Pyodide executor for validation
        executor = PyodideExecutor(config or {})
        result = await executor.validate_code(code)
        
        # Convert dict result to ValidationResponse
        issues = []
        if 'issues' in result:
            for issue in result['issues']:
                issues.append(ValidationIssue(**issue))
        
        return ValidationResponse(
            status=result.get('status', 'error'),
            message=result.get('message', ''),
            issues=issues,
            is_valid=result.get('is_valid')
        )
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return ValidationResponse(
            status="error",
            message=f"Validation failed: {e}",
            issues=[]
        )


async def get_mip_examples_handler() -> ExamplesResponse:
    """Get example MIP code snippets for both PuLP and Python-MIP.
    
    Returns:
        Dictionary with example code snippets for both libraries
    """
    examples = {
        "linear_programming": {
            "name": "Basic Linear Programming",
            "description": "Simple LP problem with two variables - automatic file generation",
            "code": '''import pulp

# Create problem
prob = pulp.LpProblem("Example_LP", pulp.LpMaximize)

# Decision variables
x = pulp.LpVariable("x", 0, None)
y = pulp.LpVariable("y", 0, None)

# Objective function
prob += 3*x + 2*y

# Constraints
prob += 2*x + y <= 100
prob += x + y <= 80
prob += x <= 40

# Problem will be automatically converted to MPS/LP format'''
        },
        
        "integer_programming": {
            "name": "Integer Programming",
            "description": "IP problem with integer variables - automatic file generation",
            "code": '''import pulp

# Create problem
prob = pulp.LpProblem("Example_IP", pulp.LpMaximize)

# Integer decision variables
x = pulp.LpVariable("x", 0, None, pulp.LpInteger)
y = pulp.LpVariable("y", 0, None, pulp.LpInteger)

# Objective function
prob += 3*x + 2*y

# Constraints
prob += 2*x + y <= 100
prob += x + y <= 80

# Problem will be automatically converted to MPS/LP format'''
        },
        
        "knapsack": {
            "name": "Knapsack Problem", 
            "description": "Classic 0-1 knapsack problem - automatic file generation",
            "code": '''import pulp

# Data
items = ['item1', 'item2', 'item3', 'item4']
values = {'item1': 10, 'item2': 40, 'item3': 30, 'item4': 50}
weights = {'item1': 5, 'item2': 4, 'item3': 6, 'item4': 3}
capacity = 10

# Create problem
prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)

# Binary variables
x = {}
for item in items:
    x[item] = pulp.LpVariable(f"x_{item}", 0, 1, pulp.LpBinary)

# Objective function
prob += pulp.lpSum([values[item] * x[item] for item in items])

# Capacity constraint
prob += pulp.lpSum([weights[item] * x[item] for item in items]) <= capacity

# Problem will be automatically converted to MPS/LP format'''
        },
        
        "manual_content_setting": {
            "name": "Manual Content Setting (PuLP)",
            "description": "Example of setting MPS/LP content manually with PuLP",
            "code": '''import pulp

# Create a simple problem
prob = pulp.LpProblem("Manual_Example", pulp.LpMaximize)
x = pulp.LpVariable("x", 0, None)
y = pulp.LpVariable("y", 0, None)

prob += 3*x + 2*y
prob += 2*x + y <= 100
prob += x <= 40

# Manually set the MPS content (if needed)
# __mps_content__ = "NAME Manual_Example\\nROWS\\nN OBJ\\nL C1\\n..."

# Or let the system auto-generate from the problem object
# (recommended approach)'''
        },
        
        "python_mip_linear": {
            "name": "Linear Programming (Python-MIP)",
            "description": "Simple LP problem using Python-MIP library",
            "code": '''import mip

# Create model
model = mip.Model("Example_LP", sense=mip.MAXIMIZE)

# Decision variables
x = model.add_var("x", lb=0)
y = model.add_var("y", lb=0)

# Objective function
model.objective = 3*x + 2*y

# Constraints
model += 2*x + y <= 100
model += x + y <= 80
model += x <= 40

# Model will be automatically converted to MPS/LP format'''
        },
        
        "python_mip_integer": {
            "name": "Integer Programming (Python-MIP)",
            "description": "IP problem with integer variables using Python-MIP",
            "code": '''import mip

# Create model
model = mip.Model("Example_IP", sense=mip.MAXIMIZE)

# Integer decision variables
x = model.add_var("x", lb=0, var_type=mip.INTEGER)
y = model.add_var("y", lb=0, var_type=mip.INTEGER)

# Objective function
model.objective = 3*x + 2*y

# Constraints
model += 2*x + y <= 100
model += x + y <= 80

# Model will be automatically converted to MPS/LP format'''
        },
        
        "python_mip_knapsack": {
            "name": "Knapsack Problem (Python-MIP)",
            "description": "Classic 0-1 knapsack problem using Python-MIP",
            "code": '''import mip

# Data
items = ['item1', 'item2', 'item3', 'item4']
values = {'item1': 10, 'item2': 40, 'item3': 30, 'item4': 50}
weights = {'item1': 5, 'item2': 4, 'item3': 6, 'item4': 3}
capacity = 10

# Create model
model = mip.Model("Knapsack", sense=mip.MAXIMIZE)

# Binary variables
x = {}
for item in items:
    x[item] = model.add_var(f"x_{item}", var_type=mip.BINARY)

# Objective function
model.objective = mip.xsum(values[item] * x[item] for item in items)

# Capacity constraint
model += mip.xsum(weights[item] * x[item] for item in items) <= capacity

# Model will be automatically converted to MPS/LP format'''
        }
    }
    
    # Convert dict examples to ExampleCode models
    example_models = {}
    for key, ex in examples.items():
        example_models[key] = ExampleCode(
            name=ex["name"],
            description=ex["description"],
            code=ex["code"],
            library="pulp" if "pulp" in ex["code"] else "python-mip" if "mip" in ex["code"] else None
        )
    
    return ExamplesResponse(
        status="success",
        examples=example_models,
        total_examples=len(examples)
    )