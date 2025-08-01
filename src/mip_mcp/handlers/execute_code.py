"""MCP handler for executing PuLP code and solving optimization problems."""

import tempfile
import os
from typing import Dict, Any, Optional
from pathlib import Path

from ..executor.code_executor import PuLPCodeExecutor, CodeExecutionError
from ..executor.sandbox import SecurityError
from ..solvers.scip_solver import SCIPSolver
from ..models.solution import OptimizationSolution
from ..utils.logger import get_logger

logger = get_logger(__name__)


async def execute_pulp_code_handler(
    code: str,
    data: Optional[Dict[str, Any]] = None,
    output_format: str = "mps",
    solver_params: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute PuLP code and solve the optimization problem.
    
    Args:
        code: PuLP Python code to execute
        data: Optional data dictionary to pass to the code
        output_format: File format for problem ("mps" or "lp")
        solver_params: Optional solver parameters
        config: Configuration dictionary
        
    Returns:
        Dictionary containing execution results and optimization solution
    """
    config = config or {}
    
    try:
        # Initialize components
        executor = PuLPCodeExecutor(config)
        solver = SCIPSolver(config.get("solvers", {}))
        
        # Apply solver parameters if provided
        if solver_params:
            solver.set_parameters(solver_params)
        
        logger.info(f"Executing PuLP code (format: {output_format})")
        
        # Execute PuLP code and generate optimization file
        stdout, stderr, file_path = await executor.execute_and_generate_files(
            code, data, output_format
        )
        
        if not file_path:
            return {
                "status": "error",
                "message": "No optimization file was generated",
                "stdout": stdout,
                "stderr": stderr,
                "solution": None
            }
        
        logger.info(f"Generated optimization file: {file_path}")
        
        # Solve the optimization problem
        solution = await solver.solve_from_file(file_path)
        
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {e}")
        
        # Prepare response
        response = {
            "status": "success",
            "message": "Code executed and problem solved successfully",
            "stdout": stdout,
            "stderr": stderr,
            "solution": solution.model_dump(),
            "file_format": output_format,
            "solver_info": solver.get_solver_info()
        }
        
        logger.info(f"Optimization completed: {solution.status}")
        return response
        
    except SecurityError as e:
        logger.error(f"Security error: {e}")
        return {
            "status": "security_error",
            "message": f"Code contains security violations: {e}",
            "stdout": "",
            "stderr": "",
            "solution": None
        }
    
    except CodeExecutionError as e:
        logger.error(f"Code execution error: {e}")
        return {
            "status": "execution_error",
            "message": f"Code execution failed: {e}",
            "stdout": "",
            "stderr": "",
            "solution": None
        }
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {e}",
            "stdout": "",
            "stderr": "",
            "solution": None
        }


async def get_solver_info_handler(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get information about available solvers.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with solver information
    """
    try:
        solver = SCIPSolver(config.get("solvers", {}) if config else {})
        solver_info = solver.get_solver_info()
        
        return {
            "status": "success",
            "solvers": {
                "scip": solver_info
            },
            "default_solver": "scip"
        }
    
    except Exception as e:
        logger.error(f"Failed to get solver info: {e}")
        return {
            "status": "error",
            "message": f"Failed to get solver information: {e}",
            "solvers": {},
            "default_solver": None
        }


async def validate_pulp_code_handler(
    code: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Validate PuLP code without executing it.
    
    Args:
        code: PuLP Python code to validate
        config: Configuration dictionary
        
    Returns:
        Dictionary with validation results
    """
    try:
        executor = PuLPCodeExecutor(config or {})
        
        # Only validate security, don't execute
        executor.security_checker.validate_code(code)
        
        return {
            "status": "valid",
            "message": "Code passed security validation",
            "issues": []
        }
    
    except SecurityError as e:
        return {
            "status": "invalid",
            "message": "Code has security issues",
            "issues": [str(e)]
        }
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "status": "error",
            "message": f"Validation failed: {e}",
            "issues": []
        }


async def get_pulp_examples_handler() -> Dict[str, Any]:
    """Get example PuLP code snippets.
    
    Returns:
        Dictionary with example code snippets
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
            "name": "Manual Content Setting",
            "description": "Example of setting MPS/LP content manually",
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
        }
    }
    
    return {
        "status": "success",
        "examples": examples,
        "total_examples": len(examples)
    }