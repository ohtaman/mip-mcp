"""MCP handler for executing PuLP code and solving optimization problems."""

import os
import sys
import contextlib
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, Union

from ..executor.pyodide_executor import PyodideExecutor
from ..exceptions import CodeExecutionError, SecurityError
from ..solvers.factory import SolverFactory
from ..models.solution import SolutionValidation
from ..models.responses import ExecutionResponse, SolverInfoResponse, ValidationResponse, ExamplesResponse, SolverInfo, ValidationIssue, ExampleCode, ProgressResponse, SolverProgress
from ..utils.solution_validator import SolutionValidator
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
    solver: Optional[str] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    validate_solution: bool = True,
    validation_tolerance: float = 1e-6,
    include_solver_output: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> ExecutionResponse:
    """Execute PuLP code and solve the optimization problem.
    
    Args:
        code: PuLP Python code to execute
        data: Optional data dictionary to pass to the code
        solver: Solver to use (default: from config, fallback: "scip")
        solver_params: Optional solver parameters
        validate_solution: Whether to validate solution against constraints
        validation_tolerance: Numerical tolerance for constraint validation
        include_solver_output: Whether to include detailed solver output in response
        config: Configuration dictionary
        
    Returns:
        Dictionary containing execution results and optimization solution.
        File format is automatically detected (LP preferred, then MPS).
        Always uses Pyodide WebAssembly sandbox for security.
    """
    # Use the non-streaming version for regular MCP calls
    async for result in execute_mip_code_with_progress(
        code, data, solver, solver_params, validate_solution, validation_tolerance, 
        include_solver_output, config
    ):
        if isinstance(result, ExecutionResponse):
            return result
    
    # Fallback (should not reach here)
    return ExecutionResponse(
        status="error",
        message="Unexpected error in execution flow",
        stdout="",
        stderr="",
        solution=None
    )


async def execute_mip_code_with_mcp_progress(
    code: str,
    mcp_context,  # FastMCP Context object
    data: Optional[Dict[str, Any]] = None,
    solver: Optional[str] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    validate_solution: bool = True,
    validation_tolerance: float = 1e-6,
    include_solver_output: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> ExecutionResponse:
    """Execute PuLP code with FastMCP-compatible progress reporting.
    
    Args:
        code: PuLP Python code to execute
        mcp_context: FastMCP Context for progress reporting
        data: Optional data dictionary to pass to the code
        solver: Solver to use (default: from config, fallback: "scip")
        solver_params: Optional solver parameters
        validate_solution: Whether to validate solution against constraints
        validation_tolerance: Numerical tolerance for constraint validation
        include_solver_output: Whether to include detailed solver output in response
        config: Configuration dictionary
        
    Returns:
        ExecutionResponse containing execution results and optimization solution.
    """
    config = config or {}
    
    # FastMCP progress callback
    async def mcp_progress_callback(progress: SolverProgress):
        """Callback to send progress via FastMCP Context."""
        try:
            # Convert stage to progress percentage (rough estimate)
            stage_progress = {
                "modeling": 30.0,
                "initializing": 40.0,
                "presolving": 50.0,
                "solving": 80.0,
                "finished": 100.0
            }
            
            current_progress = stage_progress.get(progress.stage, 0.0)
            
            # Add time-based progress for long-running operations
            if "elapsed" in (progress.message or ""):
                # For long operations, show continuous progress
                time_factor = min(progress.time_elapsed / 30.0, 0.5)  # Up to 50% extra over 30s
                current_progress += time_factor * 50.0
            
            current_progress = min(current_progress, 100.0)
            
            # Send progress via FastMCP
            logger.info(f"Sending MCP progress: {current_progress}% - {progress.stage}")
            await mcp_context.report_progress(
                progress=current_progress,
                total=100.0,
                message=f"[{progress.stage}] {progress.message or 'Processing...'}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to send MCP progress update: {e}")

    try:
        # Always use Pyodide executor for security
        executor = PyodideExecutor(config)
        logger.info("Using Pyodide executor for secure execution")
        
        # Initial progress report
        await mcp_progress_callback(SolverProgress(
            stage="initializing",
            time_elapsed=0.0,
            message="Setting up execution environment"
        ))
        
        # Set up progress callback for executor (modeling stage)
        executor.set_progress_callback(mcp_progress_callback)
        
        # Determine which solver to use
        solver_name = solver or config.get("solvers", {}).get("default", "scip")
        solver_config = config.get("solvers", {})
        
        # Create solver using factory
        solver_instance = SolverFactory.create_solver(solver_name, solver_config)
        
        # Set up progress callback for solver
        solver_instance.set_progress_callback(mcp_progress_callback)
        
        # Apply solver parameters if provided
        if solver_params:
            solver_instance.set_parameters(solver_params)
        
        # Progress update: starting code execution
        await mcp_progress_callback(SolverProgress(
            stage="modeling",
            time_elapsed=0.0,
            message="Executing PuLP code in secure environment"
        ))
        
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
        
        # Progress update: starting solver
        await mcp_progress_callback(SolverProgress(
            stage="presolving",
            time_elapsed=0.0,
            message="Starting optimization solver"
        ))
        
        # Solve the optimization problem
        solver_output_captured = None
        if include_solver_output:
            # Use solver's internal summary generation (safe for MCP)
            logger.info("Capturing solver output using internal summary generation")
            with suppress_stdout_for_mcp():
                solution = await solver_instance.solve_from_file(file_path, capture_output=True)
            
            # Extract solver output from solution's solver_info
            if hasattr(solution, 'solver_info') and solution.solver_info:
                solver_output_captured = solution.solver_info.get('detailed_output', 'No detailed output available')
            else:
                solver_output_captured = "No solver output available"
                
            logger.info(f"Generated solver output summary, length: {len(solver_output_captured)}")
        else:
            # Suppress output for clean MCP protocol
            with suppress_stdout_for_mcp():
                solution = await solver_instance.solve_from_file(file_path, capture_output=False)
        
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
            solver_info=solver_instance.get_solver_info(),
            solver_output=solver_output_captured
        )
        
        # Final progress update
        await mcp_progress_callback(SolverProgress(
            stage="finished",
            time_elapsed=0.0,
            message=f"Optimization completed: {solution.status}"
        ))
        
        logger.info(f"Optimization completed: {solution.status}")
        return response
        
    except Exception as e:
        # Clean up Pyodide if needed
        try:
            if 'executor' in locals() and hasattr(executor, 'cleanup'):
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
        else:
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


async def execute_mip_code_with_progress(
    code: str,
    data: Optional[Dict[str, Any]] = None,
    solver: Optional[str] = None,
    solver_params: Optional[Dict[str, Any]] = None,
    validate_solution: bool = True,
    validation_tolerance: float = 1e-6,
    include_solver_output: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Union[ProgressResponse, ExecutionResponse], None]:
    """Execute PuLP code with progress updates yielded during solving.
    
    Args:
        code: PuLP Python code to execute
        data: Optional data dictionary to pass to the code
        solver: Solver to use (default: from config, fallback: "scip")
        solver_params: Optional solver parameters
        validate_solution: Whether to validate solution against constraints
        validation_tolerance: Numerical tolerance for constraint validation
        include_solver_output: Whether to include detailed solver output in response
        config: Configuration dictionary
        
    Yields:
        ProgressResponse objects during execution, final ExecutionResponse when complete
    """
    config = config or {}
    
    # Progress tracking
    progress_queue: asyncio.Queue = asyncio.Queue()
    
    def progress_callback(progress: SolverProgress):
        """Callback to capture progress updates."""
        try:
            # Put progress in queue for async processing
            asyncio.create_task(progress_queue.put(ProgressResponse(progress=progress)))
        except Exception as e:
            logger.warning(f"Failed to queue progress update: {e}")
    
    try:
        # Always use Pyodide executor for security
        executor = PyodideExecutor(config)
        logger.info("Using Pyodide executor for secure execution")
        
        # Set up progress callback for executor (modeling stage)
        executor.set_progress_callback(progress_callback)
        
        # Determine which solver to use
        solver_name = solver or config.get("solvers", {}).get("default", "scip")
        solver_config = config.get("solvers", {})
        
        # Create solver using factory
        solver_instance = SolverFactory.create_solver(solver_name, solver_config)
        
        # Set up progress callback for solver
        solver_instance.set_progress_callback(progress_callback)
        
        # Apply solver parameters if provided
        if solver_params:
            solver_instance.set_parameters(solver_params)
        
        logger.info("Executing PuLP code in Pyodide sandbox")
        
        # Execute PuLP code and generate optimization file
        stdout, stderr, file_path, detected_library = await executor.execute_mip_code(
            code, data
        )
        
        if not file_path:
            yield ExecutionResponse(
                status="error",
                message="No optimization file was generated",
                stdout=stdout,
                stderr=stderr,
                solution=None
            )
            return
        
        logger.info(f"Generated optimization file: {file_path}")
        
        # Create async task for solving with progress monitoring
        async def solve_with_progress():
            """Solve the problem while monitoring progress."""
            solver_output_captured = None
            if include_solver_output:
                # Use solver's internal summary generation (safe for MCP)
                logger.info("Capturing solver output using internal summary generation")
                with suppress_stdout_for_mcp():
                    solution = await solver_instance.solve_from_file(file_path, capture_output=True)
                
                # Extract solver output from solution's solver_info
                if hasattr(solution, 'solver_info') and solution.solver_info:
                    solver_output_captured = solution.solver_info.get('detailed_output', 'No detailed output available')
                else:
                    solver_output_captured = "No solver output available"
                    
                logger.info(f"Generated solver output summary, length: {len(solver_output_captured)}")
            else:
                # Suppress output for clean MCP protocol
                with suppress_stdout_for_mcp():
                    solution = await solver_instance.solve_from_file(file_path, capture_output=False)
            
            return solution, solver_output_captured
        
        # Start solving task
        solving_task = asyncio.create_task(solve_with_progress())
        
        # Monitor progress while solving
        while not solving_task.done():
            try:
                # Wait for progress updates with timeout
                progress_response = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield progress_response
            except asyncio.TimeoutError:
                # No progress update received, continue monitoring
                continue
            except Exception as e:
                logger.warning(f"Error processing progress update: {e}")
                continue
        
        # Get the final result
        solution, solver_output_captured = await solving_task
        
        # Drain any remaining progress updates
        while not progress_queue.empty():
            try:
                progress_response = progress_queue.get_nowait()
                yield progress_response
            except asyncio.QueueEmpty:
                break
        
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
            solver_info=solver_instance.get_solver_info(),
            solver_output=solver_output_captured
        )
        
        logger.info(f"Optimization completed: {solution.status}")
        yield response
        
    except SecurityError as e:
        logger.error(f"Security error: {e}")
        yield ExecutionResponse(
            status="security_error",
            message=f"Code contains security violations: {e}",
            stdout="",
            stderr="",
            solution=None
        )
    
    except CodeExecutionError as e:
        logger.error(f"Code execution error: {e}")
        yield ExecutionResponse(
            status="execution_error",
            message=f"Code execution failed: {e}",
            stdout="",
            stderr="",
            solution=None
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        yield ExecutionResponse(
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
        config = config or {}
        solver_config = config.get("solvers", {})
        default_solver = solver_config.get("default", "scip")
        
        # Get information for all available solvers
        available_solvers = SolverFactory.get_available_solvers()
        solvers_info = {}
        
        for solver_name in available_solvers:
            try:
                solver_instance = SolverFactory.create_solver(solver_name, solver_config)
                solver_info = solver_instance.get_solver_info()
                solvers_info[solver_name] = SolverInfo(**solver_info)
            except Exception as e:
                # If solver creation fails, still include it with error info
                solvers_info[solver_name] = SolverInfo(
                    name=solver_name,
                    available=False,
                    error=str(e)
                )
        
        return SolverInfoResponse(
            status="success",
            solvers=solvers_info,
            default_solver=default_solver
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
            issues=[],
            is_valid=False
        )


async def get_mip_examples_handler() -> ExamplesResponse:
    """Get example MIP code snippets for PuLP.
    
    Returns:
        Dictionary with example code snippets for PuLP library
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
        
    }
    
    # Convert dict examples to ExampleCode models
    example_models = {}
    for key, ex in examples.items():
        example_models[key] = ExampleCode(
            name=ex["name"],
            description=ex["description"],
            code=ex["code"],
            library="pulp"
        )
    
    return ExamplesResponse(
        status="success",
        examples=example_models,
        total_examples=len(examples),
        message="Examples retrieved successfully"
    )
