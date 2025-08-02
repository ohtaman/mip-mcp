# MIP MCP Server

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![MCP](https://img.shields.io/badge/MCP-Compatible-orange.svg)](https://github.com/modelcontextprotocol)

A secure Mixed Integer Programming (MIP) optimization server using the Model Context Protocol (MCP) with PuLP and Pyodide WebAssembly security.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Usage](#usage)
  - [MCP Tools](#mcp-tools)
  - [Configuration](#configuration)
- [Examples](#examples)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## About

MIP MCP Server enables Large Language Models (LLMs) to execute PuLP optimization code and solve Mixed Integer Programming problems through a secure, standardized interface. The server uses Pyodide WebAssembly for complete process isolation while providing access to powerful optimization capabilities via SCIP solver.

### Key Benefits

- **Security First**: Complete WebAssembly sandbox isolation
- **LLM Integration**: Native Model Context Protocol support
- **High Performance**: SCIP solver with customizable parameters
- **Zero Configuration**: Automatic problem detection and solving
- **Production Ready**: Comprehensive testing and validation

## Features

- 🛡️ **Secure Execution**: Pyodide WebAssembly sandbox with complete process isolation
- 📊 **PuLP Support**: Full compatibility with PuLP optimization library
- 🔍 **Auto Detection**: Automatic problem detection from PuLP objects
- ⚡ **SCIP Integration**: High-performance optimization solving with pyscipopt
- 🌐 **MCP Protocol**: Standards-based LLM integration via Model Context Protocol
- 🎛️ **Flexible Parameters**: Customizable solver settings and validation options
- 📁 **Format Support**: Automatic LP/MPS format detection and generation
- 📈 **Progress Reporting**: Real-time optimization progress with 10-second intervals
- ✅ **Solution Validation**: Built-in constraint validation with configurable tolerance

## Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Node.js (for Pyodide WebAssembly execution)

### Installation

#### Using uv (Recommended)

```bash
git clone https://github.com/yourusername/mip-mcp.git
cd mip-mcp
uv sync
```

#### Using pip

```bash
git clone https://github.com/yourusername/mip-mcp.git
cd mip-mcp
pip install -e .
```

### Quick Start

Start the MCP server:

```bash
# Using uv
uv run mip-mcp

# Using pip installation
mip-mcp
```

The server will start and listen for MCP connections, providing optimization tools to connected LLM clients.

## Usage

### MCP Tools

#### execute_mip_code

Execute PuLP optimization code in a secure WebAssembly environment.

```python
# Example PuLP code
import pulp

prob = pulp.LpProblem("Simple_LP", pulp.LpMaximize)
x = pulp.LpVariable("x", lowBound=0)
y = pulp.LpVariable("y", lowBound=0)

prob += 3*x + 2*y  # Objective
prob += 2*x + y <= 100  # Constraint
prob += x + y <= 80     # Constraint
```

**Parameters:**
- `code` (str): PuLP Python code to execute
- `data` (dict, optional): Input data for the optimization problem
- `solver_params` (dict, optional): SCIP solver parameters
- `validate_solution` (bool, default=True): Enable solution validation
- `validation_tolerance` (float, default=1e-6): Numerical tolerance
- `include_solver_output` (bool, default=False): Include detailed solver statistics

#### get_solver_info

Retrieve information about the available SCIP solver including version and capabilities.

#### validate_mip_code

Validate PuLP code for syntax errors, security issues, and Pyodide compatibility.

#### get_mip_examples

Get example PuLP code snippets demonstrating various optimization problem types.

### Configuration

Create `config/default.yaml` to customize server behavior:

```yaml
server:
  name: "mip-mcp"
  version: "0.1.0"
  timeout: 60

solver:
  default: "scip"
  timeout: 300
  parameters:
    threads: 4
    gap: 0.01
    time_limit: 600

security:
  enable_validation: true
  execution_timeout: 60
  allowed_imports:
    - pulp
    - math
    - numpy

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Examples

### Linear Programming

```python
import pulp

# Create problem
prob = pulp.LpProblem("Diet_Problem", pulp.LpMinimize)

# Variables
x1 = pulp.LpVariable("Bread", lowBound=0)
x2 = pulp.LpVariable("Milk", lowBound=0)

# Objective function
prob += 0.15*x1 + 0.25*x2

# Constraints
prob += 4*x1 + 3*x2 >= 10  # Protein
prob += 2*x1 + 2*x2 >= 8   # Carbs
```

### Integer Programming

```python
import pulp

# Knapsack problem
prob = pulp.LpProblem("Knapsack", pulp.LpMaximize)

# Binary variables for items
items = ['item1', 'item2', 'item3']
x = pulp.LpVariable.dicts("x", items, cat='Binary')

# Objective: maximize value
values = {'item1': 10, 'item2': 20, 'item3': 15}
prob += pulp.lpSum([values[i] * x[i] for i in items])

# Constraint: weight limit
weights = {'item1': 5, 'item2': 8, 'item3': 3}
prob += pulp.lpSum([weights[i] * x[i] for i in items]) <= 10
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/mip_mcp --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_handlers.py -v
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Type checking
uv run mypy src/

# Run linting (if configured)
uv run ruff check src/ tests/
```

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Install pre-commit hooks (if using)
pre-commit install
```

## Architecture

```
src/mip_mcp/
├── __init__.py              # Package initialization
├── mcp_server.py           # Main MCP server entry point
├── server.py               # FastMCP server implementation
├── exceptions.py           # Custom exception classes
├── handlers/               # MCP request handlers
│   ├── __init__.py
│   └── execute_code.py     # Code execution handler
├── executor/               # Code execution engines
│   ├── __init__.py
│   └── pyodide_executor.py # Pyodide WebAssembly executor
├── solvers/                # Optimization solver interfaces
│   ├── __init__.py
│   ├── base.py            # Base solver interface
│   └── scip_solver.py     # SCIP solver implementation
├── models/                 # Data models and schemas
│   ├── __init__.py
│   ├── config.py          # Configuration models
│   ├── responses.py       # API response models
│   └── solution.py        # Solution data models
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── config_manager.py  # Configuration management
│   ├── library_detector.py # Library detection utilities
│   ├── logger.py          # Logging configuration
│   └── solution_validator.py # Solution validation
└── config/                 # Configuration files
    └── default.yaml       # Default configuration
```

### Key Components

- **MCP Server**: FastMCP-based server implementing Model Context Protocol
- **Pyodide Executor**: WebAssembly-based secure code execution engine
- **SCIP Solver**: High-performance optimization solver integration
- **Handlers**: Request processing and response generation
- **Models**: Pydantic-based data validation and serialization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive tests
- Update documentation as needed
- Ensure security best practices

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- **ohtaman** - *Initial work* - [ohtaman](https://github.com/ohtaman)

## Acknowledgments

- [PuLP](https://github.com/coin-or/pulp) - Python Linear Programming library
- [SCIP](https://scipopt.org/) - Solving Constraint Integer Programs
- [Pyodide](https://pyodide.org/) - Python scientific stack in WebAssembly
- [FastMCP](https://github.com/jlowin/fastmcp) - Fast Model Context Protocol implementation
- [Model Context Protocol](https://modelcontextprotocol.io/) - Standard for LLM-tool integration
