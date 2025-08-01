# MIP MCP Server

PuLP-based Mixed Integer Programming optimization server using Model Context Protocol.

## Overview

MIP MCP Server allows LLM clients to execute PuLP optimization code and solve problems using various MIP solvers. The server receives PuLP Python code, generates MPS/LP files, solves them using SCIP (via pyscipopt), and returns optimization results.

## Workflow

```
LLM Client → PuLP Code → MCP Server → MPS/LP File → SCIP Solver → Results → LLM Client
```

## Features

- **Secure PuLP Code Execution**: AST-based security validation with file writing prohibition
- **Automatic Problem Detection**: No file writing required - problems detected from PuLP objects
- **Variable-Based Content Setting**: Manual content via `__mps_content__`/`__lp_content__` variables
- **SCIP Integration**: High-performance optimization solving
- **MCP Protocol**: Standards-based LLM integration
- **Flexible Solver Parameters**: Customizable optimization settings
- **Multiple Output Formats**: Support for MPS and LP file formats

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mip-mcp

# Install with uv
uv install

# Or install with pip
pip install -e .
```

## Usage

### Starting the Server

```bash
# Using uv
uv run mip-mcp

# Or directly
python -m mip_mcp
```

### MCP Tools

The server provides the following MCP tools:

#### execute_pulp_code
Execute PuLP optimization code and solve problems in a secure sandbox.

**Security Features:**
- File writing operations (writeLP, writeMPS) are prohibited
- Automatic problem detection from PuLP objects (recommended)
- Manual content setting via `__mps_content__` or `__lp_content__` variables
- AST-based security validation

**Parameters:**
- `code` (str): PuLP Python code to execute (no file writing allowed)
- `data` (dict, optional): Input data dictionary
- `output_format` (str): Output format ('mps' or 'lp')
- `solver_params` (dict, optional): Solver parameters

#### get_solver_info
Get information about available solvers.

#### validate_pulp_code
Validate PuLP code for security and syntax violations.

**Validation Features:**
- AST analysis for dangerous operations
- File writing operation detection
- Import restriction checking
- Syntax error detection

**Parameters:**
- `code` (str): PuLP Python code to validate

#### get_pulp_examples
Get example PuLP code snippets demonstrating secure usage patterns.

**Example Types:**
- Linear programming (automatic detection)
- Integer programming (automatic detection)
- Knapsack problems (automatic detection)
- Manual content setting (advanced usage)

#### health_check
Check server health and solver availability.

## Configuration

Configuration can be provided via:
- YAML file: `config/default.yaml`
- Environment variables
- Command line arguments

### Example Configuration

```yaml
server:
  name: "mip-mcp"
  version: "0.1.0"
  host: "localhost"
  port: 8000

solver:
  default: "scip"
  timeout: 300
  parameters:
    threads: 4
    gap: 0.01

security:
  enable_validation: true
  allowed_imports:
    - pulp
    - math
    - numpy
```

## Development

```bash
# Install development dependencies
uv install --group dev

# Run tests
pytest

# Run linting
black src/
isort src/
mypy src/

# Type checking
mypy src/
```

## Architecture

```
src/
├── mip_mcp/
│   ├── __init__.py          # Entry point
│   ├── server.py            # FastMCP server implementation
│   ├── handlers/            # MCP request handlers
│   ├── executor/            # Python code execution engine
│   ├── solvers/             # Optimization solver interfaces
│   ├── models/              # Data models
│   ├── utils/               # Utilities
│   └── config/              # Configuration files
```

## License

MIT License