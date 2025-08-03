"""Tests for tuple key ConversionError fix."""

from unittest.mock import AsyncMock, patch

import pytest
from src.mip_mcp.executor.pyodide_executor import PyodideExecutor
from src.mip_mcp.utils.library_detector import MIPLibrary


class TestTupleKeyFix:
    """Test suite for tuple key ConversionError fix."""

    @pytest.fixture
    def executor(self):
        """Create PyodideExecutor instance for testing."""
        config = {
            "execution_timeout": 30.0,
            "progress_interval": 10.0,
        }
        return PyodideExecutor(config)

    def test_detect_tuple_key_patterns(self, executor):
        """Test tuple key pattern detection."""
        code_with_tuple_keys = """
for i in range(3):
    for j in ['a', 'b']:
        var_map[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')

constraint = pulp.lpSum([var_map[(i, j)] for i in range(3) for j in ['a', 'b']]) >= 1
"""

        warnings = executor._detect_tuple_key_patterns(code_with_tuple_keys)

        assert len(warnings) >= 1
        assert any("tuple key assignment" in warning.lower() for warning in warnings)
        assert any("string keys" in warning.lower() for warning in warnings)

    def test_detect_tuple_key_patterns_clean_code(self, executor):
        """Test tuple key detection with clean code (no tuple keys)."""
        clean_code = """
import pulp

prob = pulp.LpProblem("test", pulp.LpMinimize)
x = pulp.LpVariable("x", cat='Binary')
y = pulp.LpVariable("y", cat='Binary')

prob += x + y <= 1
prob += x + y
"""

        warnings = executor._detect_tuple_key_patterns(clean_code)

        assert len(warnings) == 0

    @pytest.mark.asyncio
    async def test_tuple_key_execution_success(self, executor):
        """Test that tuple keys no longer cause ConversionError."""
        tuple_key_code = """
import pulp

# Create problem with tuple keys
prob = pulp.LpProblem("test", pulp.LpMinimize)
var_map = {}

# This used to cause ConversionError
for i in range(2):
    for j in ['a', 'b']:
        var_map[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')

# Add constraint using tuple keys
prob += pulp.lpSum([var_map[(i, j)] for i in range(2) for j in ['a', 'b']]) >= 1

# Set objective
prob += pulp.lpSum([var_map[(i, j)] for i in range(2) for j in ['a', 'b']])

print("Tuple key test completed successfully")
"""

        with (
            patch.object(executor, "_initialize_pyodide", new_callable=AsyncMock),
            patch.object(
                executor, "_execute_with_periodic_progress", new_callable=AsyncMock
            ) as mock_execute,
        ):
            # Mock successful JSON-based execution
            mock_execute.return_value = {
                "success": True,
                "json_data": {
                    "execution_status": "success",
                    "stdout": "Tuple key test completed successfully",
                    "lp_content": "\\* test *\\\nMinimize\nOBJ: x_0_a + x_0_b + x_1_a + x_1_b\nSubject To\n_C1: x_0_a + x_0_b + x_1_a + x_1_b >= 1\nBinaries\nx_0_a\nx_0_b\nx_1_a\nx_1_b\nEnd",
                    "mps_content": None,
                    "problems_info": [{"name": "prob", "num_variables": 4}],
                    "variables_info": {
                        "var_map": {
                            "0_a": "x_0_a",
                            "0_b": "x_0_b",
                            "1_a": "x_1_a",
                            "1_b": "x_1_b",
                        }
                    },
                },
            }

            stdout, stderr, file_path, library = await executor.execute_mip_code(
                tuple_key_code
            )

            # Verify successful execution
            assert stderr == ""
            assert "Tuple key test completed successfully" in stdout
            assert file_path is not None
            assert library == MIPLibrary.PULP

            # Verify no ConversionError
            assert "ConversionError" not in stderr
            assert "Cannot use" not in stderr

    @pytest.mark.asyncio
    async def test_json_safe_conversion_edge_cases(self, executor):
        """Test JSON-safe conversion with various edge cases."""
        edge_case_code = """
import pulp

# Test various tuple key types
test_dict = {}

# Empty tuple
test_dict[()] = "empty"

# Mixed types
test_dict[(1, 'a', 2.5)] = "mixed"

# Nested structures
test_dict[(1, 2)] = {"nested": "value"}

# With PuLP variable
prob = pulp.LpProblem("test", pulp.LpMinimize)
test_dict[('var', 'key')] = pulp.LpVariable("test_var", cat='Binary')

print("Edge case test completed")
"""

        with (
            patch.object(executor, "_initialize_pyodide", new_callable=AsyncMock),
            patch.object(
                executor, "_execute_with_periodic_progress", new_callable=AsyncMock
            ) as mock_execute,
        ):
            # Mock successful execution with converted data
            mock_execute.return_value = {
                "success": True,
                "json_data": {
                    "execution_status": "success",
                    "stdout": "Edge case test completed",
                    "lp_content": None,
                    "mps_content": None,
                    "problems_info": [],
                    "variables_info": {
                        "test_dict": {
                            "": "empty",  # Empty tuple becomes empty string
                            "1_a_2.5": "mixed",  # Mixed types converted
                            "1_2": {"nested": "value"},  # Nested preserved
                            "var_key": {
                                "__type__": "LpVariable",
                                "__str__": "test_var",
                            },  # PuLP var converted
                        }
                    },
                },
            }

            stdout, stderr, file_path, library = await executor.execute_mip_code(
                edge_case_code
            )

            # Verify successful handling of edge cases (no LP content expected, so stderr will have message)
            assert "Edge case test completed" in stdout
            assert library == MIPLibrary.PULP
            # The important thing is no ConversionError
            assert "ConversionError" not in stderr
            assert "Cannot use" not in stderr

    def test_json_safe_conversion_function(self, executor):
        """Test the JSON-safe conversion logic in isolation."""
        # Test the conversion function embedded in the wrapper code
        wrapper = executor._prepare_execution_code("pass", None)

        # Check that the wrapper includes JSON conversion logic
        assert "__convert_to_json_safe" in wrapper
        assert "__json_result__" in wrapper
        assert "json.dumps" in wrapper

        # Check that tuple key conversion is included
        assert "tuple" in wrapper
        assert '".join' in wrapper

    @pytest.mark.asyncio
    async def test_execution_error_handling_with_json(self, executor):
        """Test error handling with JSON-based approach."""
        error_code = """
import pulp
# This will cause a Python error
undefined_variable + 1
"""

        with (
            patch.object(executor, "_initialize_pyodide", new_callable=AsyncMock),
            patch.object(
                executor, "_execute_with_periodic_progress", new_callable=AsyncMock
            ) as mock_execute,
        ):
            # Mock execution with error but JSON response
            mock_execute.return_value = {
                "success": True,
                "json_data": {
                    "execution_status": "error",
                    "stdout": "",
                    "error_message": "name 'undefined_variable' is not defined",
                    "traceback": "NameError: name 'undefined_variable' is not defined",
                    "lp_content": None,
                    "mps_content": None,
                    "problems_info": [],
                    "variables_info": {},
                },
            }

            stdout, stderr, file_path, library = await executor.execute_mip_code(
                error_code
            )

            # Verify error is properly handled through JSON
            assert "Execution error:" in stderr
            assert "undefined_variable" in stderr
            assert file_path is None
            assert library == MIPLibrary.PULP
