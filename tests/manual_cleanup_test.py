#!/usr/bin/env python3
"""Manual test script for subprocess cleanup verification.

This script can be used to manually test that subprocess cleanup works
correctly by monitoring the process tree before and after running the MCP server.

Usage:
    python tests/manual_cleanup_test.py

Then in another terminal:
    ps aux | grep node  # Check for Node.js processes before/after
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add the project root to the path for standalone execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from src.mip_mcp.executor.pyodide_executor import PyodideExecutor
from src.mip_mcp.utils.executor_registry import ExecutorRegistry


def get_nodejs_processes():
    """Get list of Node.js processes."""
    try:
        result = subprocess.run(
            ["ps", "aux", "-o", "pid,ppid,command"],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        nodejs_procs = [line for line in lines if 'node' in line.lower() and 'pyodide' in line.lower()]
        return nodejs_procs
    except Exception as e:
        print(f"Error getting process list: {e}")
        return []


@pytest.mark.asyncio
async def test_executor_cleanup():
    """Test PyodideExecutor cleanup functionality."""
    print("=== Testing PyodideExecutor Cleanup ===")

    config = {"executor": {"timeout": 60}}

    print("\n1. Initial Node.js processes:")
    initial_procs = get_nodejs_processes()
    for proc in initial_procs:
        print(f"   {proc}")
    print(f"   Total: {len(initial_procs)}")

    print("\n2. Creating PyodideExecutor...")
    executor = PyodideExecutor(config)
    await ExecutorRegistry.register(executor)

    print("\n3. Executing simple code to initialize Pyodide process...")
    try:
        result = await executor.execute_mip_code("""
import pulp

# Simple optimization problem
prob = pulp.LpProblem("Test", pulp.LpMaximize)
x = pulp.LpVariable("x", 0, 10)
y = pulp.LpVariable("y", 0, 10)

prob += x + y  # Objective
prob += x + 2*y <= 20  # Constraint
prob += 3*x + y <= 30  # Constraint

print("Problem created successfully")
        """)
        print(f"   Execution result: {result[0][:100]}...")
    except Exception as e:
        print(f"   Execution failed (expected in test environment): {e}")

    print("\n4. Node.js processes after executor creation:")
    after_create_procs = get_nodejs_processes()
    for proc in after_create_procs:
        print(f"   {proc}")
    print(f"   Total: {len(after_create_procs)}")

    new_procs = len(after_create_procs) - len(initial_procs)
    print(f"   New processes: {new_procs}")

    print("\n5. Cleaning up executor...")
    await executor.cleanup()
    await ExecutorRegistry.unregister(executor)

    # Give processes time to terminate
    await asyncio.sleep(2)

    print("\n6. Node.js processes after cleanup:")
    after_cleanup_procs = get_nodejs_processes()
    for proc in after_cleanup_procs:
        print(f"   {proc}")
    print(f"   Total: {len(after_cleanup_procs)}")

    remaining_procs = len(after_cleanup_procs) - len(initial_procs)
    print(f"   Remaining new processes: {remaining_procs}")

    if remaining_procs == 0:
        print("   ✅ PASS: All processes cleaned up successfully!")
    else:
        print("   ❌ FAIL: Some processes remain after cleanup")

    return remaining_procs == 0


@pytest.mark.asyncio
async def test_registry_cleanup():
    """Test ExecutorRegistry cleanup functionality."""
    print("\n\n=== Testing ExecutorRegistry Cleanup ===")

    config = {"executor": {"timeout": 60}}

    print("\n1. Creating multiple executors...")
    executors = []
    for i in range(3):
        executor = PyodideExecutor(config)
        await ExecutorRegistry.register(executor)
        executors.append(executor)
        print(f"   Created executor {i+1}")

    print(f"\n2. Registry count: {ExecutorRegistry.get_active_count()}")

    print("\n3. Cleaning up all executors via registry...")
    start_time = time.time()
    await ExecutorRegistry.cleanup_all()
    cleanup_time = time.time() - start_time

    print(f"   Cleanup completed in {cleanup_time:.2f} seconds")
    print(f"   Registry count after cleanup: {ExecutorRegistry.get_active_count()}")

    if ExecutorRegistry.get_active_count() == 0:
        print("   ✅ PASS: Registry cleanup successful!")
        return True
    else:
        print("   ❌ FAIL: Registry still has active executors")
        return False


@pytest.mark.asyncio
async def test_signal_handling():
    """Test signal handling (simulated)."""
    print("\n\n=== Testing Signal Handling (Simulated) ===")

    print("\n1. Creating executor and registering with registry...")
    config = {"executor": {"timeout": 60}}
    executor = PyodideExecutor(config)
    await ExecutorRegistry.register(executor)

    print(f"2. Registry count: {ExecutorRegistry.get_active_count()}")

    print("\n3. Simulating signal-triggered cleanup...")
    # This simulates what happens when a signal is received
    await ExecutorRegistry.cleanup_all()

    print(f"4. Registry count after signal cleanup: {ExecutorRegistry.get_active_count()}")

    if ExecutorRegistry.get_active_count() == 0:
        print("   ✅ PASS: Signal cleanup simulation successful!")
        return True
    else:
        print("   ❌ FAIL: Signal cleanup simulation failed")
        return False


@pytest.mark.asyncio
async def test_comprehensive_cleanup():
    """Run all cleanup tests together for comprehensive verification."""
    print("MIP-MCP Subprocess Cleanup Manual Test")
    print("=" * 50)

    print(f"Process ID: {os.getpid()}")
    print(f"Process Group ID: {os.getpgrp()}")

    # Run tests
    results = []

    try:
        results.append(await test_executor_cleanup())
    except Exception as e:
        print(f"Executor cleanup test failed: {e}")
        results.append(False)

    try:
        results.append(await test_registry_cleanup())
    except Exception as e:
        print(f"Registry cleanup test failed: {e}")
        results.append(False)

    try:
        results.append(await test_signal_handling())
    except Exception as e:
        print(f"Signal handling test failed: {e}")
        results.append(False)

    # Summary
    print("\n\n=== Test Summary ===")
    test_names = ["Executor Cleanup", "Registry Cleanup", "Signal Handling"]
    passed = sum(results)
    total = len(results)

    for i, (name, result) in enumerate(zip(test_names, results, strict=False)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Subprocess cleanup is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    # Assert for pytest
    assert passed == total, f"Only {passed}/{total} tests passed"


async def main():
    """Main function for standalone execution."""
    # Create a dummy test result that mimics pytest behavior
    class TestResult:
        def __init__(self):
            self.passed = True
            self.error = None

    try:
        await test_comprehensive_cleanup()
        return True
    except AssertionError as e:
        print(f"\nTest assertion failed: {e}")
        return False
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        exit(1)
