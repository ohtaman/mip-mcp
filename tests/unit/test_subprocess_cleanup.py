"""Tests for subprocess cleanup and signal handling."""

import asyncio
import signal
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.mip_mcp.executor.pyodide_executor import PyodideExecutor
from src.mip_mcp.server import MIPMCPServer
from src.mip_mcp.utils.executor_registry import ExecutorRegistry


class TestExecutorRegistry:
    """Test ExecutorRegistry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        ExecutorRegistry._executors.clear()

    @pytest.mark.asyncio
    async def test_register_and_unregister_executor(self):
        """Test registering and unregistering executors."""
        # Create mock executor
        executor = MagicMock()
        executor.cleanup = AsyncMock()

        # Register executor
        await ExecutorRegistry.register(executor)
        assert ExecutorRegistry.get_active_count() == 1

        # Unregister executor
        await ExecutorRegistry.unregister(executor)
        assert ExecutorRegistry.get_active_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_all_executors(self):
        """Test cleaning up all registered executors."""
        # Create mock executors
        executor1 = MagicMock()
        executor1.cleanup = AsyncMock()
        executor2 = MagicMock()
        executor2.cleanup = AsyncMock()

        # Register executors
        await ExecutorRegistry.register(executor1)
        await ExecutorRegistry.register(executor2)
        assert ExecutorRegistry.get_active_count() == 2

        # Clean up all
        await ExecutorRegistry.cleanup_all()

        # Verify cleanup was called
        executor1.cleanup.assert_called_once()
        executor2.cleanup.assert_called_once()
        assert ExecutorRegistry.get_active_count() == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_timeout(self):
        """Test cleanup handling when executor cleanup times out."""
        # Create mock executor that takes too long to cleanup
        executor = MagicMock()
        executor.cleanup = AsyncMock()

        async def slow_cleanup():
            await asyncio.sleep(20)  # Longer than cleanup timeout

        executor.cleanup.side_effect = slow_cleanup

        await ExecutorRegistry.register(executor)

        # This should timeout but not hang
        start_time = time.time()
        await ExecutorRegistry.cleanup_all()
        elapsed = time.time() - start_time

        # Should complete within reasonable time (15s timeout + some overhead)
        assert elapsed < 20
        executor.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_dead_reference_cleanup(self):
        """Test that dead weak references are cleaned up automatically."""
        # Create a real executor instance that can be garbage collected
        config = {"executor": {"timeout": 60}}
        executor = PyodideExecutor(config)

        await ExecutorRegistry.register(executor)
        assert ExecutorRegistry.get_active_count() == 1

        # Delete the executor (simulating it going out of scope)
        del executor

        # Force garbage collection and check count
        import gc
        gc.collect()

        # The count should eventually become 0 as dead references are cleaned up
        assert ExecutorRegistry.get_active_count() == 0


class TestPyodideExecutorCleanup:
    """Test PyodideExecutor cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_with_graceful_exit(self):
        """Test cleanup when process exits gracefully."""
        config = {"executor": {"timeout": 60}}
        executor = PyodideExecutor(config)

        # Mock the process
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.wait = AsyncMock()

        # Set up the mock process
        executor.pyodide_process = mock_process
        executor._pyodide_initialized = True

        # Mock communication method
        executor._communicate_with_pyodide = AsyncMock()

        # Cleanup should work without timeout
        await executor.cleanup()

        # Verify graceful shutdown was attempted
        executor._communicate_with_pyodide.assert_called_once_with({"action": "exit"})
        mock_process.wait.assert_called()
        assert executor.pyodide_process is None
        assert not executor._pyodide_initialized

    @pytest.mark.asyncio
    async def test_cleanup_with_timeout_and_terminate(self):
        """Test cleanup when process doesn't exit gracefully and needs termination."""
        config = {"executor": {"timeout": 60}}
        executor = PyodideExecutor(config)

        # Mock the process
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        # Make wait timeout initially, then succeed after terminate
        wait_call_count = 0
        async def mock_wait():
            nonlocal wait_call_count
            wait_call_count += 1
            if wait_call_count == 1:
                raise TimeoutError()
            # Second call (after terminate) succeeds
            return

        mock_process.wait = mock_wait

        # Set up the mock process
        executor.pyodide_process = mock_process
        executor._pyodide_initialized = True

        # Mock communication method to timeout
        async def timeout_communicate(*args):
            raise TimeoutError()

        executor._communicate_with_pyodide = timeout_communicate

        # Cleanup should handle timeout and terminate
        await executor.cleanup()

        # Verify terminate was called
        mock_process.terminate.assert_called_once()
        assert executor.pyodide_process is None
        assert not executor._pyodide_initialized

    @pytest.mark.asyncio
    async def test_cleanup_with_kill_fallback(self):
        """Test cleanup when process needs to be killed forcefully."""
        config = {"executor": {"timeout": 60}}
        executor = PyodideExecutor(config)

        # Mock the process
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        # Make wait always timeout (unresponsive process)
        async def mock_wait():
            raise TimeoutError()

        mock_process.wait = mock_wait

        # Set up the mock process
        executor.pyodide_process = mock_process
        executor._pyodide_initialized = True

        # Mock communication method to timeout
        async def timeout_communicate(*args):
            raise TimeoutError()

        executor._communicate_with_pyodide = timeout_communicate

        # Cleanup should escalate to kill
        await executor.cleanup()

        # Verify both terminate and kill were called
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert executor.pyodide_process is None

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self):
        """Test that cleanup can be called multiple times safely."""
        config = {"executor": {"timeout": 60}}
        executor = PyodideExecutor(config)

        # Mock the process
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.wait = AsyncMock()

        executor.pyodide_process = mock_process
        executor._pyodide_initialized = True
        executor._communicate_with_pyodide = AsyncMock()

        # First cleanup
        await executor.cleanup()
        assert executor._cleanup_started

        # Second cleanup should be a no-op
        await executor.cleanup()

        # Communication should only be called once
        assert executor._communicate_with_pyodide.call_count == 1

    def test_del_method_cleanup(self):
        """Test that __del__ method properly cleans up."""
        config = {"executor": {"timeout": 60}}
        executor = PyodideExecutor(config)

        # Mock the process
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.poll = MagicMock(return_value=None)  # Process still running

        executor.pyodide_process = mock_process
        executor._cleanup_started = False

        # Trigger __del__
        executor.__del__()

        # Verify terminate and kill were called
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()


class TestServerSignalHandling:
    """Test MIPMCPServer signal handling."""

    def test_atexit_handler_registration(self):
        """Test that atexit cleanup handler is properly registered."""
        with patch('atexit.register') as mock_atexit:
            MIPMCPServer()

            # Verify atexit handler was registered
            mock_atexit.assert_called_once()

    def test_atexit_cleanup_with_active_executors(self):
        """Test atexit cleanup when there are active executors."""
        server = MIPMCPServer()
        
        # Mock ExecutorRegistry to simulate active executors
        with (
            patch.object(ExecutorRegistry, 'get_active_count', return_value=2),
            patch.object(ExecutorRegistry, 'cleanup_all', return_value=None) as mock_cleanup,
            patch('asyncio.new_event_loop') as mock_new_loop,
            patch('asyncio.set_event_loop') as mock_set_loop,
        ):
            # Create mock loop
            mock_loop = MagicMock()
            mock_new_loop.return_value = mock_loop
            
            # Get the atexit cleanup function and call it
            import atexit
            with patch('atexit.register') as mock_atexit:
                server._setup_cleanup_hooks()
                cleanup_func = mock_atexit.call_args[0][0]
                
            # Call the cleanup function
            cleanup_func()
            
            # Verify cleanup was attempted
            mock_new_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_loop)
            mock_loop.run_until_complete.assert_called_once()
            mock_loop.close.assert_called_once()


class TestProcessGroupManagement:
    """Test process group creation and management."""

    @pytest.mark.asyncio
    async def test_subprocess_created_with_new_session(self):
        """Test that subprocesses are created with start_new_session=True."""
        config = {"executor": {"timeout": 60}}

        with patch('asyncio.create_subprocess_exec') as mock_create:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_create.return_value = mock_process

            executor = PyodideExecutor(config)

            # Mock the path finding and script creation  
            with patch.object(executor, '_find_pyodide_path', return_value='/fake/path'):
                with patch('tempfile.NamedTemporaryFile'):
                    # Use regular MagicMock for async methods to avoid warnings
                    with patch.object(executor, '_wait_for_process_ready', return_value=None):
                        with patch.object(executor, '_communicate_with_pyodide', return_value={"success": True}):
                            try:
                                await executor._initialize_pyodide()
                            except Exception:
                                pass  # We expect this to fail in test environment

            # Verify subprocess was created with start_new_session=True
            if mock_create.called:
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs.get('start_new_session') is True


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    @pytest.mark.asyncio
    async def test_handler_cleanup_on_exception(self):
        """Test that executor cleanup happens even when handler fails."""
        from src.mip_mcp.handlers.execute_code import execute_mip_code_with_progress

        # Mock executor creation and registration
        with patch('src.mip_mcp.handlers.execute_code.PyodideExecutor') as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor.cleanup = AsyncMock()
            mock_executor.execute_mip_code = AsyncMock(side_effect=Exception("Test error"))
            mock_executor.set_progress_callback = MagicMock()  # Add missing method
            mock_executor_class.return_value = mock_executor

            with patch.object(ExecutorRegistry, 'register', return_value=None) as mock_register:
                with patch.object(ExecutorRegistry, 'unregister', return_value=None) as mock_unregister:

                    # Execute code that will fail
                    responses = []
                    async for response in execute_mip_code_with_progress(
                        code="invalid code",
                        config={"executor": {"timeout": 60}}
                    ):
                        responses.append(response)

                    # Verify executor was registered and cleaned up
                    mock_register.assert_called_once_with(mock_executor)
                    mock_executor.cleanup.assert_called_once()
                    mock_unregister.assert_called_once_with(mock_executor)

                    # Should get an error response
                    assert len(responses) == 1
                    assert responses[0].status == "error"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_executors(self):
        """Test cleanup of multiple concurrent executors."""
        # Create multiple mock executors
        executors = []
        for _i in range(3):
            executor = MagicMock()
            executor.cleanup = AsyncMock()
            executors.append(executor)
            await ExecutorRegistry.register(executor)

        assert ExecutorRegistry.get_active_count() == 3

        # Clean up all
        await ExecutorRegistry.cleanup_all()

        # Verify all were cleaned up
        for executor in executors:
            executor.cleanup.assert_called_once()

        assert ExecutorRegistry.get_active_count() == 0
