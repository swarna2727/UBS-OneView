"""Synchronous MCP client for the OneView tool server.

The agents and the LangGraph workflow are synchronous, so one background thread runs an asyncio loop that holds a
single stdio session open (the server is spawned once per process, not per call). call() blocks until the tool
returns and raises McpToolError when the server reports a tool error."""

import asyncio
import atexit
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

FA_MAIN = Path(__file__).resolve().parent.parent.parent
CALL_TIMEOUT_S = 120.0


class McpToolError(RuntimeError):
    """The MCP server could not run a tool. Agents surface it instead of guessing."""


class McpToolClient:
    def __init__(self, params: Optional[StdioServerParameters] = None, start_timeout_s: float = 60.0):
        # Only a safe subset of the environment reaches the subprocess, so pass the import path explicitly.
        pythonpath = os.pathsep.join(p for p in (str(FA_MAIN), os.environ.get("PYTHONPATH", "")) if p)
        self._params = params or StdioServerParameters(
            command=sys.executable, args=["-m", "src.mcp_tools.server"], cwd=str(FA_MAIN), env={"PYTHONPATH": pythonpath})
        self._session: Optional[ClientSession] = None
        self._stop: Optional[asyncio.Event] = None
        self._runner: Optional[asyncio.Task] = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="oneview-mcp-client", daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._open(), self._loop).result(timeout=start_timeout_s)

    async def _open(self) -> None:
        self._stop = asyncio.Event()
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        self._runner = asyncio.create_task(self._serve(ready))
        await ready

    async def _serve(self, ready: asyncio.Future) -> None:
        # The transport and session are entered and exited in this one task, as the SDK's context managers require.
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    ready.set_result(None)
                    await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise
        finally:
            self._session = None

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self._session is None:
            raise McpToolError(f"{name}: the MCP tool server is not connected.")
        future = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, arguments), self._loop)
        result = future.result(timeout=CALL_TIMEOUT_S)
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise McpToolError(f"{name}: {text or 'tool error'}")
        if result.structured_content is None:
            raise McpToolError(f"{name}: no structured result")
        return result.structured_content

    def close(self) -> None:
        if not self._loop.is_running():
            return
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._runner is not None:
            try:
                asyncio.run_coroutine_threadsafe(asyncio.wait_for(asyncio.shield(self._runner), 10), self._loop).result(timeout=15)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


_client: Optional[McpToolClient] = None
_lock = threading.Lock()


def get_client() -> McpToolClient:
    """One shared session per process, started on first use."""
    global _client
    with _lock:
        if _client is None:
            _client = McpToolClient()
            atexit.register(_client.close)
        return _client
