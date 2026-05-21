"""
Sandboxed Python code execution tool.

Security posture
────────────────
This sandbox runs untrusted code in a child subprocess with:
  - Empty environment  (no parent credentials, API keys, or paths inherited)
  - Restricted cwd     (isolated temp directory, cleaned up after execution)
  - CPU time limit     (RLIMIT_CPU via preexec_fn on Unix)
  - Memory limit       (RLIMIT_AS on Unix)
  - Process limit      (RLIMIT_NPROC on Unix — prevents fork bombs)
  - Hard wall-clock timeout (asyncio.wait_for)

What this sandbox CANNOT prevent:
  - Network access — use a proper network namespace (Docker/Firecracker) for that
  - Reading files the process has permission to read under its UID
  - Kernel exploits

For production workloads with untrusted user code, replace this with a
proper container-based sandbox (e2b.dev, Modal, Morph, AWS Firecracker).
This implementation is suitable for LLM-generated code in trusted deployments.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import textwrap

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource

# Resource limits applied to the sandbox subprocess (Unix only)
_CPU_SECONDS = 10          # hard CPU time limit
_MEMORY_BYTES = 256 * 1024 * 1024   # 256 MB virtual address space
_MAX_PROCESSES = 32        # prevent fork bombs
_MAX_OUTPUT_BYTES = 100 * 1024      # 100 KB — prevents OOM from runaway print loops


def _make_preexec(cpu_seconds: int, memory_bytes: int, max_procs: int):
    """Return a preexec_fn that sets resource limits on Unix."""
    def _preexec():
        try:
            import resource
            # CPU time
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            # Virtual memory
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            # Number of child processes
            resource.setrlimit(resource.RLIMIT_NPROC, (max_procs, max_procs))
        except (ImportError, AttributeError, ValueError):
            pass  # Windows or unprivileged container — limits silently skipped
    return _preexec


async def _execute_python(
    code: str,
    timeout: int = 10,
    cpu_seconds: int = _CPU_SECONDS,
    memory_bytes: int = _MEMORY_BYTES,
) -> str:
    """
    Execute `code` in an isolated subprocess.

    Returns captured stdout+stderr as a string, or an error/timeout message.
    """
    # Wrap code to capture both stdout and stderr in a single buffer
    wrapped = textwrap.dedent(f"""\
        import sys as _sys, io as _io, traceback as _tb
        _buf = _io.StringIO()
        _sys.stdout = _buf
        _sys.stderr = _buf
        try:
{textwrap.indent(code, '            ')}
        except BaseException:
            _tb.print_exc()
        finally:
            _sys.stdout = _sys.__stdout__
            print(_buf.getvalue(), end="")
    """)

    with tempfile.TemporaryDirectory(prefix="kazi_sandbox_") as sandbox_dir:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", wrapped,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox_dir,
                # Empty environment — no credentials or path info from parent
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": "",
                    "HOME": sandbox_dir,
                },
                close_fds=True,
                preexec_fn=_make_preexec(cpu_seconds, memory_bytes, _MAX_PROCESSES),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            raw = stdout + stderr
            truncated = len(raw) > _MAX_OUTPUT_BYTES
            output = raw[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace").strip()
            if truncated:
                output += f"\n[... output truncated at {_MAX_OUTPUT_BYTES // 1024}KB]"
            return output or "(no output)"

        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            return (
                f"Execution timed out after {timeout}s. "
                "Ensure your code completes within the time limit."
            )
        except Exception as exc:
            return f"Sandbox error: {exc}"


def python_sandbox_tool(
    timeout: int = 10,
    cpu_seconds: int = _CPU_SECONDS,
    memory_bytes: int = _MEMORY_BYTES,
) -> ToolDefinition:
    """
    Return a ToolDefinition that executes Python code in a hardened subprocess sandbox.

    Parameters
    ──────────
    timeout        Wall-clock seconds before the subprocess is killed.
    cpu_seconds    CPU-time hard limit (Unix only).
    memory_bytes   Virtual address space cap in bytes (Unix only). Default: 256 MB.

    Security note: see module docstring for what this sandbox cannot prevent.
    For untrusted user code in multi-tenant production, use a container sandbox.
    """

    async def handler(code: str) -> str:
        return await _execute_python(code, timeout=timeout, cpu_seconds=cpu_seconds, memory_bytes=memory_bytes)

    return ToolDefinition(
        name="execute_python",
        description=(
            f"Execute Python code and return the output. "
            f"Use print() to produce output. "
            f"Limited to {timeout}s wall-clock time and {memory_bytes // (1024*1024)}MB memory."
        ),
        parameters=[
            ToolParameter(
                name="code",
                type="string",
                description="Python code to execute. Must use print() for output.",
                required=True,
            ),
        ],
        source=ToolSource.NATIVE,
        handler=handler,
        metadata={
            "sandbox_type": "subprocess",
            "timeout": timeout,
            "cpu_seconds": cpu_seconds,
            "memory_bytes": memory_bytes,
        },
    )
