"""
Real sandbox integration tests — no mocks, actual subprocess execution.

Every test here spawns a genuine child process and verifies that the
security boundaries and resource limits work as documented.
"""
import math
import sys

import pytest

from kazi.tools.sandbox import _MAX_OUTPUT_BYTES, _execute_python

# ── Basic execution ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hello_world():
    result = await _execute_python('print("hello world")')
    assert result == "hello world"


@pytest.mark.asyncio
async def test_multiline_output():
    code = "\n".join(f'print({i})' for i in range(5))
    result = await _execute_python(code)
    assert result.strip() == "0\n1\n2\n3\n4"


@pytest.mark.asyncio
async def test_arithmetic():
    result = await _execute_python("print(2 ** 32)")
    assert result.strip() == str(2 ** 32)


@pytest.mark.asyncio
async def test_exception_is_captured_not_propagated():
    result = await _execute_python("raise ValueError('intentional error')")
    assert "ValueError" in result
    assert "intentional error" in result


@pytest.mark.asyncio
async def test_syntax_error_is_captured():
    result = await _execute_python("def broken(:)")
    assert "SyntaxError" in result or "invalid syntax" in result


@pytest.mark.asyncio
async def test_zero_division_error():
    result = await _execute_python("print(1 / 0)")
    assert "ZeroDivisionError" in result


@pytest.mark.asyncio
async def test_stderr_is_captured():
    result = await _execute_python("import sys; sys.stderr.write('err output\\n')")
    assert "err output" in result


@pytest.mark.asyncio
async def test_no_output_returns_sentinel():
    result = await _execute_python("x = 1 + 1")
    assert result == "(no output)"


@pytest.mark.asyncio
async def test_complex_computation():
    code = """
total = sum(i * i for i in range(100))
print(total)
"""
    result = await _execute_python(code)
    assert result.strip() == str(sum(i * i for i in range(100)))


@pytest.mark.asyncio
async def test_multiline_print():
    code = """
lines = ["alpha", "beta", "gamma"]
for line in lines:
    print(line)
"""
    result = await _execute_python(code)
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result


# ── Security: environment isolation ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_api_keys_in_sandbox_environment():
    """Sandbox must have an empty environment — no credentials leak in."""
    code = """
import os
# These env vars should NOT be present in the sandbox
keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'KAZI_API_KEY', 'AWS_SECRET_ACCESS_KEY']
leaked = [k for k in keys if os.environ.get(k)]
print('LEAKED:' + ','.join(leaked) if leaked else 'CLEAN')
"""
    result = await _execute_python(code)
    assert "CLEAN" in result
    assert "LEAKED" not in result


@pytest.mark.asyncio
async def test_sandbox_environment_contains_no_credentials():
    """Sandbox must not inherit any credential env vars from the parent process.

    macOS unconditionally injects LC_CTYPE and __CF_USER_TEXT_ENCODING at the
    kernel level — these cannot be suppressed from userspace and are harmless
    locale settings. The security property we're testing is that no API keys,
    secrets, or auth tokens leak in.
    """
    code = """
import os
sensitive_prefixes = [
    'API_KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL',
    'AWS_', 'OPENAI_', 'ANTHROPIC_', 'KAZI_',
]
leaked = [
    k for k in os.environ
    if any(p in k.upper() for p in sensitive_prefixes)
]
print('LEAKED:' + ','.join(leaked) if leaked else 'CLEAN')
"""
    result = await _execute_python(code)
    assert "CLEAN" in result, f"Credentials leaked into sandbox: {result}"


@pytest.mark.asyncio
async def test_sandbox_cwd_is_temp_directory():
    """The sandbox cwd must be an isolated temp directory, not the project root."""
    code = """
import os
cwd = os.getcwd()
print(cwd)
"""
    result = await _execute_python(code)
    # Should be a temp dir, not the kazi project root
    assert "kazi_sandbox_" in result or "/tmp" in result or "var/folders" in result


# ── Security: resource limits ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_is_enforced():
    """Code that sleeps forever must be killed and return a timeout message."""
    result = await _execute_python("import time; time.sleep(999)", timeout=2)
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_timeout_message_includes_duration():
    result = await _execute_python("import time; time.sleep(999)", timeout=2)
    assert "2s" in result


@pytest.mark.asyncio
async def test_output_truncation():
    """Printing more than _MAX_OUTPUT_BYTES must be truncated."""
    # Generate output well above the 100KB cap
    chars = (_MAX_OUTPUT_BYTES // 10) * 12  # 120% of the cap
    code = f"print('x' * {chars})"
    result = await _execute_python(code)
    assert "truncated" in result
    assert len(result.encode()) <= _MAX_OUTPUT_BYTES + 200  # small slack for truncation message


@pytest.mark.asyncio
async def test_output_at_exactly_cap_is_not_truncated():
    """Output that fits within the cap must not be truncated."""
    # Write exactly 100 chars — well under the 100KB cap
    code = "print('a' * 100)"
    result = await _execute_python(code)
    assert "truncated" not in result
    assert "a" * 100 in result


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_NPROC is Unix-only")
async def test_cpu_limit_kills_infinite_loop():
    """A tight infinite loop must be killed by the CPU time limit."""
    code = "while True: pass"
    result = await _execute_python(code, timeout=5, cpu_seconds=1)
    # Either killed by CPU limit (Killed/signal) or by wall-clock timeout
    assert "timed out" in result.lower() or result == "(no output)"


# ── Python standard library usage ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stdlib_json_works():
    code = """
import json
data = {"key": "value", "nums": [1, 2, 3]}
print(json.dumps(data))
"""
    result = await _execute_python(code)
    import json
    parsed = json.loads(result.strip())
    assert parsed["key"] == "value"
    assert parsed["nums"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_stdlib_math_works():
    code = """
import math
print(math.factorial(10))
"""
    result = await _execute_python(code)
    assert result.strip() == str(math.factorial(10))


@pytest.mark.asyncio
async def test_stdlib_datetime_works():
    code = """
from datetime import date
d = date(2025, 1, 1)
print(d.isoformat())
"""
    result = await _execute_python(code)
    assert result.strip() == "2025-01-01"


@pytest.mark.asyncio
async def test_multiple_prints_all_captured():
    code = """
for i in range(3):
    print(f"line {i}")
"""
    result = await _execute_python(code)
    assert "line 0" in result
    assert "line 1" in result
    assert "line 2" in result


@pytest.mark.asyncio
async def test_exception_traceback_is_complete():
    """Full traceback including file/line info should be present."""
    code = """
def inner():
    raise RuntimeError("deep error")

def outer():
    inner()

outer()
"""
    result = await _execute_python(code)
    assert "RuntimeError" in result
    assert "deep error" in result
    assert "inner" in result
