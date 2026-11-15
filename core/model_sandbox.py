"""Run model code proposed by an agent outside the main process.

An agent proposes a model family as Python source that defines ``predict(designs, params)`` and
``sample_prior(rng, n)`` (see ``core.eig_exec.ModelFamily``). ``SandboxedFamily.from_source``
turns that source into a ``ModelFamily`` whose two functions never execute in this process.

What this does:

* ``validate_model_source`` parses the source with ``ast`` and rejects it before anything runs:
  imports other than ``numpy`` and ``math``; star and relative imports; the names ``open``,
  ``exec``, ``eval``, ``compile``, ``__import__``, ``globals``, ``locals``, ``vars``, ``os``,
  ``sys``, ``subprocess``, ``socket``, ``importlib``, ``builtins`` (and a few similar);
  ``getattr``/``setattr``/``delattr``/``hasattr`` unless the attribute name is a string literal
  without a leading underscore; any attribute starting with ``_``; numpy's file and process
  entry points (``np.load``, ``np.save``, ``np.fromfile``, ``tofile``, ``memmap``,
  ``DataSource``, ``ctypeslib``, ``f2py``, ``testing``, ``lib`` ...); frame and code-object
  attributes (``gi_frame``, ``f_back``, ``f_globals`` ...); string literals containing ``__``.
* Every ``predict`` or ``sample_prior`` call starts a fresh interpreter (``sys.executable -I``)
  in a new empty temporary directory with a minimal environment (``PATH``, ``SYSTEMROOT`` and
  single-threaded BLAS settings). Arrays travel as ``.npy`` bytes over stdin and stdout with
  ``allow_pickle=False``. Inside the child the code runs with a reduced ``__builtins__`` whose
  ``__import__`` only admits numpy and math, and anything it prints goes to stderr.
* A watchdog thread in the child ends it ``timeout_s`` seconds after the model code starts;
  the parent also kills it if it has not finished ``timeout_s + STARTUP_GRACE_S`` seconds after
  launch (start-up, mostly importing numpy, is not charged to the model). Either way the caller
  gets ``ModelTimeout``. A crash or a bad result gives ``ModelError`` with the child's stderr.
* ``sample_prior`` stays deterministic: the parent draws an integer seed from the caller's
  generator and the child builds ``np.random.default_rng(seed)`` from it.

What this does NOT do:

* It is not an OS-level sandbox. There are no seccomp filters, namespaces, job objects, or
  file-system or network isolation. The child runs as the same user with the same rights.
  File and network access are blocked only by the AST screen and the reduced builtins, which are
  defence in depth against mistakes and casual misuse, not a proof against a determined attacker
  (Python introspection has many corners).
* Memory and CPU use are not capped, beyond the time limit. A watchdog cannot interrupt one long
  C-level numpy call that holds the GIL; the parent's hard kill still ends it.
* For hostile code, run the whole evaluation inside a real sandbox (the project's Modal
  sandboxes, a container, or a VM).
"""

from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from typing import Any

import numpy as np

from core.eig_exec import ModelFamily

__all__ = [
    "STARTUP_GRACE_S",
    "ModelError",
    "ModelSourceError",
    "ModelTimeout",
    "SandboxedFamily",
    "UnsafeModelCode",
    "validate_model_source",
]

STARTUP_GRACE_S = 10.0  # allowance for interpreter start-up and importing numpy in the child
_TIMEOUT_EXIT = 124
_STDERR_TAIL = 4000

_ALLOWED_MODULES = frozenset({"numpy", "math"})
_REQUIRED_FUNCTIONS = ("predict", "sample_prior")
_FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "help",
        "exit",
        "quit",
        "memoryview",
        "os",
        "sys",
        "subprocess",
        "socket",
        "importlib",
        "builtins",
        "ctypes",
        "shutil",
        "pathlib",
        "pickle",
        "marshal",
    }
)
_ATTRIBUTE_FUNCTIONS = frozenset({"getattr", "setattr", "delattr", "hasattr"})
_FORBIDDEN_ATTRIBUTES = frozenset(
    {
        # numpy file, process and foreign-function entry points
        "load",
        "loads",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "tofile",
        "fromregex",
        "memmap",
        "open_memmap",
        "DataSource",
        "ctypeslib",
        "ctypes",
        "cffi",
        "f2py",
        "distutils",
        "testing",
        "lib",
        "dump",
        "dumps",
        # frame and code objects (a route back to the runner's real builtins)
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "cr_frame",
        "cr_code",
        "cr_await",
        "ag_frame",
        "ag_code",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "tb_frame",
        "tb_next",
        "co_code",
    }
)


class ModelSourceError(ValueError):
    """Model source that cannot be used (syntax error, missing functions)."""


class UnsafeModelCode(ModelSourceError):
    """Model source that uses a forbidden import, name or attribute."""


class ModelError(RuntimeError):
    """Model code failed while running, or returned something unusable."""


class ModelTimeout(ModelError):
    """Model code did not finish within its time limit."""


def _where(node: ast.AST) -> str:
    return f"line {getattr(node, 'lineno', '?')}"


def _check_module_path(path: str, node: ast.AST) -> None:
    parts = path.split(".")
    if parts[0] not in _ALLOWED_MODULES:
        raise UnsafeModelCode(
            f"{_where(node)}: import of {path!r} is not allowed (only numpy and math)"
        )
    for part in parts[1:]:
        if part.startswith("_") or part in _FORBIDDEN_ATTRIBUTES:
            raise UnsafeModelCode(f"{_where(node)}: import of {path!r} is not allowed")


def validate_model_source(src: str) -> None:
    """Statically screen model source. Raises ``UnsafeModelCode`` or ``ModelSourceError``."""
    if not isinstance(src, str):
        raise ModelSourceError("model source must be a string")
    try:
        tree = ast.parse(src, filename="<model>", mode="exec")
    except SyntaxError as exc:
        raise ModelSourceError(f"model source has a syntax error: {exc}") from None

    approved_calls: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _ATTRIBUTE_FUNCTIONS
        ):
            attr = node.args[1] if len(node.args) >= 2 else None
            if not (
                isinstance(attr, ast.Constant)
                and isinstance(attr.value, str)
                and not attr.value.startswith("_")
                and not node.keywords
            ):
                raise UnsafeModelCode(
                    f"{_where(node)}: {node.func.id}() needs a literal attribute name "
                    "that does not start with an underscore"
                )
            approved_calls.add(id(node.func))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module_path(alias.name, node)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                raise UnsafeModelCode(f"{_where(node)}: relative imports are not allowed")
            _check_module_path(node.module, node)
            for alias in node.names:
                if alias.name == "*":
                    raise UnsafeModelCode(f"{_where(node)}: star imports are not allowed")
                if alias.name.startswith("_") or alias.name in _FORBIDDEN_ATTRIBUTES:
                    raise UnsafeModelCode(
                        f"{_where(node)}: importing {alias.name!r} from {node.module!r} "
                        "is not allowed"
                    )
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
                raise UnsafeModelCode(f"{_where(node)}: use of {node.id!r} is not allowed")
            if node.id in _ATTRIBUTE_FUNCTIONS and id(node) not in approved_calls:
                raise UnsafeModelCode(
                    f"{_where(node)}: {node.id} may only be called directly with a literal name"
                )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeModelCode(
                    f"{_where(node)}: attribute {node.attr!r} is not allowed "
                    "(no names starting with an underscore)"
                )
            if node.attr in _FORBIDDEN_ATTRIBUTES:
                raise UnsafeModelCode(f"{_where(node)}: attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise UnsafeModelCode(
                f"{_where(node)}: string literals containing '__' are not allowed"
            )

    defined: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            defined.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            defined.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    missing = [name for name in _REQUIRED_FUNCTIONS if name not in defined]
    if missing:
        raise ModelSourceError(
            "model source must define top-level functions predict(designs, params) and "
            f"sample_prior(rng, n); missing: {', '.join(missing)}"
        )


# The child process. It reads one JSON header line plus .npy arrays from stdin, runs the model
# code with reduced builtins under a watchdog, and writes one .npy array to the original stdout.
_RUNNER = r"""
import io
import json
import os
import sys
import threading

SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "callable", "complex", "delattr", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "getattr", "hasattr", "hash", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object", "pow",
    "print", "range", "repr", "reversed", "round", "set", "setattr", "slice", "sorted",
    "str", "sum", "tuple", "zip", "property", "staticmethod", "classmethod", "super",
    "__build_class__", "ArithmeticError", "AssertionError", "Exception", "FloatingPointError",
    "IndexError", "KeyError", "LookupError", "NotImplementedError", "OverflowError",
    "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)
ALLOWED_MODULES = ("numpy", "math")


def expire():
    try:
        os.write(2, b"model code exceeded its time limit\n")
    finally:
        os._exit(TIMEOUT_EXIT)


def main():
    raw = sys.stdin.buffer.read()
    head, _, body = raw.partition(b"\n")
    header = json.loads(head.decode("utf-8"))
    out_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(out_fd, os.O_BINARY)

    import builtins
    import numpy as np

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or name.partition(".")[0] not in ALLOWED_MODULES:
            raise ImportError("import of %r is not allowed in model code" % (name,))
        return real_import(name, globals, locals, fromlist, level)

    safe = {n: getattr(builtins, n) for n in SAFE_BUILTINS if hasattr(builtins, n)}
    safe["__import__"] = guarded_import
    namespace = {"__builtins__": safe, "__name__": "model"}

    stream = io.BytesIO(body)
    mode = header["mode"]
    if mode == "predict":
        designs = np.load(stream, allow_pickle=False)
        params = np.load(stream, allow_pickle=False)

    watchdog = threading.Timer(float(header["timeout_s"]), expire)
    watchdog.daemon = True
    watchdog.start()
    exec(compile(header["source"], "<model>", "exec"), namespace)
    if mode == "predict":
        designs.setflags(write=False)
        params.setflags(write=False)
        result = namespace["predict"](designs, params)
    else:
        rng = np.random.default_rng(int(header["seed"]))
        result = namespace["sample_prior"](rng, int(header["n"]))
    result = np.asarray(result, dtype=np.float64)
    watchdog.cancel()

    buf = io.BytesIO()
    np.save(buf, result, allow_pickle=False)
    with os.fdopen(out_fd, "wb") as out:
        out.write(buf.getvalue())


main()
""".replace("TIMEOUT_EXIT", str(_TIMEOUT_EXIT))


def _child_env() -> dict[str, str]:
    env = {
        key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if os.environ.get(key) is not None
    }
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.float64), allow_pickle=False)
    return buf.getvalue()


def _tail(data: bytes | str | None) -> str:
    if not data:
        return ""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    text = text.strip()
    return text if len(text) <= _STDERR_TAIL else "..." + text[-_STDERR_TAIL:]


class SandboxedFamily:
    """Runs one family's validated model source in a child process per call."""

    def __init__(self, name: str, source: str, timeout_s: float = 2.0) -> None:
        if not (timeout_s > 0):
            raise ValueError("timeout_s must be positive")
        validate_model_source(source)
        self.name = name
        self.source = source
        self.timeout_s = float(timeout_s)

    @classmethod
    def from_source(
        cls,
        name: str,
        src: str,
        timeout_s: float = 2.0,
        prior_weight: float = 1.0,
        noise_scale: float = 1.0,
    ) -> ModelFamily:
        """Validate ``src`` and wrap it as a ``ModelFamily`` that runs out of process."""
        runner = cls(name, src, timeout_s)
        return ModelFamily(
            name=name,
            predict=runner.predict,
            sample_prior=runner.sample_prior,
            prior_weight=prior_weight,
            noise_scale=noise_scale,
        )

    def predict(self, designs: np.ndarray, params: np.ndarray) -> np.ndarray:
        arrays = [np.asarray(designs, dtype=float), np.asarray(params, dtype=float)]
        return self._call({"mode": "predict"}, arrays)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        seed = int(rng.integers(0, 2**63 - 1))
        return self._call({"mode": "sample_prior", "n": int(n), "seed": seed}, [])

    def _call(self, header: dict[str, Any], arrays: list[np.ndarray], _startup_retry: int = 0) -> np.ndarray:
        header = {**header, "source": self.source, "timeout_s": self.timeout_s}
        payload = json.dumps(header).encode("utf-8") + b"\n" + b"".join(map(_npy_bytes, arrays))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        with tempfile.TemporaryDirectory(prefix="crux-model-", ignore_cleanup_errors=True) as tmp:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", _RUNNER],
                    check=False,
                    input=payload,
                    capture_output=True,
                    cwd=tmp,
                    env=_child_env(),
                    timeout=self.timeout_s + STARTUP_GRACE_S,
                    creationflags=flags,
                )
            except subprocess.TimeoutExpired as exc:
                raise ModelTimeout(
                    f"model {self.name!r}: {header['mode']} did not finish within "
                    f"{self.timeout_s + STARTUP_GRACE_S:g} s including start-up; "
                    f"stderr: {_tail(exc.stderr)}"
                ) from None
        stderr = _tail(proc.stderr)
        startup_error = (proc.stderr or b"").decode("utf-8", errors="replace")
        if (sys.platform == "win32" and proc.returncode != 0 and _startup_retry < 2
                and "DLL load failed while importing _multiarray_umath" in startup_error
                and 'File "<model>"' not in startup_error):
            warnings.warn("Retrying NumPy DLL startup failure before model execution", RuntimeWarning, stacklevel=2)
            time.sleep(.1 * 2 ** _startup_retry)
            return self._call(header, arrays, _startup_retry + 1)
        if proc.returncode == _TIMEOUT_EXIT:
            raise ModelTimeout(
                f"model {self.name!r}: {header['mode']} exceeded {self.timeout_s:g} s; "
                f"stderr: {stderr}"
            )
        if proc.returncode != 0:
            raise ModelError(
                f"model {self.name!r}: {header['mode']} failed with exit code "
                f"{proc.returncode}; stderr: {stderr}"
            )
        try:
            result = np.load(io.BytesIO(proc.stdout), allow_pickle=False)
        except (ValueError, EOFError, OSError) as exc:  # empty or malformed .npy bytes
            raise ModelError(
                f"model {self.name!r}: could not read the result ({exc}); stderr: {stderr}"
            ) from None
        return np.asarray(result, dtype=float)
