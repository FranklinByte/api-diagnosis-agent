"""Bounded retry, timeout, and fallback helpers for model/tool calls.

Socket-level timeouts are necessary for HTTP, but arbitrary synchronous model
or tool callables cannot be interrupted safely from a Python thread.  For
picklable operations this module therefore uses one short-lived spawned worker
process per attempt and terminates it when the total deadline expires.  Local
closures remain supported for quick deterministic tests; their timeout is
observed after return because Python cannot safely kill their thread.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

from .security import redact_text


T = TypeVar("T")


class RemoteOperationError(RuntimeError):
    """An exception raised by an isolated worker process."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        super().__init__(f"{error_type}: {message}")


class _ProcessUnavailable(RuntimeError):
    """Raised when a callable cannot be safely sent to a worker process."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    timeout_seconds: float = 10.0
    backoff_seconds: float = 0.05
    process_isolation: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")


@dataclass(frozen=True)
class CallOutcome(Generic[T]):
    value: Optional[T]
    attempts: int
    elapsed_ms: float
    error: Optional[str] = None
    used_fallback: bool = False


def _worker_entry(connection: Any, operation: Callable[[], Any]) -> None:
    """Run one operation and send only bounded, serializable evidence back."""

    try:
        value = operation()
        try:
            connection.send(("ok", value))
        except (OSError, pickle.PicklingError, TypeError):
            connection.send(("error", "UnserializableResult", "operation result is not serializable"))
    except BaseException as exc:  # worker boundary must report every failure
        try:
            connection.send(("error", type(exc).__name__, redact_text(exc)))
        except (OSError, pickle.PicklingError, TypeError):
            pass
    finally:
        connection.close()


def _is_picklable(value: Any) -> bool:
    try:
        pickle.dumps(value)
        return True
    except (pickle.PicklingError, TypeError, AttributeError, OSError):
        return False


def _terminate_process(process: Any) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=0.5)
    if process.is_alive():
        process.join(timeout=0.1)


def _run_isolated(operation: Callable[[], T], timeout_seconds: float) -> T:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker_entry, args=(child, operation), daemon=True)
    try:
        process.start()
    except (OSError, RuntimeError, pickle.PicklingError, AttributeError, TypeError) as exc:
        parent.close()
        child.close()
        raise _ProcessUnavailable(str(exc)) from exc
    finally:
        # The parent must close its copy of the child endpoint or EOF delivery
        # can be delayed on Windows.
        try:
            child.close()
        except OSError:
            pass

    try:
        if not parent.poll(max(0.001, timeout_seconds)):
            _terminate_process(process)
            raise TimeoutError("operation exceeded timeout budget")
        try:
            payload = parent.recv()
        except (EOFError, OSError) as exc:
            raise RemoteOperationError("WorkerError", "worker exited without a result") from exc
        process.join(timeout=0.5)
        if not payload or payload[0] != "ok":
            error_type = str(payload[1]) if len(payload) > 1 else "WorkerError"
            message = str(payload[2]) if len(payload) > 2 else "operation failed"
            raise RemoteOperationError(error_type, message)
        return payload[1]
    finally:
        if process.is_alive():
            _terminate_process(process)
        else:
            process.join(timeout=0.1)
        parent.close()


def _run_inline(operation: Callable[[], T]) -> T:
    """Run unpicklable operations without creating a leaked background thread."""

    return operation()


def _run_with_timeout(
    operation: Callable[[], T],
    timeout_seconds: float,
    *,
    process_isolation: bool,
) -> T:
    if timeout_seconds <= 0:
        raise TimeoutError("operation exceeded timeout budget")
    if process_isolation and _is_picklable(operation):
        try:
            return _run_isolated(operation, timeout_seconds)
        except _ProcessUnavailable:
            # Preserve support for local closures and test doubles.  Such a
            # callable is executed inline and can only be timed after return.
            pass
    started = time.perf_counter()
    value = _run_inline(operation)
    if time.perf_counter() - started > timeout_seconds:
        raise TimeoutError("operation exceeded timeout budget")
    return value


def call_with_fallback(
    operation: Callable[[], T],
    *,
    fallback: Optional[Callable[[Exception], T]] = None,
    policy: RetryPolicy = RetryPolicy(),
) -> CallOutcome[T]:
    """Execute bounded attempts under one total deadline.

    Retries and backoff consume the same ``timeout_seconds`` budget.  A timeout
    kills an isolated worker before returning; no unbounded executor or
    non-daemon thread is left behind.  If a callable is not picklable, callers
    should use a native client timeout as well because Python cannot preempt an
    arbitrary inline function.
    """

    started = time.perf_counter()
    deadline = started + policy.timeout_seconds
    last_error: Optional[Exception] = None
    attempts = 0
    for attempt in range(1, policy.max_attempts + 1):
        attempts = attempt
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            last_error = TimeoutError("operation exceeded total timeout budget")
            break
        try:
            value = _run_with_timeout(
                operation,
                remaining,
                process_isolation=policy.process_isolation,
            )
            elapsed = (time.perf_counter() - started) * 1000
            return CallOutcome(value=value, attempts=attempt, elapsed_ms=elapsed)
        except Exception as exc:
            last_error = exc
            if attempt >= policy.max_attempts:
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if policy.backoff_seconds:
                time.sleep(min(policy.backoff_seconds * attempt, remaining))

    elapsed = (time.perf_counter() - started) * 1000
    if fallback and last_error and time.perf_counter() < deadline:
        try:
            remaining = deadline - time.perf_counter()
            fallback_error: Exception = last_error
            value = _run_with_timeout(
                lambda: fallback(fallback_error),
                remaining,
                process_isolation=policy.process_isolation,
            )
            return CallOutcome(
                value=value,
                attempts=attempts,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error=redact_text(last_error),
                used_fallback=True,
            )
        except Exception as exc:
            last_error = exc
    return CallOutcome(
        value=None,
        attempts=attempts or policy.max_attempts,
        elapsed_ms=elapsed,
        error=redact_text(last_error) if last_error else "operation failed",
    )


__all__ = ["CallOutcome", "RemoteOperationError", "RetryPolicy", "call_with_fallback"]
