You are a senior systems engineer. Build a complete, production-grade, single-file Python 3.12+ module implementing a persistent, crash-safe, asynchronous task scheduler and worker pool with dependency DAG resolution.

### Requirements
1. **Zero External Dependencies:** Use ONLY the Python 3.12+ standard library (`asyncio`, `sqlite3`, `json`, `time`, `uuid`, `dataclasses`, `typing`, `logging`, etc.).
2. **DAG & Dependency Management:**
   - Tasks can declare upstream `depends_on: list[str]` (task IDs).
   - Detect and reject circular dependencies immediately upon submission with a custom `CycleDetectedError`.
   - A task may only transition to `READY` when all upstream dependencies have reached `SUCCEEDED`.
   - If an upstream task fails permanently, all downstream dependent tasks must automatically cascade to `CANCELLED` with a clear failure reason.
3. **Persistence & Crash Recovery:**
   - Persist all task states, payloads, retry counts, and timestamps in SQLite using WAL mode.
   - Implement an atomic lease mechanism (`lease_owner`, `lease_expires_at`) so multiple worker loops cannot claim the same task concurrently.
   - Upon scheduler startup, automatically recover any tasks stuck in `RUNNING` whose lease has expired, returning them to `READY` (or `FAILED` if retries are exhausted).
4. **Execution, Retries & Timeouts:**
   - Support both `async def` and synchronous callables (run sync functions via `asyncio.to_thread`).
   - Enforce per-task execution timeouts (`timeout_seconds`).
   - Implement exponential backoff with jitter for failed/timed-out tasks up to `max_retries`.
5. **Graceful Shutdown:**
   - Provide an `async def shutdown(grace_period: float)` method that stops claiming new tasks, waits for in-flight tasks up to `grace_period`, and cancels/re-queues any remaining tasks cleanly.

### Output Format
Provide your response in two strict sections:
1. **`architecture.md`:** A concise explanation (under 250 words) of your SQLite locking/lease strategy, cycle detection algorithm, and cascade cancellation logic.
2. **`scheduler.py`:** Complete, runnable Python code with strict type hints, docstrings, and an `if __name__ == "__main__":` test suite using `asyncio.run()` that verifies:
   - Diamond DAG execution order (`A -> [B, C] -> D`)
   - Cycle detection rejection
   - Upstream failure cascading to `CANCELLED`
   - Simulated crash recovery of an expired lease
   - Timeout + retry backoff behavior
