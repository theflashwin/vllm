"""vLLM's fs KV tier with its disk slowed to a set bandwidth.

Emulates a slower disk than the node has, so runs can make the disk tier the
bottleneck. Each direction behaves like a serial device: a transfer of n bytes
finishes no earlier than n / rate after the previous transfer in that
direction ends. Reads and writes are limited separately, so the write-through
traffic does not take read bandwidth; a real shared disk would be slower.

Secondary tier config (metrics keep the label "fs"):
    {"type": "SlowFileSystemTier", "module_path": "kvplace.slow_disk",
     "root_dir": "/tmp/kvplace_fs", "gb_per_s": 0.25}
"""

import threading
import time
from collections.abc import Callable, Iterable

from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager


class SerialDevice:
    """Hands out finish times for transfers on a device of fixed bandwidth."""

    def __init__(
        self, bytes_per_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if bytes_per_s <= 0:
            raise ValueError(f"bandwidth must be positive, got {bytes_per_s}")
        self._rate = bytes_per_s
        self._clock = clock
        self._free_at = 0.0
        self._lock = threading.Lock()

    def reserve(self, nbytes: int) -> float:
        """Returns when a transfer of nbytes started now would finish."""
        with self._lock:
            start = max(self._clock(), self._free_at)
            self._free_at = start + nbytes / self._rate
            return self._free_at


def throttled(
    task: Callable[[], None],
    device: SerialDevice,
    nbytes: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[], None]:
    """Runs task, then waits until the device would have finished it."""

    def run() -> None:
        done_at = device.reserve(nbytes)
        task()
        delay = done_at - clock()
        if delay > 0:
            sleep(delay)

    return run


class _ThrottledPool:
    """Wraps the tier's thread pool so each job waits for its device time."""

    def __init__(self, pool, tier: "SlowFileSystemTier") -> None:
        self._pool = pool
        self._tier = tier

    def __getattr__(self, name: str):
        return getattr(self._pool, name)

    def _wrap(
        self, job_id, n_tasks: int, tasks: Iterable[Callable], device: SerialDevice
    ) -> list[Callable]:
        nbytes = self._tier._job_block_counts[job_id] * self._tier._block_size
        return [throttled(t, device, nbytes // n_tasks) for t in tasks]

    def enqueue_load(self, job_id, n_tasks: int, tasks: Iterable[Callable]) -> None:
        self._pool.enqueue_load(
            job_id, n_tasks, self._wrap(job_id, n_tasks, tasks, self._tier.read_dev)
        )

    def enqueue_store(self, job_id, n_tasks: int, tasks: Iterable[Callable]) -> None:
        self._pool.enqueue_store(
            job_id, n_tasks, self._wrap(job_id, n_tasks, tasks, self._tier.write_dev)
        )


class SlowFileSystemTier(FileSystemTierManager):
    """FileSystemTierManager limited to gb_per_s (1 GB = 1e9 bytes).

    Args:
        gb_per_s: Read bandwidth.
        write_gb_per_s: Write bandwidth; defaults to gb_per_s.

    """

    def __init__(
        self,
        *args,
        gb_per_s: float,
        write_gb_per_s: float | None = None,
        tier_type: str = "fs",
        **kwargs,
    ) -> None:
        super().__init__(*args, tier_type="fs", **kwargs)
        self.read_dev = SerialDevice(gb_per_s * 1e9)
        self.write_dev = SerialDevice((write_gb_per_s or gb_per_s) * 1e9)
        self._pool = _ThrottledPool(self._pool, self)
