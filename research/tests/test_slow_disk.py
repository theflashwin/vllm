"""The emulated slow disk tier, built the way run_baselines.sh configures it."""

import mmap
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from kvplace.slow_disk import SerialDevice

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

N_BLOCKS = 4
BLOCK_ELEMENTS = mmap.PAGESIZE  # float32, so 4 pages per block


def test_serial_device_queues_transfers():
    now = [10.0]
    dev = SerialDevice(100.0, clock=lambda: now[0])
    assert dev.reserve(50) == 10.5
    assert dev.reserve(100) == 11.5  # waits for the first transfer
    now[0] = 20.0
    assert dev.reserve(100) == 21.0  # idle device starts at once
    with pytest.raises(ValueError):
        SerialDevice(0)


def _spec() -> MagicMock:
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
        replicated_layout=False,
    )
    spec.blocks_per_chunk = 1
    return spec


def _job(job_id: int, keys) -> TransferJob:
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.arange(len(keys), dtype=np.int64),
        is_promotion=False,
        req_context=ReqContext(req_id="r"),
    )


def test_tier_round_trips_at_the_set_bandwidth(tmp_path):
    raw = torch.zeros(N_BLOCKS * BLOCK_ELEMENTS * 4 + mmap.PAGESIZE, dtype=torch.uint8)
    shift = -raw.data_ptr() % mmap.PAGESIZE
    nbytes = N_BLOCKS * BLOCK_ELEMENTS * 4
    kv = raw[shift : shift + nbytes].view(torch.float32).view(N_BLOCKS, -1)
    kv[:] = torch.rand_like(kv)
    want = kv.clone()
    seconds = 0.2
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "SlowFileSystemTier",
            "module_path": "kvplace.slow_disk",
            "gb_per_s": nbytes / seconds / 1e9,
            "root_dir": str(tmp_path),
        },
        memoryview(kv.numpy()),
        _spec(),
    )
    try:
        assert tier.tier_type == "fs"
        keys = [make_offload_key(i.to_bytes(8, "big"), 0) for i in range(N_BLOCKS)]
        for job_id, submit in enumerate((tier.submit_store, tier.submit_load)):
            if submit == tier.submit_load:
                kv.zero_()
            start = time.monotonic()
            submit(_job(job_id, keys))
            tier.drain_jobs()
            elapsed = time.monotonic() - start
            [result] = tier.get_finished_jobs()
            assert result.success
            assert seconds * 0.95 <= elapsed < seconds * 3
            assert result.transfer_time >= seconds * 0.95
        assert torch.equal(kv, want)
    finally:
        tier.shutdown()
