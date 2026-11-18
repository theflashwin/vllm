"""Simulate tool-nfh2 plus a final flag at compaction / max-context ends only."""
import sys
import kvplace.sim as sim
from kvplace.predictors import ToolMedian, get_predictor
from kvplace.hints import ReuseHint

class CompactionFinal(ToolMedian):
    def __call__(self, session, k, gaps):
        if k == len(session.turns) - 1 and session.meta.get("end_reason") in ("compaction", "max_context"):
            return ReuseHint(session.session_id, k, None, final=True)
        return super().__call__(session, k, gaps)

def patched(name):
    if name == "compact-final":
        return CompactionFinal(200, exact_final=False, human_quantile=0.9, end_aware=False)
    return get_predictor(name)

sim.get_predictor = patched
trace, ts, gpu, disk, cpu, wm = sys.argv[1:7]
sys.argv = ["sim", trace, "--policies", "lru", "reuse_evict", "--predictors", "compact-final",
            "--gpu-blocks", gpu, "--cpu-blocks", cpu, "--time-scale", ts, "--gpu-admission",
            "--contend-sec", "--write-model", wm,
            "--cost-json", f"results/sim_scenarios/cost_llama_disk{disk}.json"]
sim.main()
