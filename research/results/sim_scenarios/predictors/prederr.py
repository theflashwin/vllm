import math, sys, heapq, statistics
from collections import defaultdict
from kvplace.trace import read_trace
from kvplace.predictors import get_predictor

sessions = read_trace(sys.argv[1])
pred = get_predictor("tool-nfh2")
# Events in trace time: each turn finishes, predictor predicts its gap, then the
# gap's end reveals the duration (observe).
ev = []
for s in sessions:
    t = s.start_s
    for k, turn in enumerate(s.turns):
        ev.append((t, 0, id(s), k, s))
        t += turn.tool_duration_s + 1.0
heapq.heapify(ev)
pending = []
rows = []
while ev:
    t, kind, _, k, s = heapq.heappop(ev)
    if kind == 1:
        pred.observe(s.turns[k].tool_name, s.turns[k].tool_duration_s)
        continue
    h = pred(s, k, [])
    last = k == len(s.turns) - 1
    actual = None if last else s.turns[k].tool_duration_s
    tool = h.tool or "?"
    rows.append((tool, h.expected_reuse_s, actual, last))
    if not last:
        heapq.heappush(ev, (t + s.turns[k].tool_duration_s, 1, id(s), k, s))

def cat(tool):
    return "human" if tool == "human" else "tool"
by = defaultdict(list)
for tool, p, a, last in rows:
    by["session end" if last else cat(tool)].append((p, a))
print(f"{'kind':<12}{'n':>6}{'actual p50':>12}{'pred p50':>10}{'|log err| p50':>15}{'pred >10x':>10}{'pred <0.1x':>11}{'no pred':>8}")
for k, xs in by.items():
    acts = [a for p, a in xs if a is not None]
    preds = [p for p, a in xs if p is not None]
    errs = [abs(math.log((p + 0.05) / (a + 0.05))) for p, a in xs if p is not None and a is not None]
    over = sum(1 for p, a in xs if p is not None and a is not None and p > 10 * (a + 0.05))
    under = sum(1 for p, a in xs if p is not None and a is not None and p < 0.1 * (a + 0.05))
    nop = sum(1 for p, a in xs if p is None)
    n = len(xs)
    f = lambda v: f"{statistics.median(v):.2f}" if v else "-"
    print(f"{k:<12}{n:>6}{f(acts):>12}{f(preds):>10}{f(errs):>15}{over/n:>10.0%}{under/n:>11.0%}{nop/n:>8.0%}")
hum = sorted(a for t, p, a, l in rows if t == "human" and a is not None)
if hum:
    q = lambda x: hum[int(x * (len(hum) - 1))]
    print("human waits: p10 %.1fs p25 %.1fs p50 %.1fs p75 %.1fs p90 %.1fs; share < 30 s: %.0f%%" % (q(.1), q(.25), q(.5), q(.75), q(.9), 100 * sum(a < 30 for a in hum) / len(hum)))
