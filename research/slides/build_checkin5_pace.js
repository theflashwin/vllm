const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/home/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/e19991af-638a-4b62-a5a4-387f1007c000/1399b0ab-3158-46ab-89db-433056873a43/skills/pptx/scripts/apply_theme.js");

const FIG = "/Users/home/Documents/GitHub/vllm/research/figures/";
const OUT = "/Users/home/Documents/GitHub/vllm/research/slides/checkin5_pace.pptx";

const THEME = {
  name: "CS 6222 Check-in",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "202124", lt1: "FFFFFF", dk2: "5F6368", lt2: "EEF1F5",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "B3261E", accent6: "137333", hlink: "2A78D6", folHlink: "6D2E46",
  },
};
const HEX = { border: "D0D4DA", grid: "E4E3DF", muted: "5F6368" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5, same as the existing deck
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "CS 6222 Check-in 5";
const C = pres.SchemeColor;

const X0 = 0.88;
const W = 13.333 - 2 * X0;
const BODY = 13;

pres.defineSlideMaster({
  title: "TITLE_ONLY",
  background: { color: "FFFFFF" },
  objects: [
    {
      placeholder: {
        options: {
          name: "title", type: "title", x: X0, y: 0.7, w: W, h: 0.7,
          fontSize: 28, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0,
        },
        text: "",
      },
    },
  ],
  slideNumber: { x: 12.25, y: 6.95, w: 0.4, h: 0.3, fontSize: 9, color: C.text2, align: "right" },
});

const bullets = (items, size = BODY) =>
  items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({
      text: r.text,
      options: {
        ...(j === 0 ? { bullet: { indent: 12 }, paraSpaceAfter: 5 } : {}),
        ...(r.bold ? { bold: true } : {}),
        fontSize: size,
        color: C.text1,
        breakLine: j === runs.length - 1 && i < items.length - 1,
      },
    }));
  }).flat();

const addBullets = (slide, items, x, y, w, h) =>
  slide.addText(bullets(items), { x, y, w, h, valign: "top", margin: 0, isTextBox: true, objectName: "Bullets" });

const cell = (text, opts = {}) => ({
  text,
  options: { fontSize: 12, color: C.text1, valign: "middle", margin: [2, 6, 2, 6], ...opts },
});
const r = (t, o = {}) => cell(t, { align: "right", ...o });
const head = (text, opts = {}) => cell(text, { bold: true, fill: { color: C.background2 }, ...opts });
const table = (slide, rows, opts) =>
  slide.addTable(rows, { border: { type: "solid", pt: 0.75, color: HEX.border }, rowH: 0.32, objectName: "Table", ...opts });

const figure = (slide, file, x, y, w, alt) =>
  slide.addImage({ path: FIG + file, x, y, w, h: w * 680 / 1800, altText: alt, objectName: "Figure" });

const summary = (slide, lines, y) =>
  slide.addText(
    lines.map((t, i) => ({ text: t, options: { breakLine: i < lines.length - 1, paraSpaceAfter: 8 } })),
    { x: X0, y, w: W, h: 0.35 * lines.length + 0.1, fontSize: BODY, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Summary" },
  );
const FX = 6.6, FW = 5.85, TW = 5.4;


// 28: overview
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Check in 5: What We Did", { placeholder: "title" });
  addBullets(s, [
    [{ text: "Moved live experiments to PACE: ", bold: true }, { text: "why, and the problems we had to fix" }],
    [{ text: "Baselines on PACE: ", bold: true }, { text: "ours vs LRU vs no offloading, then a slow disk and a larger CPU tier" }],
    [{ text: "Simulator scenarios: ", bold: true }, { text: "a bigger trace, a shared disk and longer contexts" }],
    [{ text: "Next experiments: ", bold: true }, { text: "what we are running and why each is promising" }],
  ], X0, 1.75, W, 2.2);
  summary(s, [
    "Baselines so far: offloading helps the tail, but which block we evict does not.",
    "The next experiments target the one setting the simulator says can change that.",
  ], 3.3);
}

// 29: why PACE
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Moving Live Runs to PACE", { placeholder: "title" });
  table(s, [
    [head(""), head("Modal (Check-ins 1-4)"), head("PACE-ICE (now)")],
    [cell("Cost"), cell("Paid credits, under $30/month"), cell("Free course allocation")],
    [cell("Runs per setting"), cell("One"), cell("3-4, many in parallel")],
    [cell("GPU"), cell("A10G, A100"), cell("L40S (48 GB)")],
    [cell("Model"), cell("Qwen2.5-7B"), cell("Llama 3.1 8B (Qwen is not allowed)")],
    [cell("Load"), cell("Overloaded (median TTFT ~18 s on A100)"), cell("Sustainable (median 0.11 s)")],
  ], { x: X0, y: 1.7, w: 8.6, colW: [2.2, 3.2, 3.2] });
  addBullets(s, [
    "Runs of the same setting vary by ±40 ms at p90, so one run cannot show a gain under ~15%",
    "Check-in 3's live result came from a single, overloaded run",
    "On PACE we ran over 40 live runs in a week",
  ], X0, 4.1, W, 1.4);
  summary(s, ["We needed repeated runs at a sustainable load before trusting any live result."], 5.6);
}

// 30: problems
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("PACE: Problems We Ran Into", { placeholder: "title" });
  table(s, [
    [head("Problem"), head("Fix")],
    [cell("PACE prohibits Qwen models"), cell("Llama 3.1 8B; its KV is 2.3× larger per token, so the CPU tier was rescaled to 4.19 GB to hold the same tokens")],
    [cell("Every job died at startup"), cell("vLLM's subprocess imported our uncompiled source tree from the repo root; each server now starts in its results folder")],
    [cell("Engine failed to start"), cell("FlashInfer's sampler compiles at startup and needs ninja, which compute nodes lack; use vLLM's PyTorch sampler")],
    [cell("Hints doubled the load"), cell("Each hint was an extra padded request per turn; hints now reach the policy by UDP (811 of 811 received)")],
    [cell("Shared nodes and time limits"), cell("Ports from the job ID; check the disk tier is a real disk; stop replays before the walltime so results are saved")],
    [cell("Real disk too fast to matter"), cell("At 2.1 GB/s a disk read costs ~26 ms; we added an emulated slow disk (0.5, 0.25 GB/s)")],
  ], { x: X0, y: 1.7, w: W, colW: [3.2, W - 3.2], rowH: 0.42 });
}

// 31: baselines
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Baselines: Ours vs LRU vs No Offloading", { placeholder: "title" });
  table(s, [
    [head("Resume TTFT"), head("p50", { align: "right" }), head("p90", { align: "right" }), head("p99", { align: "right" }), head("Recompute", { align: "right" })],
    [cell("No offloading (1 run)"), r("0.092 s"), r("0.39 s"), r("3.06 s"), r("813k")],
    [cell("LRU (4 runs)"), r("0.115 s"), r("0.29 s"), r("1.56 s"), r("279k")],
    [cell("Ours (4 runs)"), r("0.115 s"), r("0.26 s"), r("1.78 s"), r("279k")],
  ], { x: X0, y: 1.7, w: TW, colW: [1.65, 0.85, 0.85, 0.85, 1.2] });
  addBullets(s, [
    "Offloading recomputes 2.9× less and halves p99; it costs ~23 ms at p50",
    "Ours vs LRU: within run-to-run noise (p = 0.49)",
    "The disk keeps every block, so eviction only picks a DRAM hit or a disk hit (~26 ms)",
  ], X0, 3.3, TW, 2.0);
  figure(s, "live35_l40s_pooled.png", FX, 1.7, FW, "Resume TTFT CDFs for LRU, ours and no offloading");
  summary(s, ["Offloading helps the tail; which block we evict does not."], 5.0);
  s.addText("Setup: one L40S, vLLM 0.31, Llama 3.1 8B; live35 (35 public TraceLab Claude Code sessions); GPU cache ~61k tokens, CPU tier 4.19 GB (about one session), disk 2.1 GB/s. Ours used tool-based hints without compaction hints.", {
    x: X0, y: 6.4, w: W, h: 0.45, fontSize: 11, color: C.text2, margin: 0, valign: "top", isTextBox: true,
  });
}

// 32: slow disk and CPU size
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Baselines: Slow Disk and CPU Tier Size", { placeholder: "title" });
  table(s, [
    [head("Setting"), head("LRU p99", { align: "right" }), head("Ours p99", { align: "right" }), head("Queue wait", { align: "right" }), head("Recompute", { align: "right" })],
    [cell("4.19 GB, disk 2.1 GB/s*"), r("1.8 s"), r("1.7 s"), r("80 ms"), r("279k")],
    [cell("4.19 GB, disk 0.5 GB/s"), r("7.5 s"), r("7.4 s"), r("340 ms"), r("320-460k")],
    [cell("4.19 GB, disk 0.25 GB/s"), r("14.7 s"), r("16.6 s"), r("740 ms"), r("400-480k")],
    [cell("32 GB, disk 0.25 GB/s"), r("1.6 s"), r("1.5 s"), r("60 ms"), r("279k")],
  ], { x: X0, y: 1.7, w: 6.2, colW: [2.1, 0.95, 1.0, 1.05, 1.1] });
  addBullets(s, [
    [{ text: "Tail grows: ", bold: true }, { text: "at 0.25 GB/s one ~4 GB session takes ~17 s to load, and other requests queue behind it" }],
    [{ text: "Recompute grows: ", bold: true }, { text: "CPU blocks stay pinned until written to the slow disk, so 11-18% of CPU stores fail" }],
    [{ text: "Policy can't help: ", bold: true }, { text: "4.19 GB holds about one session, so both policies evict almost everything (~1.3k CPU hits vs ~25k disk hits)" }],
    [{ text: "32 GB fixes it: ", bold: true }, { text: "the whole trace fits, so nothing is read from disk and nothing is evicted" }],
  ], X0, 3.75, 6.2, 2.3);
  figure(s, "live35_l40s_cpusize.png", 7.35, 1.7, 5.1, "Resume TTFT CDFs at 0.25 GB/s with a 4.19 GB and a 32 GB CPU tier");
  summary(s, ["Capacity and disk speed set the tail. Eviction needs a CPU tier between one session and the whole working set."], 6.1);
  s.addText("3 runs per setting; * earlier batch, 2 runs each. Queue wait: mean time before prefill.", {
    x: X0, y: 6.65, w: W, h: 0.3, fontSize: 10, color: C.text2, margin: 0, isTextBox: true,
  });
}

// 33: simulator scenarios
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Simulator: Scenarios Where Eviction Can Matter", { placeholder: "title" });
  table(s, [
    [head("Scenario"), head("What it means"), head("LRU p99", { align: "right" }), head("Ours", { align: "right" }), head("True reuse times", { align: "right" })],
    [cell("live70", { bold: true }), cell("Our next live trace: 70 sessions, contexts ~2× live35's"), r("9.5 s"), r("-10%"), r("-78%")],
    [cell("Shared disk", { bold: true }), cell("live70, but reads and writes share one disk, so writes slow reads"), r("9.2 s"), r("-1%"), r("-80%")],
    [cell("Base", { bold: true }), cell("70 other TraceLab sessions, waits capped at 10 min"), r("5.1 s"), r("-8%"), r("-75%")],
    [cell("Longer contexts", { bold: true }), cell("Base with every context 2× longer; disk 2.1 GB/s"), r("14.4 s"), r("-35%"), r("-34%")],
  ], { x: X0, y: 1.7, w: W, colW: [1.9, 5.3, 1.3, 1.3, W - 9.8], rowH: 0.45 });
  addBullets(s, [
    "Ours drops a session's old KV when the agent compacts its context, which vLLM cannot know on its own",
    "It helps where compactions are common (live70, longer contexts) and loses to LRU when they are rare and waits are long",
    "True reuse times show much more is possible: the rest of the gap is sessions that end without saying so",
  ], X0, 4.75, W, 1.1);
  summary(s, ["Knowing which sessions are over matters more than predicting when the rest return."], 5.95);
  s.addText("CPU tier 21 GB, disk 0.25 GB/s unless noted; one simulator run each; changes are against LRU. Ours: tool-based reuse hints plus a hint when the agent compacts.", {
    x: X0, y: 6.45, w: W, h: 0.45, fontSize: 10, color: C.text2, margin: 0, valign: "top", isTextBox: true,
  });
}

// 34: next experiments
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Next Experiments: Why They Are Promising", { placeholder: "title" });
  table(s, [
    [head("Experiment"), head("Why it is promising")],
    [cell("live70 with a 12.6 or 21 GB CPU tier, disk 0.25 GB/s"), cell("Mid-sized DRAM and slow misses: in the simulator, true reuse times cut p99 by 31-78%")],
    [cell("Ours with true reuse times"), cell("Upper bound: separates \"no setting helps\" from \"our hints are the limit\"")],
    [cell("Compaction hints in live runs"), cell("Send a hint when the agent compacts; simulator: -10% p99 on live70, -35% with longer contexts")],
    [cell("No disk tier"), cell("A wrong eviction means a full recompute; without offloading p99 was 3.1 s vs 1.6 s")],
    [cell("Shared disk, longer contexts"), cell("True reuse times cut p99 by 34-80%; ours helps most with longer contexts")],
  ], { x: X0, y: 1.7, w: W, colW: [4.0, W - 4.0], rowH: 0.5 });
  summary(s, ["Either outcome is a result: we learn when agent-aware eviction helps, or why capacity sets agent latency instead."], 5.1);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
