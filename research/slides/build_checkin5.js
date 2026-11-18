const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/home/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/e19991af-638a-4b62-a5a4-387f1007c000/1399b0ab-3158-46ab-89db-433056873a43/skills/pptx/scripts/apply_theme.js");

const FIG = "/Users/home/Documents/GitHub/vllm/research/figures/";
const OUT = "/Users/home/Documents/GitHub/vllm/research/slides/checkin5.pptx";

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

// 28: what we did
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Check-in 5: What We Did", { placeholder: "title" });
  addBullets(s, [
    "Moved live runs to PACE-ICE: L40S, Llama 3.1 8B (Qwen is not allowed there), repeated runs per setting",
    "Setup: GPU cache 3,800 blocks, CPU tier 4.19 GB (about one session), local disk ~2.1 GB/s, live35 trace (35 Claude Code sessions, 811 turns)",
    "Load the server can sustain: median resume TTFT 0.11 s, against ~18 s in the overloaded Check-in 3 A100 run",
    "Removed the extra hint request: the reuse estimate is only known after the response, so it used to be sent as a second, padded request each turn",
    "It now goes to the policy as a UDP message: 811 engine requests instead of 1,622, and all 811 hints arrived",
    "Tested a slow disk (emulated) and a larger CPU tier: 30 more runs",
  ], X0, 1.75, W, 4.5);
  s.addNotes(
    "We moved live runs from Modal to PACE so we can repeat each setting. With request-based hints, one run had a resume " +
    "p99 of 2.4 s, above every UDP run (1.3-2.0 s); that comparison rests on a single run.",
  );
}

// 29: policy vs LRU
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Policy vs LRU at a Sustainable Load", { placeholder: "title" });
  table(s, [
    [head("Resume TTFT, 4 runs each"), head("p50", { align: "right" }), head("p90", { align: "right" }), head("p99", { align: "right" })],
    [cell("LRU"), r("0.115 s"), r("0.289 s"), r("1.56 s")],
    [cell("Ours"), r("0.115 s"), r("0.259 s"), r("1.78 s")],
  ], { x: X0, y: 1.7, w: 5.4, colW: [2.3, 1.03, 1.03, 1.04] });
  addBullets(s, [
    "No difference: run-level p90, p = 0.49; the same setting varies about ±40 ms at p90",
    "The disk tier keeps every block, so CPU eviction only chooses between a DRAM hit and a disk hit, never recompute",
    "Recomputed tokens were identical in all 8 runs: 278,878",
    "Prompt tokens: 92% from the GPU cache, 5% from CPU or disk, 3% recomputed",
  ], X0, 2.85, 5.4, 3.5);
  figure(s, "live35_l40s_pooled.png", 6.6, 1.7, 5.85, "Resume TTFT CDFs for LRU, ours and GPU-only, four runs each pooled");
}

// 30: slow disk
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Slow Disk: Tail Grows, Policy Still Doesn't Help", { placeholder: "title" });
  table(s, [
    [head("4.19 GB CPU, 3 runs"), head("LRU p99", { align: "right" }), head("Ours p99", { align: "right" }), head("Queue wait", { align: "right" })],
    [cell("Disk 2.1 GB/s (real)*"), r("1.8 s"), r("1.7 s"), r("80 ms")],
    [cell("Disk 0.5 GB/s"), r("7.5 s"), r("7.4 s"), r("340 ms")],
    [cell("Disk 0.25 GB/s"), r("14.7 s"), r("16.6 s"), r("740 ms")],
  ], { x: X0, y: 1.7, w: 5.4, colW: [2.3, 1.03, 1.03, 1.04] });
  addBullets(s, [
    "Emulated disk: each read and write waits until a disk of the set speed would have finished it",
    "At 0.25 GB/s one ~4 GB session takes ~17 s to load, which matches the p99",
    "Only the slowest 5-10% of resumes change; p50 stays at 0.11 s",
    "Ours vs LRU: p ≥ 0.2 for p90, p99 and mean at every speed",
  ], X0, 3.2, 5.4, 3.0);
  figure(s, "live35_l40s_slowdisk.png", 6.6, 1.7, 5.85, "Resume TTFT CDFs at disk speeds 2.1, 0.5 and 0.25 GB/s for LRU and ours");
  s.addText("* Earlier batch, 2 runs each. Queue wait: mean time before prefill.", {
    x: X0, y: 6.55, w: W, h: 0.3, fontSize: 10, color: C.text2, margin: 0, isTextBox: true,
  });
}

// 31: capacity, not policy
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Capacity, Not Policy", { placeholder: "title" });
  table(s, [
    [head("Disk 0.25 GB/s"), head("CPU 4.19 GB", { align: "right" }), head("CPU 32 GB", { align: "right" })],
    [cell("Resume p99"), r("15-17 s"), r("1.5 s")],
    [cell("Read from disk"), r("86-103 GB"), r("0 GB")],
    [cell("CPU allocations that fail"), r("11-18%"), r("0%")],
    [cell("Recomputed tokens"), r("400-480k"), r("279k")],
  ], { x: X0, y: 1.7, w: 5.4, colW: [2.3, 1.55, 1.55] });
  addBullets(s, [
    "32 GB holds the whole trace, so nothing is read from disk and there is nothing to evict",
    "A slow disk keeps CPU blocks pinned while they are written, so new KV can't be stored and is recomputed later",
    "Both effects come from capacity and disk speed, not from which block is evicted",
  ], X0, 3.55, 5.4, 2.8);
  figure(s, "live35_l40s_cpusize.png", 6.6, 1.7, 5.85, "Resume TTFT CDFs at 0.25 GB/s with a 4.19 GB and a 32 GB CPU tier");
}

// 32: simulator CPU-size sweep
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Where Could the Policy Matter? (Simulator)", { placeholder: "title" });
  table(s, [
    [head("Resumes read from disk"), head("4.2 GB", { align: "right" }), head("6.3 GB", { align: "right" }), head("8.4 GB", { align: "right" }), head("12.6 GB", { align: "right" }), head("16.8 GB", { align: "right" }), head("≥ 23 GB", { align: "right" })],
    [cell("LRU"), r("3.5%"), r("2.5%"), r("1.9%"), r("0.7%"), r("0.2%"), r("0%")],
    [cell("Ours, true reuse times"), r("3.4%"), r("2.2%"), r("1.4%"), r("0.4%"), r("0%"), r("0%")],
    [cell("Ours, our predictor"), r("3.8%"), r("2.9%"), r("2.0%"), r("1.6%"), r("0.8%"), r("0%")],
  ], { x: X0, y: 1.7, w: 9.5, colW: [2.9, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1] });
  addBullets(s, [
    "Below ~6 GB the CPU tier holds about one session, so no policy has real choices; above ~17 GB everything fits",
    "In between, with true reuse times, ours reads less from disk (0.7% → 0.4% of resumes at 12.6 GB)",
    "Our real predictor is worse than LRU in that range, so prediction accuracy is the open question",
    "A bigger trace (70 sessions, contexts twice as long) sends ~3× more resumes to disk; true reuse times halve them at 12.6 GB (4.7% → 2.2%)",
  ], X0, 3.4, W, 2.8);
  s.addText("Simulator: GPU cache 3,800 blocks, GPU admission on, default disk costs. Table: live35 at time scale 0.5; last bullet: live70 at 2.0.", {
    x: X0, y: 6.55, w: W, h: 0.3, fontSize: 10, color: C.text2, margin: 0, isTextBox: true,
  });
}

// 33: next steps
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Next Steps", { placeholder: "title" });
  addBullets(s, [
    [{ text: "Bigger trace: ", bold: true }, { text: "live70 (70 sessions); load check running on PACE to find the highest rate the L40S sustains" }],
    [{ text: "Make eviction a real choice: ", bold: true }, { text: "at that rate, CPU tier 12.6 and 21 GB, disk 0.25 GB/s; LRU vs ours vs ours with true reuse times" }],
    [{ text: "Make wrong evictions expensive: ", bold: true }, { text: "the same runs with no disk tier, so an evicted session must be recomputed" }],
    [{ text: "Predictor: ", bold: true }, { text: "if true reuse times win live but ours doesn't, close that gap" }],
    [{ text: "Final paper: ", bold: true }, { text: "either the setting where agent-aware eviction helps, or why capacity and disk speed set the tail instead" }],
  ], X0, 1.75, W, 3.5);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
