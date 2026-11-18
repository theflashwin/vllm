const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/home/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/e19991af-638a-4b62-a5a4-387f1007c000/1399b0ab-3158-46ab-89db-433056873a43/skills/pptx/scripts/apply_theme.js");

const FIG = "/Users/home/Documents/GitHub/vllm/research/figures/";
const OUT = "/Users/home/Documents/GitHub/vllm/research/slides/checkin5_story.pptx";

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

// 28: what we did / tried
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Check in 5: What We Did", { placeholder: "title" });
  addBullets(s, [
    "Moved live runs to PACE (L40S, Llama 3.1 8B) so each setting can be repeated",
    "Hints now reach the policy by UDP instead of an extra request per turn, halving engine requests",
    "Looked for a setting where eviction matters: slow disk, larger CPU tier, then the simulator",
  ], X0, 1.6, W, 1.1);
  table(s, [
    [head("Approach"), head("Result"), head("Evidence")],
    [cell("Ours vs LRU, live"), cell("Same: p90 0.289 vs 0.259 s, within run-to-run noise"), cell("Live, 4 runs each")],
    [cell("Slow disk (0.5, 0.25 GB/s)"), cell("p99 rises to 7-17 s; ours and LRU still the same"), cell("Live, 3 runs each")],
    [cell("32 GB CPU tier"), cell("Whole trace fits: no disk reads, p99 1.5 s, nothing to evict"), cell("Live, 3 runs each")],
    [cell("CPU tier sweep"), cell("Only 6-17 GB gives the policy choices; true reuse times beat LRU, our predictor doesn't"), cell("Simulator")],
  ], { x: X0, y: 2.9, w: W, colW: [3.0, 6.3, W - 9.3] });
  summary(s, [
    "Capacity and disk speed set the tail, not eviction order.",
    "Eviction can only matter with a mid-sized CPU tier and accurate reuse prediction.",
  ], 5.2);
}

// 29: live baseline
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Live Runs: Ours vs LRU", { placeholder: "title" });
  table(s, [
    [head("Resume TTFT, 4 runs each"), head("p50", { align: "right" }), head("p90", { align: "right" }), head("p99", { align: "right" })],
    [cell("LRU"), r("0.115 s"), r("0.289 s"), r("1.56 s")],
    [cell("Ours"), r("0.115 s"), r("0.259 s"), r("1.78 s")],
  ], { x: X0, y: 1.7, w: TW, colW: [2.3, 1.03, 1.03, 1.04] });
  addBullets(s, [
    "The p90 gap is noise: the same setting varies ±40 ms (p = 0.49)",
    "The disk keeps a copy of every block, so eviction only picks a DRAM hit or a disk hit (~26 ms)",
    "Recompute was identical in all 8 runs; 92% of tokens come from the GPU cache",
    "Setup: CPU tier 4.19 GB (about one session), live35 trace (35 Claude Code sessions)",
  ], X0, 2.9, TW, 2.6);
  figure(s, "live35_l40s_pooled.png", FX, 1.7, FW, "Resume TTFT CDFs for LRU, ours and GPU-only, four runs each pooled");
  summary(s, ["Eviction can only matter if a miss is expensive."], 4.95);
}

// 30: slow disk
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Live Runs: Slow Disk", { placeholder: "title" });
  table(s, [
    [head("CPU tier 4.19 GB"), head("LRU p99", { align: "right" }), head("Ours p99", { align: "right" })],
    [cell("Disk 2.1 GB/s (real)*"), r("1.8 s"), r("1.7 s")],
    [cell("Disk 0.5 GB/s"), r("7.5 s"), r("7.4 s")],
    [cell("Disk 0.25 GB/s"), r("14.7 s"), r("16.6 s")],
  ], { x: X0, y: 1.7, w: TW, colW: [2.6, 1.4, 1.4] });
  addBullets(s, [
    "Disk tier slowed in software, like shared or network storage",
    "Loading one ~4 GB session takes ~17 s at 0.25 GB/s, which sets the p99",
    "Ours and LRU match at every speed (p ≥ 0.2, 3 runs each)",
  ], X0, 3.2, TW, 2.2);
  figure(s, "live35_l40s_slowdisk.png", FX, 1.7, FW, "Resume TTFT CDFs at disk speeds 2.1, 0.5 and 0.25 GB/s for LRU and ours");
  summary(s, ["Misses now cost seconds, but the policy still doesn't help."], 4.6);
  s.addText("* Earlier batch, 2 runs each.", {
    x: X0, y: 6.6, w: W, h: 0.3, fontSize: 10, color: C.text2, margin: 0, isTextBox: true,
  });
}

// 31: capacity
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Live Runs: CPU Tier Size", { placeholder: "title" });
  table(s, [
    [head("Disk 0.25 GB/s"), head("CPU 4.19 GB", { align: "right" }), head("CPU 32 GB", { align: "right" })],
    [cell("Resume p99"), r("15-17 s"), r("1.5 s")],
    [cell("Read from disk"), r("86-103 GB"), r("0 GB")],
    [cell("Recomputed tokens"), r("400-480k"), r("279k")],
  ], { x: X0, y: 1.7, w: TW, colW: [2.3, 1.55, 1.55] });
  addBullets(s, [
    "4.19 GB holds about one session: nothing to choose between",
    "32 GB holds the whole trace: nothing to evict, ours and LRU identical",
    "A slow disk also pins CPU blocks, so 11-18% of stores fail and that KV is recomputed",
  ], X0, 3.2, TW, 2.2);
  figure(s, "live35_l40s_cpusize.png", FX, 1.7, FW, "Resume TTFT CDFs at 0.25 GB/s with a 4.19 GB and a 32 GB CPU tier");
  summary(s, ["Capacity sets the tail. Eviction can only matter when DRAM holds some sessions, but not all."], 4.75);
}

// 32: simulator
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Simulator: Where Eviction Can Matter", { placeholder: "title" });
  table(s, [
    [head("Resumes read from disk"), head("4.2 GB", { align: "right" }), head("6.3 GB", { align: "right" }), head("8.4 GB", { align: "right" }), head("12.6 GB", { align: "right" }), head("16.8 GB", { align: "right" }), head("≥ 23 GB", { align: "right" })],
    [cell("LRU"), r("3.5%"), r("2.5%"), r("1.9%"), r("0.7%"), r("0.2%"), r("0%")],
    [cell("Ours, true reuse times"), r("3.4%"), r("2.2%"), r("1.4%"), r("0.4%"), r("0%"), r("0%")],
    [cell("Ours, our predictor"), r("3.8%"), r("2.9%"), r("2.0%"), r("1.6%"), r("0.8%"), r("0%")],
  ], { x: X0, y: 1.7, w: 9.5, colW: [2.9, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1] });
  addBullets(s, [
    "Only between ~6 and 17 GB does the policy have real choices",
    "There, true reuse times beat LRU, but our predictor loses: it evicts sessions that come back sooner than predicted",
    "A bigger trace (70 sessions) widens that range and sends ~3× more resumes to disk",
  ], X0, 3.3, W, 1.6);
  summary(s, ["The open question is prediction accuracy, not where KV lives."], 4.55);
  s.addText("Simulator: GPU cache 3,800 blocks, GPU admission on, default disk costs. Table: live35; last bullet: live70.", {
    x: X0, y: 6.6, w: W, h: 0.3, fontSize: 10, color: C.text2, margin: 0, isTextBox: true,
  });
}

// 33: future directions
{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Future Directions", { placeholder: "title" });
  addBullets(s, [
    [{ text: "Bigger trace: ", bold: true }, { text: "live70 at the highest load the L40S sustains" }],
    [{ text: "Mid-sized CPU tier: ", bold: true }, { text: "12.6 and 21 GB with a 0.25 GB/s disk; LRU vs ours vs ours with true reuse times" }],
    [{ text: "No disk tier: ", bold: true }, { text: "a wrong eviction means a full recompute" }],
    [{ text: "Predictor: ", bold: true }, { text: "if true reuse times beat LRU live, close the gap with our predictor; if not, the paper explains why capacity, not eviction order, sets agent latency" }],
  ], X0, 1.75, W, 3.0);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
