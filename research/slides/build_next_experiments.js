const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/Users/home/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/e19991af-638a-4b62-a5a4-387f1007c000/1399b0ab-3158-46ab-89db-433056873a43/skills/pptx/scripts/apply_theme.js");

const FIG = "/Users/home/Documents/GitHub/vllm/research/figures/";
const OUT = "/Users/home/Documents/GitHub/vllm/research/slides/next_experiments.pptx";

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


{
  const s = pres.addSlide({ masterName: "TITLE_ONLY" });
  s.addText("Next Experiments", { placeholder: "title" });
  table(s, [
    [head("Experiment"), head("What it tests"), head("Why it is promising")],
    [cell("live70, CPU tier 12.6 and 21 GB, disk 0.25 GB/s"), cell("LRU vs ours vs ours with true reuse times, live"), cell("Mid-sized DRAM and slow misses: the setting where the simulator shows a gap")],
    [cell("Compaction hints, live"), cell("The replay tells the policy when the agent compacts its context"), cell("Simulator: ours -1% to -35% p99 against LRU")],
    [cell("No disk tier"), cell("Evicted KV must be recomputed, not reloaded"), cell("Misses cost more: without offloading p99 was 3.1 s vs 1.6 s")],
    [cell("Shared disk, live"), cell("Reads and writes share one disk bandwidth"), cell("Simulator: true reuse times cut p99 by 80%")],
    [cell("Longer contexts, live"), cell("A new TraceLab trace with 2× longer contexts"), cell("Simulator: ours -35% p99, matching true reuse times")],
  ], { x: X0, y: 1.7, w: W, colW: [3.3, 3.9, W - 7.2], rowH: 0.55 });
  summary(s, ["If ours still matches LRU here, the paper explains why capacity and disk speed, not eviction order, set agent latency."], 5.3);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
