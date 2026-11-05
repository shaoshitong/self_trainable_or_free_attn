#!/usr/bin/env node
// Self-check with plain Node: the model's worklists, timelines and numerics, then the generated artifacts
// executed against a small DOM stand-in. This is not a browser: it cannot judge how the result looks.
const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const vm = require("vm");
const { execFileSync } = require("child_process");
const { createModel } = require("./model.js");
const { generate, ROOT_ID } = require("./build.js");
const measured = require("./h100-results.json");

const model = createModel();
const SCENE_IDS = model.SCENES.map((s) => s.id);
const DIRS = ["F", "B"];
const results = [];
function check(name, fn) {
  const detail = fn();
  results.push(name);
  console.log("ok  " + name + (detail ? "  (" + detail + ")" : ""));
}

// --------------------------------------------------------------------- masks and worklists
const SPEC = {
  uniform: { rows: [2, 2, 2, 2, 2, 2, 2, 2], cols: [2, 2, 2, 2, 2, 2, 2, 2], route: "unsplit" },
  hotQ: { rows: [8, 2, 1, 1, 1, 1, 1, 1], cols: [2, 2, 2, 2, 2, 2, 2, 2], route: "F only" },
  hotKV: { rows: [2, 2, 2, 2, 2, 2, 2, 2], cols: [8, 2, 1, 1, 1, 1, 1, 1], route: "B only" },
  both: { rows: [8, 2, 1, 1, 1, 1, 1, 1], cols: [8, 2, 1, 1, 1, 1, 1, 1], route: "both" },
};
const edgeKeys = (edges) => edges.map((e) => e[0] + "," + e[1]).sort();
const maskKeys = (mask) => edgeKeys(mask.flatMap((cols, i) => cols.map((j) => [i, j])));

check("every mask has 16 selected pairs with the specified degrees and route", () => {
  for (const id of SCENE_IDS) {
    const mask = model.MASKS[id];
    assert.strictEqual(model.edgeCount(mask), 16, id);
    assert.strictEqual(new Set(maskKeys(mask)).size, 16, id + " duplicate pair");
    assert.deepStrictEqual(model.degrees(mask), { rows: SPEC[id].rows, cols: SPEC[id].cols }, id);
    assert.strictEqual(model.route(mask).name, SPEC[id].route, id);
  }
  assert.deepStrictEqual(maskKeys(model.MASKS.hotKV), maskKeys(model.transposeMask(model.MASKS.hotQ)));
  assert.deepStrictEqual(model.MASKS.uniform[7], [0, 7]);
});

check("whole-list and chunk worklists cover every selected pair exactly once", () => {
  let lists = 0;
  for (const id of SCENE_IDS) for (const dir of DIRS) {
    const mask = model.MASKS[id];
    const whole = model.wholeWorklist(mask, dir);
    const { tasks } = model.chunkWorklist(mask, dir);
    assert.deepStrictEqual(edgeKeys(whole.flatMap((t) => t.edges)), maskKeys(mask));
    assert.deepStrictEqual(edgeKeys(tasks.flatMap((t) => t.edges)), maskKeys(mask));
    tasks.forEach((t) => assert.ok(t.edges.length <= model.CHUNK, "chunk over the limit"));
    lists += whole.length + tasks.length;
  }
  return lists + " tasks";
});

// --------------------------------------------------------------------- timelines
function auditTimeline(tl, label) {
  const work = tl.tasks.filter((t) => t.edges.length);
  assert.deepStrictEqual(work.flatMap((t) => t.edges).length, 16, label);
  const seen = new Set(work.flatMap((t) => edgeKeys(t.edges)));
  assert.strictEqual(seen.size, 16, label);
  assert.strictEqual(tl.makespan, Math.max(...tl.tasks.map((t) => t.end)), label);
  assert.ok(tl.makespan <= model.TIME_DOMAIN, label + " past the drawn axis");
  assert.ok(tl.makespan >= 16 / model.WORKERS, label + " beats the work bound");
  tl.tasks.forEach((t) => {
    assert.ok(t.lane === -1 || (t.lane >= 0 && t.lane < model.WORKERS), label);
    if (t.edges.length) assert.strictEqual(t.end - t.start, t.edges.length * model.PAIR_COST, label + " " + t.name);
  });
  for (let w = 0; w < model.WORKERS; w++) {
    const lane = tl.tasks.filter((t) => t.lane === w || t.lane === -1).sort((a, b) => a.start - b.start);
    lane.forEach((t, k) => { if (k) assert.ok(t.start >= lane[k - 1].end, label + " lane " + w + " overlaps"); });
  }
  // Sample between scheduling events: no lane runs two pairs at once, and progress only goes forward.
  const samples = tl.events.flatMap((e, k) => (k + 1 < tl.events.length ? [e, (e + tl.events[k + 1]) / 2] : [e]));
  let done = 0;
  samples.forEach((t) => {
    const states = [...model.edgeStates(tl, t).values()];
    const perLane = {};
    states.filter((s) => s.state === "running").forEach((s) => { perLane[s.lane] = (perLane[s.lane] || 0) + 1; });
    Object.values(perLane).forEach((c) => assert.strictEqual(c, 1, label));
    const nowDone = states.filter((s) => s.state === "done").length;
    assert.ok(nowDone >= done, label);
    done = nowDone;
    assert.ok(model.busyLanes(tl, t).size <= model.WORKERS, label);
  });
  assert.strictEqual([...model.edgeStates(tl, tl.makespan).values()].filter((s) => s.state === "done").length, 16, label);
  assert.strictEqual(model.WORKERS * tl.makespan, tl.work + tl.overhead + tl.idle, label + " capacity");
  assert.strictEqual(tl.work, 16, label);
}
function replayGreedy(tasks, free) {
  return tasks.map((t) => {
    let lane = 0;
    free.forEach((f, w) => { if (f < free[lane]) lane = w; });
    const start = free[lane];
    free[lane] = start + t.edges.length * model.PAIR_COST;
    return { lane, start };
  });
}

check("timelines account for 16 pairs with no overlapping lane work", () => {
  let audited = 0;
  for (const id of SCENE_IDS) for (const dir of DIRS) {
    const orig = model.originalTimeline(model.MASKS[id], dir), bal = model.balancedTimeline(model.MASKS[id], dir);
    auditTimeline(orig, id + "/" + dir + "/orig");
    auditTimeline(bal, id + "/" + dir + "/bal");
    audited += 2;
  }
  return audited + " timelines";
});

check("schedules replay as greedy, with planning and merges at the correct barriers", () => {
  for (const id of SCENE_IDS) for (const dir of DIRS) {
    const mask = model.MASKS[id];
    const orig = model.originalTimeline(mask, dir), bal = model.balancedTimeline(mask, dir);
    const replayO = replayGreedy(model.wholeWorklist(mask, dir), [0, 0, 0, 0]);
    orig.tasks.forEach((t, k) => assert.deepStrictEqual({ lane: t.lane, start: t.start }, replayO[k]));
    const work = bal.tasks.filter((t) => t.edges.length);
    const p = model.PLAN_COST;
    const replayB = replayGreedy(model.chunkWorklist(mask, dir).tasks, [p, p, p, p]);
    work.forEach((t, k) => assert.deepStrictEqual({ lane: t.lane, start: t.start }, replayB[k]));
    const workEnd = Math.max(...work.map((t) => t.end));
    const merges = bal.tasks.filter((t) => t.kind === "merge");
    merges.forEach((m) => { assert.ok(m.start >= workEnd); assert.strictEqual(m.end - m.start, model.MERGE_COST); });
    assert.strictEqual(merges.length, bal.splitLists.length);
    assert.strictEqual(bal.tasks.filter((t) => t.kind === "plan").length, 1);
    work.forEach((t) => assert.ok(t.start >= p));
    // Short lists stay direct; only lists longer than the chunk size split.
    model.selectedLists(mask, dir).forEach((l) => {
      const parts = work.filter((t) => t.list === l.list);
      assert.strictEqual(parts.length, Math.max(1, l.edges.length > model.CHUNK ? Math.ceil(l.edges.length / model.CHUNK) : 1));
    });
  }
});

check("makespans match the hand calculation, and uniform gets no fake speedup", () => {
  const expected = {
    uniform: { F: [4, 4.5], B: [4, 4.5], totals: [8, 8, 9] },
    hotQ: { F: [8, 5], B: [4, 4.5], totals: [12, 9, 9.5] },
    hotKV: { F: [4, 4.5], B: [8, 5], totals: [12, 9, 9.5] },
    both: { F: [8, 5], B: [8, 5], totals: [16, 10, 10] },
  };
  for (const id of SCENE_IDS) {
    const v = model.sceneView(id), e = expected[id];
    for (const dir of DIRS) assert.deepStrictEqual([v.dirs[dir].orig.makespan, v.dirs[dir].bal.makespan], e[dir], id + " " + dir);
    assert.deepStrictEqual([v.totals.original, v.totals.routed, v.totals.forced], e.totals, id);
    for (const dir of DIRS) {
      const split = dir === "F" ? v.route.F : v.route.B;
      assert.strictEqual(split, v.dirs[dir].bal.makespan < v.dirs[dir].orig.makespan, id + " " + dir + " router disagrees with the timeline");
    }
  }
  const u = model.sceneView("uniform").dirs.F;
  assert.ok(u.bal.makespan > u.orig.makespan);
  assert.strictEqual(u.bal.tasks.filter((t) => t.kind === "chunk").length, 0);
  return "orig/balanced: uniform 4/4.5, hot 8/5";
});

// --------------------------------------------------------------------- numerics
const SEEDS = model.range(20).map((s) => model.SEED + s);
const TIGHT = 1e-12;

check("partial O + LSE merge matches the full masked softmax on seeded D=4 data", () => {
  let worst = 0;
  for (const id of SCENE_IDS) for (const seed of SEEDS) for (const chunk of [1, 2, 3]) {
    worst = Math.max(worst, model.numericExample(id, { seed, chunk }).forwardDelta);
  }
  assert.ok(worst < TIGHT, "forward delta " + worst);
  return "worst " + worst.toExponential(1) + " over 240 runs, chunks 1/2/3";
});

check("the merge stays stable where a naive exponential overflows", () => {
  let worst = 0;
  const { Q, K } = model.seededTensors(model.SEED, 30);
  const largest = Math.max(...Q.flatMap((q) => K.map((k) => q.reduce((s, x, d) => s + x * k[d], 0))));
  assert.strictEqual(Math.exp(largest), Infinity, "stress data does not overflow; test is vacuous");
  for (const id of SCENE_IDS) {
    const n = model.numericExample(id, { scale: 30 });
    n.fwd.forEach((r) => r.O.concat([r.LSE]).forEach((x) => assert.ok(Number.isFinite(x))));
    worst = Math.max(worst, model.maxAbsDiff(n.fwd.map((r) => r.O), n.ref.map((r) => r.O)));
    const relLse = Math.max(...n.fwd.map((r, i) => Math.abs(r.LSE - n.ref[i].LSE) / Math.max(1, Math.abs(n.ref[i].LSE))));
    assert.ok(relLse < TIGHT, id + " LSE " + relLse);
  }
  assert.ok(worst < 1e-9, "stress O delta " + worst);
  return "largest S " + largest.toFixed(0) + ", worst O delta " + worst.toExponential(1);
});

check("backward from the saved global softmax matches the gradient reference", () => {
  let worst = 0;
  for (const id of SCENE_IDS) for (const seed of SEEDS) for (const chunk of [1, 2, 3]) {
    worst = Math.max(worst, model.numericExample(id, { seed, chunk }).backwardDelta);
  }
  assert.ok(worst < TIGHT, "backward delta " + worst);
  return "worst " + worst.toExponential(1) + " over 240 runs";
});

check("the gradient reference agrees with central finite differences of the loss", () => {
  const h = 1e-5;
  let worst = 0;
  for (const id of SCENE_IDS) {
    const base = model.seededTensors(model.SEED + 7);
    const grad = model.gradientReference(base, model.MASKS[id]);
    const loss = (t) => model.fullMaskedSoftmax(t, model.MASKS[id]).reduce(
      (s, r, i) => s + r.O.reduce((a, x, d) => a + x * t.G[i][d], 0), 0);
    for (const [name, g] of [["Q", grad.dQ], ["K", grad.dK], ["V", grad.dV]]) {
      for (let i = 0; i < model.N; i++) for (let d = 0; d < model.D; d++) {
        const keep = base[name][i][d];
        base[name][i][d] = keep + h; const up = loss(base);
        base[name][i][d] = keep - h; const down = loss(base);
        base[name][i][d] = keep;
        worst = Math.max(worst, Math.abs((up - down) / (2 * h) - g[i][d]));
      }
    }
  }
  assert.ok(worst < 1e-7, "finite difference " + worst);
  return "worst " + worst.toExponential(1);
});

check("the shown wrong paths are actually wrong", () => {
  for (const id of ["hotQ", "both"]) assert.ok(model.numericExample(id).unweightedDelta > 1e-2, id);
  for (const id of ["hotKV", "both"]) assert.ok(model.numericExample(id).independentDelta > 1e-2, id);
  assert.strictEqual(model.numericExample("uniform").unweightedDelta, null);
  assert.strictEqual(model.numericExample("uniform").independentDelta, null);
});

// --------------------------------------------------------------------- generated artifacts
const here = __dirname;
const outDir = path.join(here, "..");
const HTML_PATH = path.join(outDir, "vsa2-load-balance.html");
const CANVAS_PATH = path.join(outDir, "vsa2-load-balance.canvas.tsx");
const html = fs.readFileSync(HTML_PATH, "utf8");
const canvas = fs.readFileSync(CANVAS_PATH, "utf8");

check("generated files are current, small, and self-contained", () => {
  const generated = generate();
  assert.strictEqual(html, generated["vsa2-load-balance.html"], "run build.js");
  assert.strictEqual(canvas, generated["vsa2-load-balance.canvas.tsx"], "run build.js");
  for (const [name, text] of [["html", html], ["canvas", canvas]]) {
    assert.ok(Buffer.byteLength(text) < 1024 * 1024, name);
    assert.ok(!/\bfetch\s*\(|XMLHttpRequest|WebSocket|EventSource|sendBeacon|https?:\/\/|\bimport\s*\(/.test(text), name + " reaches for the network");
    assert.ok(!/document\.currentScript|tabindex|position:\s*fixed/i.test(text), name);
    assert.ok(text.includes("教学时间单位"), name);
    assert.ok(!/TFLOP|MFU/i.test(text), name + " carries an unmeasured utilization claim");
    // Toy timing and historical H100 measurements must remain explicitly separated.
    const h100 = ["不是 H100 benchmark", "H100 TK 直接 reduce-add，没有单独的 dK/dV 合并 kernel"];
    h100.forEach((s) => assert.ok(text.includes(s), name + " lost: " + s));
    assert.ok(text.includes("H100 实测 · FWD / BWD") && text.includes("H100_RESULTS ="), name + " lacks historical results");
    ["与上方教学时间轴分开", "F+B 不能由两个中位数相加", "router 不计", "诊断 H2/K4 另标"].forEach((s) => assert.ok(text.includes(s), name + " lost measurement scope: " + s));
    // Literal colors only as light-dark() last resorts behind theme variables.
    const bare = text.replace(/light-dark\([^)]*\)/g, "").match(/#[0-9a-f]{3,8}\b/gi) || [];
    assert.deepStrictEqual(bare, [], name + " hardcoded color");
  }
  assert.ok(!/<!doctype|<html|<head|<body/i.test(html));
  assert.ok(!html.includes('\\"'), "escaped quotes in the fragment");
  assert.ok(html.includes('document.getElementById("' + ROOT_ID + '")'));
  assert.ok(html.includes('id="' + ROOT_ID + '"') && canvas.includes('id: "' + ROOT_ID + '"'));
  assert.ok(/^import React, \{ useEffect, useRef \} from "react";$/m.test(canvas));
  assert.ok(/^export default function App\(\)/m.test(canvas));
  return Buffer.byteLength(html) + " and " + Buffer.byteLength(canvas) + " bytes";
});

check("the Canvas module parses (no JSX or types, so plain ESM parsing applies)", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vsa2-lb-"));
  const file = path.join(dir, "canvas.mjs");
  fs.writeFileSync(file, canvas);
  execFileSync(process.execPath, ["--check", file], { stdio: "pipe" });
  fs.rmSync(dir, { recursive: true });
});

// --------------------------------------------------------------------- a small DOM stand-in
// SVG shapes are absent on purpose: they must self-close, or a browser nests whatever follows inside them.
const VOID = new Set(["input", "br", "hr"]);
class El {
  constructor(tag, doc) {
    this.tag = tag; this.doc = doc; this.attrs = {}; this.children = []; this.parent = null; this.listeners = {};
    this.text = ""; this.tagName = tag.toUpperCase();
    this.classList = {
      add: (c) => { const set = new Set((this.attrs.class || "").split(/\s+/).filter(Boolean)); set.add(c); this.attrs.class = [...set].join(" "); },
    };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  get value() { return this.attrs.value; }
  set value(v) { this.attrs.value = String(v); }
  get max() { return this.attrs.max; }
  set max(v) { this.attrs.max = String(v); }
  get clientWidth() { return this.doc.widthOf(this); }
  get textContent() { return this.text + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this.children = []; this.text = String(v); }
  set innerHTML(markup) { this.children = []; this.text = ""; parseInto(this, markup, this.doc); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || new Set()).add(fn); }
  removeEventListener(type, fn) { if (this.listeners[type]) this.listeners[type].delete(fn); }
  listenerCount() { return Object.values(this.listeners).reduce((s, l) => s + l.size, 0); }
  dispatch(event) {
    event.target = event.target || this;
    event.preventDefault = event.preventDefault || (() => {});
    for (let el = this; el; el = el.parent) if (el.listeners[event.type]) [...el.listeners[event.type]].forEach((fn) => fn(event));
  }
  click() { this.dispatch({ type: "click" }); }
  matches(selector) { return matchSimple(this, selector); }
  closest(selector) { for (let el = this; el; el = el.parent) if (el.matches(selector)) return el; return null; }
  descendants(out = []) { this.children.forEach((c) => { out.push(c); c.descendants(out); }); return out; }
  querySelectorAll(selector) { return this.descendants().filter((el) => el.matches(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
function matchSimple(el, selector) {
  const m = /^([a-z0-9]*)(?:#([\w-]+))?(?:\.([\w-]+))?(?:\[([\w-]+)(?:="([^"]*)")?\])?$/i.exec(selector);
  assert.ok(m, "selector not handled by the stand-in: " + selector);
  if (m[1] && el.tag !== m[1]) return false;
  if (m[2] && el.attrs.id !== m[2]) return false;
  if (m[3] && !(el.attrs.class || "").split(/\s+/).includes(m[3])) return false;
  if (m[4] && !(m[4] in el.attrs)) return false;
  if (m[5] !== undefined && el.attrs[m[4]] !== m[5]) return false;
  return true;
}
// Parses the markup the view generates and rejects unbalanced tags, which a browser would silently repair.
function parseInto(parent, markup, doc) {
  const token = /<!--[\s\S]*?-->|<\/([a-z0-9]+)\s*>|<([a-z0-9]+)((?:\s+[\w:-]+(?:="[^"]*")?)*)\s*(\/?)>/gi;
  const stack = [parent];
  let at = 0, m;
  const addText = (s) => { if (s.trim()) stack[stack.length - 1].text += s; };
  while ((m = token.exec(markup))) {
    addText(markup.slice(at, m.index));
    at = token.lastIndex;
    if (m[0].startsWith("<!--")) continue;
    if (m[1]) {
      const open = stack.pop();
      assert.strictEqual(open.tag, m[1], "mismatched closing tag </" + m[1] + "> for <" + open.tag + ">");
      continue;
    }
    const el = new El(m[2], doc);
    for (const a of m[3].matchAll(/([\w:-]+)(?:="([^"]*)")?/g)) el.attrs[a[1]] = a[2] === undefined ? "" : a[2];
    el.parent = stack[stack.length - 1];
    el.parent.children.push(el);
    if (m[2] === "style") {
      const end = markup.indexOf("</style>", at);
      el.text = markup.slice(at, end);
      at = end + "</style>".length;
      token.lastIndex = at;
    } else if (!m[4] && !VOID.has(m[2])) stack.push(el);
  }
  addText(markup.slice(at));
  assert.strictEqual(stack.length, 1, "unclosed <" + stack[stack.length - 1].tag + ">");
}

function createHost(width, opts) {
  const o = opts || {};
  const doc = {
    width,
    widthOf(el) {
      if (el.getAttribute("data-slot") === "matrix" && this.width >= 640) return Math.floor((this.width - 14) / 2);
      return this.width;
    },
  };
  const documentRoot = new El("div", doc);
  let nextRaf = 1;
  const rafQueue = new Map();
  const observers = [];
  const sandbox = {
    Math, Number, String, Array, Object, Map, Set, Buffer, JSON, console,
    getComputedStyle: () => ({ borderRadius: o.radius === undefined ? "6px" : o.radius }),
    matchMedia: (q) => ({ matches: !!o.reduced && q.includes("reduce") }),
    requestAnimationFrame: (fn) => { rafQueue.set(nextRaf, fn); return nextRaf++; },
    cancelAnimationFrame: (id) => { rafQueue.delete(id); },
    ResizeObserver: class {
      constructor(fn) { this.fn = fn; this.connected = false; observers.push(this); }
      observe() { this.connected = true; }
      disconnect() { this.connected = false; }
    },
    document: { getElementById: (id) => documentRoot.querySelector("#" + id) },
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  let now = 0;
  return {
    doc, documentRoot, sandbox, rafQueue, observers,
    drive(maxFrames) {
      let frames = 0;
      while (rafQueue.size && frames < maxFrames) {
        const [id, fn] = rafQueue.entries().next().value;
        rafQueue.delete(id);
        now += 16;
        fn(now);
        frames++;
      }
      return frames;
    },
  };
}
function mountHtml(width, opts) {
  const host = createHost(width, opts);
  const match = /^([\s\S]*)<script>([\s\S]*)<\/script>\s*$/.exec(html);
  assert.ok(match, "fragment shape");
  host.documentRoot.innerHTML = match[1];
  vm.runInContext(match[2], host.sandbox);
  return Object.assign(host, { root: host.documentRoot.querySelector("#" + ROOT_ID), unmount: null });
}
function mountCanvas(width, opts) {
  const host = createHost(width, opts);
  const source = canvas
    .replace(/^import React.*$/m, "")
    .replace(/^export default function App/m, "globalThis.App = function App");
  const cleanups = [];
  const effects = [];
  const refs = [];
  host.sandbox.React = {
    createElement: (tag, props) => ({ tag, props }),
  };
  host.sandbox.useRef = (v) => { const r = { current: v }; refs.push(r); return r; };
  host.sandbox.useEffect = (fn) => { effects.push(fn); };
  host.sandbox.globalThis = host.sandbox;
  vm.runInContext(source, host.sandbox);
  const element = host.sandbox.App();
  assert.strictEqual(element.tag, "section");
  const root = new El("section", host.doc);
  root.attrs.id = element.props.id;
  root.innerHTML = element.props.dangerouslySetInnerHTML.__html;
  element.props.ref.current = root;
  // React StrictMode runs effect, cleanup, effect.
  effects.forEach((fn) => { const c = fn(); assert.strictEqual(typeof c, "function"); c(); });
  effects.forEach((fn) => cleanups.push(fn()));
  return Object.assign(host, { root, unmount: () => cleanups.forEach((c) => c()) });
}

const q = (host, selector) => { const el = host.root.querySelector(selector); assert.ok(el, "missing " + selector); return el; };
const svgs = (host) => host.root.querySelectorAll("svg");
const svgText = (svg) => svg.querySelectorAll("text").map((t) => t.textContent).join("|");
const key = (host, k) => q(host, '[data-act="step"]').dispatch({ type: "keydown", key: k });
function estimateWidth(s) { let w = 0; for (const ch of s) w += ch.charCodeAt(0) > 0x2e7f ? 12 : 6.8; return w; }

// Every label stays inside its SVG and clear of the others, using generous width estimates.
function auditGeometry(host, label) {
  svgs(host).forEach((svg) => {
    const [, , vw, vh] = svg.getAttribute("viewBox").split(" ").map(Number);
    assert.ok(Number(svg.getAttribute("width")) <= host.doc.width, label + " svg wider than its host");
    assert.ok(vw <= host.doc.width, label);
    const boxes = svg.querySelectorAll("text").filter((t) => t.textContent).map((t) => {
      const w = estimateWidth(t.textContent), x = Number(t.getAttribute("x")), y = Number(t.getAttribute("y"));
      const anchor = t.getAttribute("text-anchor");
      const x0 = anchor === "end" ? x - w : anchor === "middle" ? x - w / 2 : x;
      return { s: t.textContent, x0, x1: x0 + w, y0: y - 10, y1: y + 2 };
    });
    boxes.forEach((b) => {
      assert.ok(b.x0 >= -0.5 && b.x1 <= vw + 0.5, label + ": '" + b.s + "' leaves the svg " + b.x0 + ".." + b.x1 + " of " + vw);
      assert.ok(b.y0 >= -1 && b.y1 <= vh + 1, label + ": '" + b.s + "' vertical");
    });
    boxes.forEach((a, i) => boxes.slice(i + 1).forEach((b) => {
      const overlap = a.x0 < b.x1 - 0.5 && b.x0 < a.x1 - 0.5 && a.y0 < b.y1 - 0.5 && b.y0 < a.y1 - 0.5;
      assert.ok(!overlap, label + ": '" + a.s + "' overlaps '" + b.s + "'");
    }));
  });
}
const WIDTHS = [320, 360, 736];

check("historical H100 phases preserve matched baselines, signs and actual split counts", () => {
  assert.strictEqual(measured.job_id, "802632");
  assert.strictEqual(measured.repeats, 10);
  assert.strictEqual(measured.cases.length, 12);
  let count = 0;
  for (const c of measured.cases) {
    assert.strictEqual(c.includes_router, false);
    assert.strictEqual(c.selection_mode, "frozen");
    const baseline = c.arms.find((a) => a.arm === "tk");
    assert.strictEqual(c.arms.length, 3);
    for (const a of c.arms) for (const key of measured.phases) {
      const oldMs = baseline.median_ms[key], newMs = a.median_ms[key];
      assert.ok(Number.isFinite(oldMs) && oldMs > 0 && Number.isFinite(newMs) && newMs > 0);
      assert.ok(Math.abs(a.latency_reduction_pct_vs_tk[key] - 100 * (1 - newMs / oldMs)) < 1e-10);
      assert.ok(Math.abs(a.speedup_vs_tk[key] - oldMs / newMs) < 1e-10);
      count++;
    }
    if (c.tokens === 220000) assert.strictEqual(c.L, 220032);
    if (c.shape_role === "diagnostic") assert.strictEqual(c.H, 2);
  }
  const kv = measured.cases.find((c) => c.id === "short/32768/hotkv").arms.find((a) => a.arm === "tk_auto_c64");
  assert.ok(kv.latency_reduction_pct_vs_tk.fwd < 0 && kv.latency_reduction_pct_vs_tk.bwd > 33);
  const q = measured.cases.find((c) => c.id === "short/32768/hotq_extreme").arms.find((a) => a.arm === "tk_auto_c64");
  assert.strictEqual(q.actual_plan.fwd.split_rows, 0);
  assert.ok(q.latency_reduction_pct_vs_tk.fwd_bwd < -4);
  return count + " finite phase medians and ratios";
});

for (const [artifact, mountOne] of [["HTML fragment", mountHtml], ["Canvas component", mountCanvas]]) {
  check(artifact + ": measured FWD and BWD both render all cases with correct fast/slow signs", () => {
    const host = mountOne(320);
    const cases = q(host, '[data-slot="measured"]').querySelectorAll("[data-measured-case]");
    assert.strictEqual(cases.length, 28);
    for (const el of cases) {
      const c = measured.cases.find((c) => c.id === el.getAttribute("data-measured-case"));
      const arm = c.arms.find((a) => a.arm === el.getAttribute("data-measured-arm"));
      const baseline = c.arms.find((a) => a.arm === "tk");
      for (const key of measured.phases) {
        const text = el.querySelector('[data-measured-phase="' + key + '"]').textContent;
        assert.ok(text.includes(baseline.median_ms[key].toFixed(3)) && text.includes(arm.median_ms[key].toFixed(3)), text);
        const pct = 100 * (1 - arm.median_ms[key] / baseline.median_ms[key]);
        assert.ok(text.includes((pct >= 0 ? "快 " : "慢 ") + Math.abs(pct).toFixed(1) + "%"), text);
      }
      if (c.shape_role === "diagnostic") assert.ok(el.textContent.includes("诊断 H2/K4，非模型配置"));
    }
    if (host.unmount) host.unmount();
    return "84 visible or expandable phase comparisons";
  });
  check(artifact + ": every scene and direction renders the right router, totals and equivalence", () => {
    let frames = 0;
    for (const width of WIDTHS) {
      const host = mountOne(width);
      for (const id of SCENE_IDS) for (const dir of DIRS) for (const follow of ["orig", "bal"]) {
        q(host, '[data-scene="' + id + '"]').click();
        q(host, '[data-dir="' + dir + '"]').click();
        q(host, '[data-follow="' + follow + '"]').click();
        const label = artifact + " " + width + " " + id + "/" + dir + "/" + follow;
        assert.strictEqual(q(host, '[data-scene="' + id + '"]').getAttribute("aria-pressed"), "true", label);
        assert.strictEqual(q(host, '[data-dir="' + dir + '"]').getAttribute("aria-pressed"), "true", label);
        const v = model.sceneView(id), tl = v.dirs[dir];
        assert.strictEqual(q(host, '[data-slot="router"]').querySelector('[data-on="true"]').textContent, v.route.name, label);
        const [matrix, timeline] = svgs(host);
        const titles = svgText(timeline);
        assert.ok(titles.includes("完工 " + tl.orig.makespan), label + " " + titles);
        assert.ok(titles.includes("完工 " + tl.bal.makespan), label + " " + titles);
        assert.ok(titles.includes("教学时间单位"), label);
        const split = dir === "F" ? v.route.F : v.route.B;
        assert.strictEqual(/强制(反例| chunk 反例)/.test(titles), !split, label + " lower timeline name: " + titles);
        const live = q(host, '[data-slot="live"]').textContent;
        assert.strictEqual(live.includes("强制 chunk 反例"), !split, label + " " + live);
        const routerNote = q(host, '[data-note="router"]').textContent;
        ["玩具规则", "未计 router/统计开销", "per-head", "全局工作量", "heavy-row", "规划/launch 开销"].forEach((s) => assert.ok(routerNote.includes(s), label + " " + s));
        assert.ok(q(host, '[data-slot="router"]').textContent.includes("玩具 F+B"), label);
        const bnote = q(host, '[data-slot="bnote"]');
        const hasSigma = dir === "B" && tl.bal.splitLists.length > 0;
        assert.strictEqual(bnote.textContent.includes("没有单独的 dK/dV 合并 kernel"), hasSigma, label);
        assert.strictEqual(bnote.hidden, !hasSigma, label);
        assert.strictEqual(q(host, '[data-slot="pipe"]').textContent.includes("Σ 累加（教学）"), dir === "B" && split, label);
        assert.strictEqual(matrix.querySelectorAll("[data-cell]").length, 16, label);
        const fills = timeline.querySelectorAll("[data-fill]");
        assert.strictEqual(fills.length, tl.orig.tasks.length + tl.bal.tasks.length, label);
        fills.forEach((f) => assert.strictEqual(f.getAttribute("width"), "0", label));
        assert.ok(q(host, '[data-slot="pipe"]').textContent.includes("O") || dir === "B", label);
        const math = q(host, '[data-slot="math"]').textContent;
        assert.ok(math.includes(dir === "F" ? "输出等价" : "梯度等价"), label + " " + math);
        const reported = /max\|Δ\| = ([0-9e−-]+)/.exec(math);
        assert.ok(reported && Number(reported[1].replace("−", "-")) < 1e-12, label + " reports " + (reported && reported[1]));
        auditGeometry(host, label + " start");
        // Walk the scheduling events; marks must advance, and the labels must still fit at the end.
        let guard = 0, last = -1;
        const tTextOf = () => Number(/t = ([0-9.]+)/.exec(q(host, '[data-out="time"]').textContent)[1]);
        while (tTextOf() < Math.max(tl.orig.makespan, tl.bal.makespan) && guard++ < 60) {
          q(host, '[data-act="step"]').click();
          assert.ok(tTextOf() > last, label + " step did not advance");
          last = tTextOf();
          frames++;
          auditGeometry(host, label + " t=" + last);
        }
        assert.ok(guard < 60, label);
        const lastMatrix = svgs(host)[0];
        const ticks = lastMatrix.querySelectorAll("[data-cell-text]").map((t) => t.textContent);
        assert.ok(ticks.every((t) => t === "✓"), label + " cells at the end: " + ticks.join(""));
        const widths = timeline.querySelectorAll("[data-fill]").map((f) => Number(f.getAttribute("width")));
        widths.forEach((w) => assert.ok(w > 0, label));
        assert.strictEqual(q(host, '[data-act="play"]').textContent, "重播", label);
        q(host, '[data-slot="math"]').querySelectorAll("[data-reveal]").forEach((c) => assert.strictEqual(c.getAttribute("data-pending"), "false", label));
      }
      if (host.unmount) host.unmount();
    }
    return frames + " steps across widths " + WIDTHS.join("/");
  });

  check(artifact + ": first render is paused at t = 0, and play ends without looping", () => {
    const host = mountOne(736);
    assert.strictEqual(host.rafQueue.size, 0, "started without the user");
    assert.strictEqual(q(host, '[data-act="play"]').textContent, "播放");
    assert.strictEqual(q(host, '[data-scene="hotQ"]').getAttribute("aria-pressed"), "true");
    q(host, '[data-act="play"]').click();
    assert.strictEqual(q(host, '[data-act="play"]').textContent, "暂停");
    const running = host.drive(40);
    assert.ok(running === 40 && host.rafQueue.size === 1);
    const mid = Number(q(host, '[data-act="seek"]').value);
    assert.ok(mid > 0 && mid < 8, "mid " + mid);
    const midMatrix = svgs(host)[0].querySelectorAll("[data-cell-text]").map((t) => t.textContent);
    assert.ok(midMatrix.some((t) => /^W[1-4]$/.test(t)), "no running pair is labelled mid-play");
    assert.ok(midMatrix.some((t) => t !== "✓"), "finished too early");
    q(host, '[data-act="play"]').click();
    assert.strictEqual(host.rafQueue.size, 0, "pause left a frame queued");
    q(host, '[data-act="play"]').click();
    host.drive(5000);
    assert.strictEqual(host.rafQueue.size, 0, "kept animating after the end");
    assert.strictEqual(q(host, '[data-act="seek"]').value, "8");
    assert.strictEqual(q(host, '[data-act="play"]').textContent, "重播");
    assert.ok(q(host, '[data-slot="live"]').textContent.includes("播放结束"));
    q(host, '[data-act="play"]').click();
    assert.ok(Number(q(host, '[data-act="seek"]').value) < 1, "replay did not rewind");
    if (host.unmount) host.unmount();
  });

  check(artifact + ": keyboard, seeking, resize, reduced motion and cleanup", () => {
    const host = mountOne(736);
    const seek = q(host, '[data-act="seek"]');
    key(host, "ArrowRight");
    assert.strictEqual(seek.value, "0.5");
    key(host, "End");
    assert.strictEqual(seek.value, "8");
    key(host, "ArrowLeft");
    assert.strictEqual(seek.value, "5");
    key(host, "Home");
    assert.strictEqual(seek.value, "0");
    seek.value = "3.25";
    seek.dispatch({ type: "input" });
    assert.ok(q(host, '[data-out="time"]').textContent.includes("t = 3.3") || q(host, '[data-out="time"]').textContent.includes("t = 3.2"));
    const head = q(host, "[data-playhead]").getAttribute("x1");
    assert.ok(Number(head) > 26);
    // A narrower host rebuilds the marks and keeps the time.
    host.doc.width = 360;
    host.observers.forEach((o) => o.fn());
    assert.strictEqual(svgs(host)[1].getAttribute("width"), "360");
    assert.strictEqual(seek.value, "3.25");
    auditGeometry(host, artifact + " after resize");
    // Space toggles from outside a button, and a scene change stops playback.
    seek.dispatch({ type: "keydown", key: " " });
    assert.strictEqual(host.rafQueue.size, 1);
    q(host, '[data-scene="hotKV"]').click();
    assert.strictEqual(host.rafQueue.size, 0);
    assert.strictEqual(seek.value, "0");
    if (host.unmount) {
      q(host, '[data-act="play"]').click();
      assert.strictEqual(host.rafQueue.size, 1);
      host.unmount();
      assert.strictEqual(host.rafQueue.size, 0, "cleanup left a frame queued");
      assert.strictEqual(host.root.listenerCount(), 0, "cleanup left listeners");
      assert.ok(host.observers.every((o) => !o.connected), "cleanup left an observer");
    }
    const calm = mountOne(736, { reduced: true });
    const allowed = new Set(model.sceneView("hotQ").dirs.F.orig.events.concat(model.sceneView("hotQ").dirs.F.bal.events).map(String));
    q(calm, '[data-act="play"]').click();
    for (let i = 0; i < 400 && calm.rafQueue.size; i++) {
      calm.drive(1);
      assert.ok(allowed.has(q(calm, '[data-act="seek"]').value), "reduced motion slid to " + q(calm, '[data-act="seek"]').value);
    }
    assert.strictEqual(q(calm, '[data-act="seek"]').value, "8");
    const bare = mountOne(360, { radius: "0px" });
    assert.ok(bare.root.getAttribute("class").includes("vlb-fallback"));
    const themed = mountOne(360);
    assert.ok(!(themed.root.getAttribute("class") || "").includes("vlb-fallback"));
  });
}

console.log("\n" + results.length + " checks passed (Node " + process.version + "; DOM stand-in, no browser).");
