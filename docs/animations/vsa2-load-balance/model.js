// Shared toy model: selected-pair masks, worklists, greedy timelines, router rule and the seeded D=4 numerics.
// Times are teaching units with a simplified per-pair cost; nothing here is a measurement.
function createModel() {
  const N = 8, D = 4, WORKERS = 4, CHUNK = 2, PAIR_COST = 1, PLAN_COST = 0.5, MERGE_COST = 0.5, TIME_DOMAIN = 8;
  const SEED = 20261105, EPS = 1e-12;
  const range = (n) => Array.from({ length: n }, (_, i) => i);
  const zeros = (n) => new Array(n).fill(0);
  const matrix = (rows, cols) => range(rows).map(() => zeros(cols));
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const addScaled = (out, x, c) => { for (let d = 0; d < out.length; d++) out[d] += c * x[d]; };
  const sum = (xs) => xs.reduce((s, x) => s + x, 0);

  // --- masks: Q rows x KV columns, 16 selected pairs in every scene
  function transposeMask(mask) {
    const out = range(N).map(() => []);
    mask.forEach((cols, i) => cols.forEach((j) => out[j].push(i)));
    return out;
  }
  const hotQ = [range(N), [0, 1], [2], [3], [4], [5], [6], [7]];
  const MASKS = {
    uniform: range(N).map((i) => [i, (i + 1) % N].sort((a, b) => a - b)),
    hotQ,
    hotKV: transposeMask(hotQ),
    both: [range(N), [0, 1], [0], [0], [0], [0], [0], [0]],
  };
  const SCENES = [
    { id: "uniform", label: "均匀" },
    { id: "hotQ", label: "热 Q" },
    { id: "hotKV", label: "热 KV" },
    { id: "both", label: "都热" },
  ];
  const degrees = (mask) => ({ rows: mask.map((c) => c.length), cols: transposeMask(mask).map((r) => r.length) });
  const edgeCount = (mask) => sum(mask.map((c) => c.length));

  // --- worklists: F owns q rows (q->kv lists), B owns kv columns (kv->q lists)
  function selectedLists(mask, dir) {
    if (dir === "F") return mask.map((cols, i) => ({ list: i, name: "q" + i, edges: cols.map((j) => [i, j]) }));
    return transposeMask(mask).map((rows, j) => ({ list: j, name: "k" + j, edges: rows.map((i) => [i, j]) }));
  }
  const makeTask = (kind, l, edges, part) => ({
    kind,
    list: l.list,
    listName: l.name,
    name: part ? l.name + "#" + part : l.name,
    part: part || 0,
    edges,
    cost: edges.length * PAIR_COST,
    hot: l.edges.length > CHUNK,
  });
  function wholeWorklist(mask, dir) {
    return selectedLists(mask, dir).map((l) => makeTask("whole", l, l.edges));
  }
  // Heavy lists first as contiguous chunks, then one direct task per short list.
  function chunkWorklist(mask, dir) {
    const lists = selectedLists(mask, dir);
    const heavy = lists.filter((l) => l.edges.length > CHUNK).sort((a, b) => b.edges.length - a.edges.length);
    const tasks = [];
    heavy.forEach((l) => {
      for (let e = 0, part = 1; e < l.edges.length; e += CHUNK, part++) {
        tasks.push(makeTask("chunk", l, l.edges.slice(e, e + CHUNK), part));
      }
    });
    lists.filter((l) => l.edges.length <= CHUNK).forEach((l) => tasks.push(makeTask("direct", l, l.edges)));
    return { tasks, split: heavy };
  }

  // --- greedy list scheduling; ties go to the lowest lane
  function placeGreedy(tasks, free, out) {
    tasks.forEach((task) => {
      let lane = 0;
      for (let w = 1; w < free.length; w++) if (free[w] < free[lane]) lane = w;
      const start = free[lane], end = start + task.cost;
      free[lane] = end;
      out.push(Object.assign({}, task, { lane, start, end }));
    });
  }
  function finishTimeline(id, tasks, splitLists) {
    const makespan = Math.max(...tasks.map((t) => t.end));
    const lanes = range(WORKERS).map((w) =>
      tasks.filter((t) => t.lane === w || t.lane === -1).sort((a, b) => a.start - b.start));
    const idleSegments = [];
    lanes.forEach((tl, lane) => {
      let at = 0;
      tl.forEach((t) => { if (t.start > at) idleSegments.push({ lane, start: at, end: t.start }); at = t.end; });
      if (at < makespan) idleSegments.push({ lane, start: at, end: makespan });
    });
    const work = sum(tasks.filter((t) => t.edges.length).map((t) => t.cost));
    const overhead = sum(tasks.filter((t) => !t.edges.length).map((t) => (t.lane === -1 ? WORKERS : 1) * t.cost));
    const idle = sum(idleSegments.map((s) => s.end - s.start));
    const events = [...new Set(tasks.flatMap((t) => [t.start, t.end]))].sort((a, b) => a - b);
    return { id, tasks, splitLists, makespan, work, overhead, idle, idleSegments, events };
  }
  function originalTimeline(mask, dir) {
    const out = [];
    placeGreedy(wholeWorklist(mask, dir), zeros(WORKERS), out);
    return finishTimeline("orig", out, []);
  }
  // Planning is a barrier on every lane; merges or reduce-adds wait for all tasks.
  function balancedTimeline(mask, dir) {
    const { tasks, split } = chunkWorklist(mask, dir);
    const out = [{ kind: "plan", name: "规划", list: -1, part: 0, edges: [], cost: PLAN_COST, hot: false, lane: -1, start: 0, end: PLAN_COST }];
    const free = zeros(WORKERS).map(() => PLAN_COST);
    placeGreedy(tasks, free, out);
    const barrier = Math.max(...free);
    const merges = split.map((l) => ({
      kind: "merge", name: (dir === "F" ? "合并 " : "累加 ") + l.name, list: l.list, listName: l.name, part: 0,
      edges: [], cost: MERGE_COST, hot: true,
    }));
    placeGreedy(merges, free.map(() => barrier), out);
    return finishTimeline("bal", out, split.map((l) => l.list));
  }

  // Per-pair progress inside a timeline: a task walks its pairs in order.
  function edgeStates(timeline, t) {
    const states = new Map();
    timeline.tasks.forEach((task) => {
      task.edges.forEach((e, k) => {
        const s = task.start + k * PAIR_COST;
        const state = t >= s + PAIR_COST ? "done" : t >= s ? "running" : "pending";
        states.set(e[0] + "," + e[1], { state, lane: task.lane });
      });
    });
    return states;
  }
  function busyLanes(timeline, t) {
    const lanes = new Set();
    timeline.tasks.forEach((task) => {
      if (t >= task.start && t < task.end) { if (task.lane === -1) range(WORKERS).forEach((w) => lanes.add(w)); else lanes.add(task.lane); }
    });
    return lanes;
  }

  // --- toy router rule: split a direction when its longest list exceeds one worker's fair share
  function route(mask) {
    const deg = degrees(mask), threshold = edgeCount(mask) / WORKERS;
    const rowMax = Math.max(...deg.rows), colMax = Math.max(...deg.cols);
    const F = rowMax > threshold, B = colMax > threshold;
    return { rowMax, colMax, threshold, F, B, name: F && B ? "both" : F ? "F only" : B ? "B only" : "unsplit" };
  }
  function sceneView(id) {
    const mask = MASKS[id], routed = route(mask);
    const dirs = {};
    ["F", "B"].forEach((dir) => { dirs[dir] = { orig: originalTimeline(mask, dir), bal: balancedTimeline(mask, dir) }; });
    const pick = (dir, split) => (split ? dirs[dir].bal : dirs[dir].orig).makespan;
    return {
      id, mask, route: routed, degrees: degrees(mask), dirs,
      totals: {
        original: pick("F", false) + pick("B", false),
        routed: pick("F", routed.F) + pick("B", routed.B),
        forced: pick("F", true) + pick("B", true),
      },
    };
  }

  // --- seeded D=4 numerics. LSE is log2 of the summed exponentials, as in docs/vsa2_splitkv.md.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function seededTensors(seed, scale) {
    const rand = mulberry32(seed), s = scale === undefined ? 1 : scale;
    const tensor = (c) => range(N).map(() => range(D).map(() => (rand() * 2 - 1) * c));
    return { Q: tensor(1.2 * s), K: tensor(1.2 * s), V: tensor(1), G: tensor(1) };
  }
  function log2SumExp(values) {
    const m = Math.max(...values);
    return (m + Math.log(sum(values.map((v) => Math.exp(v - m))))) / Math.LN2;
  }
  function attend(S, cols, V) {
    const m = Math.max(...S), O = zeros(D);
    let total = 0;
    S.forEach((s, c) => { const w = Math.exp(s - m); total += w; addScaled(O, V[cols[c]], w); });
    return { O: O.map((x) => x / total), LSE: log2SumExp(S) };
  }
  // Independent reference: a dense 8x8 softmax with -Infinity on unselected pairs.
  function fullMaskedSoftmax({ Q, K, V }, mask) {
    return Q.map((q, i) => {
      const S = K.map((k, j) => (mask[i].includes(j) ? dot(q, k) : -Infinity));
      const m = Math.max(...S), w = S.map((s) => Math.exp(s - m)), total = sum(w);
      const O = zeros(D);
      w.forEach((x, j) => addScaled(O, V[j], x / total));
      return { O, LSE: (m + Math.log(total)) / Math.LN2 };
    });
  }
  // q rows longer than `chunk`: partial O + LSE per chunk, then the stable merge at M = max LSE.
  function forwardChunked({ Q, K, V }, mask, chunk) {
    return mask.map((cols, i) => {
      const S = cols.map((j) => dot(Q[i], K[j]));
      if (cols.length <= chunk) return Object.assign(attend(S, cols, V), { parts: null });
      const parts = [];
      for (let c = 0; c < cols.length; c += chunk) {
        const sub = cols.slice(c, c + chunk);
        parts.push(Object.assign(attend(S.slice(c, c + chunk), sub, V), { cols: sub }));
      }
      const M = Math.max(...parts.map((p) => p.LSE));
      const weights = parts.map((p) => Math.pow(2, p.LSE - M));
      const total = sum(weights), O = zeros(D);
      parts.forEach((p, c) => addScaled(O, p.O, weights[c] / total));
      return { O, LSE: M + Math.log2(total), parts, weights, M, total };
    });
  }
  // Loss L = sum_id G_id O_id. Reference gradients walk q rows directly with one softmax per row.
  function gradientReference({ Q, K, V, G }, mask) {
    const dQ = matrix(N, D), dK = matrix(N, D), dV = matrix(N, D);
    mask.forEach((cols, i) => {
      const S = cols.map((j) => dot(Q[i], K[j]));
      const m = Math.max(...S), w = S.map((s) => Math.exp(s - m)), total = sum(w);
      const P = w.map((x) => x / total), O = zeros(D);
      cols.forEach((j, c) => addScaled(O, V[j], P[c]));
      cols.forEach((j, c) => {
        const A = P[c] * (dot(G[i], V[j]) - dot(G[i], O));
        addScaled(dV[j], G[i], P[c]);
        addScaled(dK[j], Q[i], A);
        addScaled(dQ[i], K[j], A);
      });
    });
    return { dQ, dK, dV };
  }
  // KV-owner backward: k2q chunks read the saved global LSE and O, and only add their contributions.
  function backwardChunked(tensors, mask, saved, chunk) {
    const { Q, K, V, G } = tensors;
    const dQ = matrix(N, D), dK = matrix(N, D), dV = matrix(N, D), columns = [];
    transposeMask(mask).forEach((rows, j) => {
      const size = rows.length > chunk ? chunk : Math.max(rows.length, 1), parts = [];
      for (let c = 0; c < rows.length; c += size) {
        const sub = rows.slice(c, c + size), P = [], cdV = zeros(D);
        sub.forEach((i) => {
          const p = Math.exp(dot(Q[i], K[j]) - saved[i].LSE * Math.LN2);
          const A = p * (dot(G[i], V[j]) - dot(G[i], saved[i].O));
          P.push(p);
          addScaled(cdV, G[i], p);
          addScaled(dK[j], Q[i], A);
          addScaled(dQ[i], K[j], A);
        });
        addScaled(dV[j], cdV, 1);
        parts.push({ rows: sub, P, dV: cdV });
      }
      columns.push({ list: j, parts, split: rows.length > chunk });
    });
    return { dQ, dK, dV, columns };
  }
  // The tempting wrong path: normalize each column chunk on its own instead of using the saved softmax.
  function independentChunkDV({ Q, K, G }, rows, j, chunk) {
    const out = zeros(D);
    for (let c = 0; c < rows.length; c += chunk) {
      const sub = rows.slice(c, c + chunk);
      const S = sub.map((i) => dot(Q[i], K[j]));
      const m = Math.max(...S), w = S.map((s) => Math.exp(s - m)), total = sum(w);
      sub.forEach((i, k) => addScaled(out, G[i], w[k] / total));
    }
    return out;
  }
  function maxAbsDiff(a, b) {
    if (typeof a === "number") return Math.abs(a - b);
    return Math.max(...a.map((x, i) => maxAbsDiff(x, b[i])));
  }
  function numericExample(id, opts) {
    const o = opts || {}, chunk = o.chunk || CHUNK, mask = MASKS[id];
    const tensors = seededTensors(o.seed === undefined ? SEED : o.seed, o.scale);
    const ref = fullMaskedSoftmax(tensors, mask), fwd = forwardChunked(tensors, mask, chunk);
    const refGrad = gradientReference(tensors, mask), bwd = backwardChunked(tensors, mask, fwd, chunk);
    const forwardDelta = Math.max(maxAbsDiff(fwd.map((r) => r.O), ref.map((r) => r.O)), maxAbsDiff(fwd.map((r) => r.LSE), ref.map((r) => r.LSE)));
    const backwardDelta = Math.max(maxAbsDiff(bwd.dQ, refGrad.dQ), maxAbsDiff(bwd.dK, refGrad.dK), maxAbsDiff(bwd.dV, refGrad.dV));
    const splitRow = fwd.findIndex((r) => r.parts);
    const splitCol = bwd.columns.findIndex((c) => c.split);
    const unweightedDelta = splitRow < 0 ? null : maxAbsDiff(
      fwd[splitRow].parts.reduce((acc, p) => { addScaled(acc, p.O, 1 / fwd[splitRow].parts.length); return acc; }, zeros(D)),
      ref[splitRow].O);
    const independent = splitCol < 0 ? null : independentChunkDV(tensors, transposeMask(mask)[splitCol], splitCol, chunk);
    return {
      tensors, ref, fwd, refGrad, bwd, forwardDelta, backwardDelta, splitRow, splitCol, unweightedDelta,
      independentDelta: independent ? maxAbsDiff(independent, refGrad.dV[splitCol]) : null,
    };
  }

  return {
    N, D, WORKERS, CHUNK, PAIR_COST, PLAN_COST, MERGE_COST, TIME_DOMAIN, SEED, EPS, SCENES, MASKS,
    range, degrees, edgeCount, transposeMask, selectedLists, wholeWorklist, chunkWorklist,
    originalTimeline, balancedTimeline, edgeStates, busyLanes, route, sceneView,
    seededTensors, fullMaskedSoftmax, forwardChunked, gradientReference, backwardChunked, numericExample, maxAbsDiff,
  };
}
if (typeof module !== "undefined") module.exports = { createModel }; // node-only
