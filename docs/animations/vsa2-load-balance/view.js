// DOM view for one mounted animation. Rebuilds marks on state or width changes; paint() only moves them.
function mount(root) {
  const model = createModel();
  const { N, WORKERS, CHUNK, TIME_DOMAIN } = model;
  const views = {};
  const numerics = {};
  model.SCENES.forEach((s) => { views[s.id] = model.sceneView(s.id); numerics[s.id] = model.numericExample(s.id); });
  const state = { scene: "hotQ", dir: "F", follow: "bal", t: 0, playing: false };
  const slot = (name) => root.querySelector('[data-slot="' + name + '"]');
  const slots = { pipe: slot("pipe"), matrix: slot("matrix"), router: slot("router"), math: slot("math"), timeline: slot("timeline"), bnote: slot("bnote"), live: slot("live"), measured: slot("measured") };
  const playBtn = root.querySelector('[data-act="play"]');
  const seek = root.querySelector('[data-act="seek"]');
  const timeOut = root.querySelector('[data-out="time"]');
  const reduced = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const UNITS_PER_SECOND = 1.4;
  let marks = null, raf = 0, lastFrame = 0, rawT = 0, width = 0;

  const fmt = (x) => (Number.isInteger(x) ? String(x) : x.toFixed(1));
  const num = (x, digits) => {
    const s = Math.abs(x).toFixed(digits === undefined ? 2 : digits);
    return (x < 0 && Number(s) !== 0 ? "−" : "") + s;
  };
  const vec = (xs) => xs.map((x) => num(x)).join(" ");
  const delta = (x) => (x === 0 ? "0" : x.toExponential(0).replace("e-", "e−"));
  const textWidth = (s) => {
    let w = 0;
    for (const ch of String(s)) w += ch.charCodeAt(0) > 0x2e7f ? 12 : 6.8;
    return w;
  };
  const clamp01 = (x) => Math.min(1, Math.max(0, x));
  const scene = () => views[state.scene];
  const timelines = () => scene().dirs[state.dir];
  const routedSplit = () => (state.dir === "F" ? scene().route.F : scene().route.B);
  const chunkName = () => (routedSplit() ? "均衡" : "强制 chunk 反例");
  const endTime = () => Math.max(timelines().orig.makespan, timelines().bal.makespan);
  const allEvents = () => [...new Set([...timelines().orig.events, ...timelines().bal.events])].sort((a, b) => a - b);

  // Historical full-operator measurements, never used to set the toy timeline's timing.
  function renderMeasured() {
    const names = { uniform: "均匀对照", hotq_extreme: "hot-Q · 大量重行", straggler_q: "hot-Q · 稀有长行", hotkv: "hot-KV", hotqkv: "hot-Q + hot-KV" };
    function measuredCase(c, armName) {
      const baseline = c.arms.find((a) => a.arm === "tk");
      const arm = c.arms.find((a) => a.arm === armName);
      const method = armName === "tk_auto_c64" ? "Auto C64" : "Manual C64 双向";
      const diagnostic = c.shape_role === "diagnostic" ? " · 诊断 H2/K4，非模型配置" : "";
      const padded = c.L !== c.tokens ? "，补齐 L=" + c.L.toLocaleString("en-US") : "";
      const plan = "拆分列表 F/B = " + arm.actual_plan.fwd.split_rows + "/" + arm.actual_plan.bwd_kv.split_rows;
      const phases = [["fwd", "FWD · 前向"], ["bwd", "BWD · 反向"], ["fwd_bwd", "F+B · 合计"]].map(([key, label]) => {
        const oldMs = baseline.median_ms[key], newMs = arm.median_ms[key];
        const pct = 100 * (1 - newMs / oldMs);
        const change = (pct >= 0 ? "快 " : "慢 ") + Math.abs(pct).toFixed(1) + "%";
        return '<div data-measured-phase="' + key + '"><dt class="text-small">' + label + '</dt><dd class="tabular-nums"><span class="vlb-measured-times"><span>' + oldMs.toFixed(3) + ' →</span><span>' + newMs.toFixed(3) + ' ms</span></span><span data-measured-change>' + change + "</span></dd></div>";
      }).join("");
      return '<section class="vlb-measured-case" data-measured-case="' + c.id + '" data-measured-arm="' + armName + '"><div>' + names[c.case] + " · L=" + c.tokens.toLocaleString("en-US") + diagnostic + padded + '</div><div class="text-small vlb-small">' + method + " · " + plan + '</div><dl class="vlb-measured-phases">' + phases + "</dl></section>";
    }
    const anchors = ["long/220000/straggler_q", "short/32768/hotq_extreme", "short/32768/hotkv", "long/220000/hotkv"];
    const primary = anchors.map((id) => measuredCase(H100_RESULTS.cases.find((c) => c.id === id), "tk_auto_c64")).join("");
    const all = H100_RESULTS.cases.flatMap((c) => ["tk_split_c64", "tk_auto_c64"].map((a) => measuredCase(c, a))).join("");
    slots.measured.innerHTML = '<div class="vlb-measured-list">' + primary + '</div><details class="vlb-measured-details"><summary>全部 12 个 case：Manual / Auto、长序列及诊断对照</summary><div class="vlb-measured-list">' + all + '</div></details><div class="text-small vlb-small">来源：已有 job ' + H100_RESULTS.job_id + '，同一批实验；每项均与同 case 的 TK 未拆基线比较。以上均为固定选块测试。</div>';
  }

  // --- pipeline
  const SHAPES = { "8xD": [12, 24], "Dx8": [24, 12], "8x8": [24, 24] };
  function token(label, shape, phase) {
    const glyph = shape ? '<i class="vlb-shape" style="width:' + SHAPES[shape][0] + "px;height:" + SHAPES[shape][1] + 'px"></i>' : "";
    const dim = shape ? ' <span class="text-small vlb-dim">' + shape.replace("x", "×") + "</span>" : "";
    return '<span class="vlb-token" data-phase="' + (phase || "") + '">' + glyph + "<span>" + label + dim + "</span></span>";
  }
  const op = (s) => '<span class="vlb-dim" aria-hidden="true">' + s + "</span>";
  function renderPipe() {
    const split = routedSplit(), parts = [];
    if (state.dir === "F") {
      if (split) parts.push(token("规划 chunk", null, "plan"), op("→"));
      parts.push(token("Q", "8xD", "work"), op("·"), token("Kᵀ", "Dx8", "work"), op("→"), token("S", "8x8", "work"), op("→"), token("softmax", null, "work"), op("→"), token("×V", "8xD", "work"), op("→"));
      if (split) parts.push(token("partial O + LSE", null, "work"), op("→"), token("LSE 合并", null, "merge"), op("→"));
      parts.push(token("O", "8xD", "done"));
    } else {
      if (split) parts.push(token("规划 chunk", null, "plan"), op("→"));
      parts.push(token("dO", "8xD", "work"), op("·"), token("保存的全局 P", "8x8", "work"), op("→"));
      if (split) parts.push(token("chunk 贡献", null, "work"), op("→"), token("Σ 累加（教学）", null, "merge"), op("→"));
      parts.push(token("dQ dK dV", "8xD", "done"));
    }
    slots.pipe.innerHTML = parts.join("");
  }

  // --- matrix with row and column degree marginals
  function matrixSvg(available) {
    const left = 28, top = 16, right = 52, base = 34;
    const cell = Math.floor(Math.min((available - left - right) / N, 36));
    const size = cell * N, w = left + size + right, h = top + size + base + 16;
    const v = scene(), mask = v.mask, deg = v.degrees, th = v.route.threshold;
    const out = [];
    const barMax = 30;
    const scale = (d) => (d / N) * barMax;
    const tl = timelines();
    const lists = model.selectedLists(mask, state.dir);
    model.range(N).forEach((i) => {
      out.push('<text x="' + (left - 3) + '" y="' + (top + i * cell + cell / 2 + 4) + '" text-anchor="end" class="vlb-tm">q' + i + "</text>");
      out.push('<text x="' + (left + i * cell + cell / 2) + '" y="12" text-anchor="middle" class="vlb-tm">k' + i + "</text>");
    });
    model.range(N).forEach((i) => model.range(N).forEach((j) => {
      const x = left + j * cell, y = top + i * cell, selected = mask[i].includes(j);
      if (!selected) {
        out.push('<rect x="' + (x + 1) + '" y="' + (y + 1) + '" width="' + (cell - 2) + '" height="' + (cell - 2) + '" rx="2" class="vlb-track" fill-opacity="0.45"/>');
        return;
      }
      const list = state.dir === "F" ? lists[i] : lists[j];
      const hot = list.edges.length > CHUNK;
      out.push('<rect data-cell="' + i + "," + j + '" x="' + (x + 1) + '" y="' + (y + 1) + '" width="' + (cell - 2) + '" height="' + (cell - 2) + '" rx="2" class="' + (hot ? "vlb-hot" : "vlb-work") + '" fill-opacity="0.14" data-tooltip="q' + i + " · k" + j + '"/>');
      out.push('<text data-cell-text="' + i + "," + j + '" x="' + (x + cell / 2) + '" y="' + (y + cell / 2 + 4) + '" text-anchor="middle">·</text>');
    }));
    // Row degrees on the right, column degrees underneath; the active direction carries the color.
    const barX = left + size + 4, barY = top + size + base;
    const rowActive = state.dir === "F", colActive = state.dir === "B";
    const barClass = (d, active) => (!active ? "vlb-neutral" : d > th ? "vlb-hot" : "vlb-work");
    model.range(N).forEach((i) => {
      const d = deg.rows[i], y = top + i * cell;
      out.push('<rect x="' + barX + '" y="' + (y + cell * 0.2) + '" width="' + scale(d) + '" height="' + cell * 0.6 + '" class="' + barClass(d, rowActive) + '" fill-opacity="' + (rowActive ? 0.7 : 0.35) + '"/>');
      out.push('<text x="' + (barX + barMax + 3) + '" y="' + (y + cell / 2 + 4) + '"' + (rowActive ? "" : ' class="vlb-tm"') + ">" + d + "</text>");
      const c = deg.cols[i], cx = left + i * cell;
      out.push('<rect x="' + (cx + cell * 0.2) + '" y="' + (barY - scale(c)) + '" width="' + cell * 0.6 + '" height="' + scale(c) + '" class="' + barClass(c, colActive) + '" fill-opacity="' + (colActive ? 0.7 : 0.35) + '"/>');
      out.push('<text x="' + (cx + cell / 2) + '" y="' + (barY + 13) + '" text-anchor="middle"' + (colActive ? "" : ' class="vlb-tm"') + ">" + c + "</text>");
    });
    const thX = barX + scale(th), thY = barY - scale(th);
    out.push('<line x1="' + thX + '" x2="' + thX + '" y1="' + top + '" y2="' + (top + size) + '" class="vlb-finish"/>');
    out.push('<line x1="' + left + '" x2="' + (left + size) + '" y1="' + thY + '" y2="' + thY + '" class="vlb-finish"/>');
    out.push('<text x="' + barX + '" y="12" class="vlb-tm">行度</text>');
    out.push('<text x="' + (thX + 3) + '" y="' + (top + size + 13) + '" class="vlb-tm">阈 ' + fmt(th) + "</text>");
    out.push('<text x="0" y="' + (barY + 13) + '" class="vlb-tm">列度</text>');
    // Chunk outlines appear only while following the balanced timeline.
    if (state.follow === "bal") {
      tl.bal.tasks.filter((t) => t.kind === "chunk").forEach((t) => {
        const is = t.edges.map((e) => e[0]), js = t.edges.map((e) => e[1]);
        const x0 = Math.min(...js), x1 = Math.max(...js), y0 = Math.min(...is), y1 = Math.max(...is);
        out.push('<rect x="' + (left + x0 * cell + 0.5) + '" y="' + (top + y0 * cell + 0.5) + '" width="' + ((x1 - x0 + 1) * cell - 1) + '" height="' + ((y1 - y0 + 1) * cell - 1) + '" rx="3" class="vlb-outline"/>');
      });
    }
    return '<svg class="vlb-svg" role="img" aria-label="8 行 Q、8 列 KV 的选中对矩阵，行度 ' + deg.rows.join(",") + "，列度 " + deg.cols.join(",") + '" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + " " + h + '">' + out.join("") + "</svg>";
  }

  // --- router
  function renderRouter() {
    const v = scene(), r = v.route, t = v.totals;
    const pills = ["F only", "B only", "both", "unsplit"].map((n) =>
      '<span class="vlb-pill" data-on="' + (n === r.name) + '">' + n + "</span>").join("");
    const line = (axis, max, split, tag) => axis + " max " + max + (split ? " > " : " ≤ ") + fmt(r.threshold) + " → " + (split ? "拆 " : "不拆 ") + tag;
    const forced = t.forced !== t.routed ? "，强制全拆反例 " + fmt(t.forced) : "";
    slots.router.innerHTML =
      '<div class="vlb-route"><div class="vlb-pills"><span>玩具 Router</span>' + pills + "</div>" +
      '<div class="tabular-nums">' + line("行", r.rowMax, r.F, "F") + "；" + line("列", r.colMax, r.B, "B") + "</div>" +
      '<div class="text-small tabular-nums vlb-small">玩具 F+B ' + fmt(t.original) + " → " + fmt(t.routed) + forced + " 教学时间单位</div>" +
      '<div class="text-small vlb-small" data-note="router">玩具规则：max 度 > 16 对 / 4 工人 = 4；此处未计 router/统计开销。真实 auto 看 per-head 相对均值、全局工作量和 heavy-row 预算，全 direct 决策也有 GPU 规划/launch 开销。</div></div>';
  }

  // --- numeric merge and gradient panel
  const pending = (reveal) => ' data-reveal="' + reveal + '"';
  function renderMath() {
    const n = numerics[state.scene], bal = timelines().bal;
    const endOf = (name) => bal.tasks.find((t) => t.name === name).end;
    const cell = (s, reveal, cls) => "<span" + (reveal === undefined ? "" : pending(reveal)) + (cls ? ' class="' + cls + '"' : "") + ">" + s + "</span>";
    let html;
    if (state.dir === "F") {
      const row = n.splitRow;
      const ok = '<div class="tabular-nums">✓ 输出等价 · 与全量 masked softmax 相比 max|Δ| = ' + delta(n.forwardDelta) + "</div>";
      if (row < 0) {
        html = '<div class="text-small vlb-small">没有超过 chunk = ' + CHUNK + " 的行 → 直接路径，无需合并</div>" + ok;
      } else {
        const r = n.fwd[row], mergeEnd = endOf("合并 q" + row);
        const rows = r.parts.map((p, c) => {
          const end = endOf("q" + row + "#" + (c + 1));
          return cell("q" + row + "#" + (c + 1), end) + cell(vec(p.O), end) + cell(num(p.LSE), end) + cell(num(r.weights[c]), end);
        }).join("");
        html = '<div class="vlb-math text-small tabular-nums">' +
          cell("chunk", undefined, "vlb-dim") + cell("partial O (D=4)", undefined, "vlb-dim") + cell("LSE", undefined, "vlb-dim") + cell("w", undefined, "vlb-dim") + rows +
          '<div class="vlb-rule"></div>' +
          cell("合并", mergeEnd) + cell(vec(r.O), mergeEnd) + cell(num(r.LSE), mergeEnd) + cell("Σ " + num(r.total), mergeEnd) +
          cell("全量", mergeEnd) + cell(vec(n.ref[row].O), mergeEnd) + cell(num(n.ref[row].LSE), mergeEnd) + cell("✓", mergeEnd) +
          "</div>" + ok +
          '<div class="text-small vlb-small">✗ 不按 w 加权：max|Δ| = ' + num(n.unweightedDelta) + "</div>";
      }
    } else {
      const col = n.splitCol;
      const ok = '<div class="tabular-nums">✓ 梯度等价 · dQ dK dV 与全局参考相比 max|Δ| = ' + delta(n.backwardDelta) + "</div>";
      if (col < 0) {
        html = '<div class="text-small vlb-small">没有超过 chunk = ' + CHUNK + " 的列 → 直接路径，读保存的全局 P</div>" + ok;
      } else {
        const parts = n.bwd.columns[col].parts, mergeEnd = endOf("累加 k" + col);
        const rows = parts.map((p, c) => {
          const end = endOf("k" + col + "#" + (c + 1));
          return cell("k" + col + "#" + (c + 1), end) + cell(p.P.map((x) => num(x)).join(" "), end) + cell(vec(p.dV), end);
        }).join("");
        html = '<div class="vlb-math vlb-math-b text-small tabular-nums">' +
          cell("chunk", undefined, "vlb-dim") + cell("全局 P", undefined, "vlb-dim") + cell("dV 贡献 (D=4)", undefined, "vlb-dim") + rows +
          '<div class="vlb-rule"></div>' +
          cell("Σ 累加", mergeEnd) + cell("", mergeEnd) + cell(vec(n.bwd.dV[col]), mergeEnd) +
          cell("参考", mergeEnd) + cell("✓", mergeEnd) + cell(vec(n.refGrad.dV[col]), mergeEnd) +
          "</div>" + ok +
          '<div class="text-small vlb-small">✗ 每个 chunk 各做 softmax：max|Δ| = ' + num(n.independentDelta) + "</div>";
      }
    }
    slots.math.innerHTML = '<div class="vlb-route">' + html + "</div>";
  }

  // --- aligned timelines on one axis
  function timelineSvg(w) {
    const mL = 26, mR = 8, laneH = 22, laneGap = 2, titleH = 20, panelGap = 10;
    const lanesH = WORKERS * (laneH + laneGap) - laneGap;
    const unit = (w - mL - mR) / TIME_DOMAIN;
    const x = (t) => mL + t * unit;
    const tls = timelines(), split = routedSplit();
    const out = ['<defs><pattern id="vlb-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2" height="5" class="vlb-neutral"/></pattern></defs>'];
    const change = tls.bal.makespan - tls.orig.makespan;
    const changeText = change === 0 ? "" : "（" + (change > 0 ? "+" : "−") + fmt(Math.abs(change)) + "）";
    const panels = [
      { tl: tls.orig, long: "原始整表 · 完工 " + fmt(tls.orig.makespan) + " · 空闲 " + fmt(tls.orig.idle), short: "原始 完工 " + fmt(tls.orig.makespan) + " 空闲 " + fmt(tls.orig.idle) },
      {
        tl: tls.bal,
        long: (split ? "均衡 chunk" : "强制 chunk 反例（玩具 Router 未选）") + " · 完工 " + fmt(tls.bal.makespan) + changeText + " · 空闲 " + fmt(tls.bal.idle),
        short: (split ? "均衡" : "强制反例") + " 完工 " + fmt(tls.bal.makespan) + changeText + " 空闲 " + fmt(tls.bal.idle),
      },
    ];
    let y = 0;
    const lanesTop = [];
    panels.forEach((p, pi) => {
      const statusMax = textWidth("工作 4/4 · 16/16 对");
      const narrow = textWidth(p.long) + statusMax + 8 > w;
      out.push('<text x="0" y="' + (y + 14) + '">' + (narrow ? p.short : p.long) + "</text>");
      out.push('<text data-status="' + pi + '" data-narrow="' + narrow + '" x="' + (w - 1) + '" y="' + (y + 14) + '" text-anchor="end" class="vlb-tm"></text>');
      const top = y + titleH;
      lanesTop.push(top);
      model.range(WORKERS).forEach((lane) => {
        const ly = top + lane * (laneH + laneGap);
        out.push('<rect x="' + mL + '" y="' + ly + '" width="' + (w - mL - mR) + '" height="' + laneH + '" class="vlb-track" fill-opacity="0.45"/>');
        out.push('<text data-lane-label="' + pi + "," + lane + '" x="' + (mL - 4) + '" y="' + (ly + 15) + '" text-anchor="end">W' + (lane + 1) + "</text>");
      });
      p.tl.idleSegments.forEach((s) => {
        const ly = top + s.lane * (laneH + laneGap), sx = x(s.start), sw = (s.end - s.start) * unit;
        out.push('<rect x="' + sx + '" y="' + ly + '" width="' + sw + '" height="' + laneH + '" fill="url(#vlb-hatch)" fill-opacity="0.3"/>');
        out.push('<rect data-idle="' + pi + '" data-start="' + s.start + '" data-end="' + s.end + '" x="' + sx + '" y="' + ly + '" width="0" height="' + laneH + '" fill="url(#vlb-hatch)" fill-opacity="0.8"/>');
        if (sw >= 34) out.push('<text x="' + (sx + sw / 2) + '" y="' + (ly + 15) + '" text-anchor="middle" class="vlb-tm">空闲</text>');
      });
      p.tl.tasks.forEach((t) => {
        const isPlan = t.lane === -1;
        const ly = isPlan ? top : top + t.lane * (laneH + laneGap);
        const h = isPlan ? lanesH : laneH;
        const tx = x(t.start) + 0.5, tw = (t.end - t.start) * unit - 1;
        const cls = t.kind === "plan" || t.kind === "merge" ? "vlb-over" : t.hot ? "vlb-hot" : "vlb-work";
        const tip = t.name + " · W" + (isPlan ? "1-4" : t.lane + 1) + " · " + fmt(t.start) + "–" + fmt(t.end) + " 单位";
        out.push('<rect x="' + tx + '" y="' + ly + '" width="' + tw + '" height="' + h + '" rx="2" class="' + cls + '" fill-opacity="0.22" data-tooltip="' + tip + '"/>');
        out.push('<rect data-fill="' + pi + '" data-start="' + t.start + '" data-end="' + t.end + '" x="' + tx + '" y="' + ly + '" width="0" height="' + h + '" rx="2" class="' + cls + '" fill-opacity="0.6"/>');
        const cx = tx + tw / 2;
        if (isPlan) {
          if (tw >= 13) ["规", "划"].forEach((ch, k) => out.push('<text x="' + cx + '" y="' + (ly + lanesH / 2 - 2 + k * 14) + '" text-anchor="middle">' + ch + "</text>"));
          return;
        }
        const longLabel = t.kind === "whole" && t.hot ? t.name + " 整表 " + t.edges.length + " 对" : t.name;
        const shortLabel = t.kind === "merge" ? (state.dir === "F" ? "合" : "Σ") : t.name;
        const label = textWidth(longLabel) <= tw - 4 ? longLabel : textWidth(shortLabel) <= tw - 2 ? shortLabel : "";
        if (label) out.push('<text x="' + cx + '" y="' + (ly + 15) + '" text-anchor="middle">' + label + "</text>");
      });
      out.push('<line x1="' + x(p.tl.makespan) + '" x2="' + x(p.tl.makespan) + '" y1="' + (top - 1) + '" y2="' + (top + lanesH + 1) + '" class="vlb-finish"/>');
      y = top + lanesH + panelGap;
    });
    const axisY = y - panelGap + 4;
    out.push('<line x1="' + mL + '" x2="' + (w - mR) + '" y1="' + axisY + '" y2="' + axisY + '" class="vlb-grid"/>');
    const step = unit >= 44 ? 1 : 2;
    for (let t = 0; t <= TIME_DOMAIN; t += step) {
      out.push('<line x1="' + x(t) + '" x2="' + x(t) + '" y1="' + axisY + '" y2="' + (axisY + 3) + '" class="vlb-grid"/>');
      out.push('<text x="' + x(t) + '" y="' + (axisY + 15) + '" text-anchor="' + (t === TIME_DOMAIN ? "end" : "middle") + '" class="vlb-tm">' + t + "</text>");
    }
    out.push('<text x="' + (mL + (w - mL - mR) / 2) + '" y="' + (axisY + 30) + '" text-anchor="middle" class="vlb-tm">教学时间单位（非 ms）</text>');
    out.push('<line data-playhead x1="' + mL + '" x2="' + mL + '" y1="' + lanesTop[0] + '" y2="' + axisY + '" class="vlb-head-line"/>');
    const h = axisY + 36;
    const label = "4 个工人的对齐时间线：原始完工 " + fmt(tls.orig.makespan) + "，" + chunkName() + "完工 " + fmt(tls.bal.makespan) + "，教学时间单位";
    return '<svg class="vlb-svg" role="img" aria-label="' + label + '" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + " " + h + '">' + out.join("") + "</svg>";
  }

  // --- build and paint
  function collectMarks() {
    const readRange = (el) => ({ el, start: Number(el.getAttribute("data-start")), end: Number(el.getAttribute("data-end")), w: 0 });
    const all = (sel) => Array.from(root.querySelectorAll(sel));
    const tl = slots.timeline.querySelector("svg");
    const vb = tl.getAttribute("viewBox").split(" ").map(Number);
    const mL = 26, mR = 8;
    const unit = (vb[2] - mL - mR) / TIME_DOMAIN;
    marks = {
      unit,
      mL,
      playhead: root.querySelector("[data-playhead]"),
      fills: [0, 1].map((p) => all('[data-fill="' + p + '"]').map((el) => {
        const m = readRange(el);
        m.w = (m.end - m.start) * unit - 1;
        return m;
      })),
      idles: [0, 1].map((p) => all('[data-idle="' + p + '"]').map((el) => {
        const m = readRange(el);
        m.w = (m.end - m.start) * unit;
        return m;
      })),
      cells: all("[data-cell]").map((el) => {
        const key = el.getAttribute("data-cell");
        return { el, key, text: root.querySelector('[data-cell-text="' + key + '"]') };
      }),
      status: all("[data-status]"),
      laneLabels: all("[data-lane-label]").map((el) => {
        const parts = el.getAttribute("data-lane-label").split(",").map(Number);
        return { el, panel: parts[0], lane: parts[1] };
      }),
      tokens: all("[data-phase]"),
      reveals: all("[data-reveal]").map((el) => ({ el, at: Number(el.getAttribute("data-reveal")) })),
    };
  }
  function phaseAt(t) {
    const tl = routedSplit() ? timelines().bal : timelines().orig;
    if (t <= 0) return "";
    if (t >= tl.makespan) return "done";
    const active = tl.tasks.find((k) => t >= k.start && t < k.end && k.kind === "plan");
    if (active) return "plan";
    return tl.tasks.some((k) => k.kind === "merge" && t >= k.start && t < k.end) ? "merge" : "work";
  }
  function paint() {
    const t = state.t, tls = [timelines().orig, timelines().bal];
    marks.fills.forEach((list) => list.forEach((m) => m.el.setAttribute("width", clamp01((t - m.start) / (m.end - m.start)) * m.w)));
    marks.idles.forEach((list) => list.forEach((m) => m.el.setAttribute("width", clamp01((t - m.start) / (m.end - m.start)) * m.w)));
    const px = marks.mL + t * marks.unit;
    marks.playhead.setAttribute("x1", px);
    marks.playhead.setAttribute("x2", px);
    const followed = state.follow === "orig" ? tls[0] : tls[1];
    const edges = model.edgeStates(followed, t);
    marks.cells.forEach((c) => {
      const e = edges.get(c.key);
      const running = e.state === "running";
      c.el.setAttribute("fill-opacity", running ? 0.75 : e.state === "done" ? 0.4 : 0.14);
      const cls = c.el.getAttribute("class").replace(" vlb-running", "");
      c.el.setAttribute("class", running ? cls + " vlb-running" : cls);
      c.text.textContent = running ? "W" + (e.lane + 1) : e.state === "done" ? "✓" : "·";
    });
    const busy = tls.map((tl) => model.busyLanes(tl, t));
    marks.status.forEach((el, i) => {
      const done = [...model.edgeStates(tls[i], t).values()].filter((e) => e.state === "done").length;
      const narrow = el.getAttribute("data-narrow") === "true";
      el.textContent = "工作 " + busy[i].size + "/" + WORKERS + (narrow ? "" : " · " + done + "/16 对");
    });
    marks.laneLabels.forEach((l) => l.el.setAttribute("data-idle", !busy[l.panel].has(l.lane)));
    const phase = phaseAt(t);
    marks.tokens.forEach((el) => el.setAttribute("data-on", el.getAttribute("data-phase") === phase));
    marks.reveals.forEach((r) => r.el.setAttribute("data-pending", t < r.at));
    const end = endTime();
    timeOut.textContent = "t = " + t.toFixed(1) + " / " + fmt(end) + " 教学时间单位";
    seek.value = t;
    playBtn.textContent = state.playing ? "暂停" : t >= end ? "重播" : "播放";
  }
  function build() {
    width = Math.max(280, Math.round(slots.timeline.clientWidth || root.clientWidth || 736));
    const matrixWidth = Math.max(280, Math.round(slots.matrix.clientWidth || Math.min(width, 360)));
    slots.matrix.innerHTML = matrixSvg(matrixWidth);
    slots.timeline.innerHTML = timelineSvg(width);
    collectMarks();
    paint();
  }
  function announce(extra) {
    const v = scene(), tls = timelines();
    const label = model.SCENES.find((s) => s.id === state.scene).label;
    slots.live.textContent = label + "，" + (state.dir === "F" ? "前向" : "反向") + "，玩具 Router " + v.route.name + "，原始完工 " + fmt(tls.orig.makespan) + "，" + chunkName() + "完工 " + fmt(tls.bal.makespan) + " 教学时间单位" + (extra || "");
  }
  function rebuildAll() {
    seek.max = endTime();
    root.querySelectorAll("[data-scene]").forEach((b) => b.setAttribute("aria-pressed", b.getAttribute("data-scene") === state.scene));
    root.querySelectorAll("[data-dir]").forEach((b) => b.setAttribute("aria-pressed", b.getAttribute("data-dir") === state.dir));
    root.querySelectorAll("[data-follow]").forEach((b) => b.setAttribute("aria-pressed", b.getAttribute("data-follow") === state.follow));
    renderPipe();
    renderRouter();
    renderMath();
    const reduces = state.dir === "B" && timelines().bal.splitLists.length > 0;
    slots.bnote.hidden = !reduces;
    slots.bnote.textContent = reduces ? "B 的 Σ 是数学汇总的教学简化，接近 generic 的 partial 梯度合并；H100 TK 直接 reduce-add，没有单独的 dK/dV 合并 kernel。" : "";
    build();
  }

  // --- transport
  function stop() {
    state.playing = false;
    if (raf) window.cancelAnimationFrame(raf);
    raf = 0;
  }
  function setTime(t) {
    state.t = Math.min(endTime(), Math.max(0, t));
    rawT = state.t;
    paint();
  }
  function frame(now) {
    const dt = Math.min(0.1, lastFrame ? (now - lastFrame) / 1000 : 0);
    lastFrame = now;
    rawT += dt * UNITS_PER_SECOND;
    const end = endTime();
    if (rawT >= end) {
      state.t = end;
      rawT = end;
      stop();
      paint();
      announce("，播放结束");
      return;
    }
    // Reduced motion: hold each scheduling event instead of sliding.
    state.t = reduced ? allEvents().filter((e) => e <= rawT).pop() : rawT;
    paint();
    raf = window.requestAnimationFrame(frame);
  }
  function play() {
    if (state.t >= endTime()) setTime(0);
    state.playing = true;
    lastFrame = 0;
    rawT = state.t;
    paint();
    raf = window.requestAnimationFrame(frame);
  }
  const toggle = () => { if (state.playing) { stop(); paint(); } else play(); };
  function stepBy(direction) {
    stop();
    const events = allEvents();
    const next = direction > 0 ? events.find((e) => e > state.t + 1e-9) : events.filter((e) => e < state.t - 1e-9).pop();
    setTime(next === undefined ? (direction > 0 ? endTime() : 0) : next);
  }
  function change(patch) {
    stop();
    Object.assign(state, patch, { t: 0 });
    rawT = 0;
    rebuildAll();
    announce();
  }
  function onClick(event) {
    const el = event.target.closest("button");
    if (!el) return;
    const sceneId = el.getAttribute("data-scene"), dir = el.getAttribute("data-dir"), follow = el.getAttribute("data-follow"), act = el.getAttribute("data-act");
    if (sceneId) change({ scene: sceneId });
    else if (dir) change({ dir });
    else if (follow) { state.follow = follow; rebuildAll(); }
    else if (act === "play") toggle();
    else if (act === "step") stepBy(1);
  }
  function onInput(event) {
    if (event.target !== seek) return;
    stop();
    setTime(Number(seek.value));
  }
  function onKey(event) {
    const onRange = event.target === seek, onButton = event.target.tagName === "BUTTON";
    if (event.key === " " && !onButton) { event.preventDefault(); toggle(); }
    else if (event.key === "ArrowRight" && !onRange) { event.preventDefault(); stepBy(1); }
    else if (event.key === "ArrowLeft" && !onRange) { event.preventDefault(); stepBy(-1); }
    else if (event.key === "Home" && !onRange) { event.preventDefault(); stop(); setTime(0); }
    else if (event.key === "End" && !onRange) { event.preventDefault(); stop(); setTime(endTime()); }
  }

  root.addEventListener("click", onClick);
  root.addEventListener("input", onInput);
  root.addEventListener("keydown", onKey);
  const radius = window.getComputedStyle(playBtn).borderRadius;
  if (!radius || radius === "0px") root.classList.add("vlb-fallback");
  let observer = null;
  if (typeof ResizeObserver !== "undefined") {
    observer = new ResizeObserver(() => {
      const next = Math.round(slots.timeline.clientWidth);
      if (next && next !== width) build();
    });
    observer.observe(slots.timeline);
  }
  renderMeasured();
  rebuildAll();
  announce();
  return () => {
    stop();
    root.removeEventListener("click", onClick);
    root.removeEventListener("input", onInput);
    root.removeEventListener("keydown", onKey);
    if (observer) observer.disconnect();
  };
}
