// NWI Semi-Auto Web 版の画面ロジック。
// デスクトップ版 (main.py / ui_components.py) と同じ操作の流れを再現する:
//   画像選択 (プレビュー) -> 計測開始 (AI 解析) -> ノッチ幅を 2 クリック -> 保存して次へ
(function () {
  'use strict';

  const G = window.NWIGeometry;
  const STORAGE_KEY = 'nwi-semi-auto/session/v1';
  const CSV_HEADERS = [
    'ImageName', 'Timestamp', 'Mode', 'NWI',
    'Femur_Px', 'Notch_Px',
    'P1_x', 'P1_y', 'P2_x', 'P2_y',
    'Femur_L_x', 'Femur_L_y', 'Femur_R_x', 'Femur_R_y',
    'Notch_L_x', 'Notch_L_y', 'Notch_R_x', 'Notch_R_y',
    'Laterality_AI', 'Laterality_Confidence', 'Laterality_Status',
    'Knee_Position',
  ];
  // 両膝の画像は膝ごとに分けて計測する
  const KNEE_NAMES = { screen_left: '画面左', screen_right: '画面右' };
  const LATERALITY_NOTES = {
    multiple_knees: ' — 両膝の可能性。片膝に切り抜いて再解析してください。',
    inconsistent: ' — 両膝の左右の組み合わせが不自然です。画像の L/R 表示を確認してください。',
  };
  const COLORS = {
    contour: 'rgba(0, 255, 0, 0.75)',
    otherContour: 'rgba(0, 255, 0, 0.3)',
    baseline: 'rgb(200, 0, 200)',
    femur: 'rgb(0, 255, 255)',
    notch: 'rgb(255, 140, 0)',
    temp: 'rgb(255, 255, 0)',
  };
  // Lambda のペイロード上限 (6 MB) に収まるよう、大きな画像はブラウザ側で縮小してから送る
  const MAX_UPLOAD_SIDE = 3000;
  const MAX_UPLOAD_BYTES = 4.5 * 1024 * 1024;

  const $ = (id) => document.getElementById(id);
  const el = {
    stage: $('stage'),
    canvas: $('canvas'),
    nwi: $('nwi-value'),
    side: $('side-value'),
    prompt: $('prompt'),
    start: $('btn-start'),
    busy: $('busy'),
    empty: $('empty'),
    list: $('image-list'),
    dataBody: $('data-body'),
    savedCount: $('saved-count'),
    roundLabel: $('round-label'),
    fileInput: $('file-input'),
    welcome: $('welcome'),
    savedDialog: $('saved-dialog'),
  };

  const state = {
    items: [],          // { key, name, label, url, kind: 'sample' | 'upload', blob?, 両膝なら imageKey, position, region, layout }
    current: null,
    img: null,
    phase: 'IDLE',      // IDLE | PREVIEW | MEASURED
    busy: false,        // AI 解析中
    analysis: null,     // { contour, baseline }  AI の結果
    data: null,         // 計測結果
    lineMode: false,    // ノッチ幅を引ける状態か
    temp: null,         // 引いている途中の線 { start, end }
    view: { s: 1, ox: 0, oy: 0 },
    pan: null,
    cache: new Map(),   // 画像ごとの AI 結果 (このページを開いている間だけ)。キーは imageKey()
    session: loadSession(),
  };

  // ---------------- 保存データ (ブラウザの localStorage) ----------------

  function loadSession() {
    try {
      const s = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (s && s.round && s.rounds) return s;
    } catch (e) { /* 保存データなし */ }
    return { round: 1, rounds: { 1: {} } };
  }

  function saveSession() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.session)); } catch (e) { /* 保存不可の環境 */ }
  }

  function results() {
    const rounds = state.session.rounds;
    if (!rounds[state.session.round]) rounds[state.session.round] = {};
    return rounds[state.session.round];
  }

  // ---------------- 表示の更新 ----------------

  function setPrompt(text, kind) {
    if (!text) { el.prompt.hidden = true; return; }
    el.prompt.textContent = text;
    el.prompt.className = `prompt ${kind || 'normal'}`;
    el.prompt.hidden = false;
  }

  function setNWI(value) {
    el.nwi.textContent = typeof value === 'number' ? `NWI: ${value.toFixed(4)}` : `NWI: ${value}`;
  }

  function updateTable(data) {
    const side = data?.Laterality || state.analysis?.laterality;
    const label = document.getElementById('laterality-label');
    const name = side?.side === 'left' ? '左膝' : side?.side === 'right' ? '右膝' : '要確認';
    const where = state.current?.position ? `（${KNEE_NAMES[state.current.position]}）` : '';
    const text = side ? `左右（AI候補）: ${name}${where}` : '左右（AI候補）: 未判定';
    label.textContent = text + (LATERALITY_NOTES[side?.status] || '');
    // 画像の上 (NWI の下) にも出す
    el.side.textContent = text;
    el.side.hidden = !side;
    el.side.classList.toggle('review', Boolean(side) && !side.side);
    el.dataBody.innerHTML = '';
    if (!data) return;
    const rows = [
      ['NWI (比率)', data.NWI.toFixed(4)],
      ['大腿骨幅 (px)', data.Femur_Px.toFixed(2)],
      ['ノッチ幅 (px)', data.Notch_Px.toFixed(2)],
    ];
    for (const [label, value] of rows) {
      const tr = document.createElement('tr');
      const th = document.createElement('td');
      const td = document.createElement('td');
      th.textContent = label;
      td.textContent = value;
      tr.append(th, td);
      el.dataBody.appendChild(tr);
    }
  }

  function renderList() {
    const res = results();
    el.list.innerHTML = '';
    for (const item of state.items) {
      const li = document.createElement('li');
      const done = Boolean(res[item.key]);
      li.textContent = (done ? '✔ ' : '') + item.label;
      if (done) li.classList.add('measured');
      if (state.current === item) li.classList.add('active');
      li.addEventListener('click', () => selectItem(item));
      el.list.appendChild(li);
    }
    el.savedCount.textContent = `${Object.keys(res).length} 件`;
    el.roundLabel.textContent = `ラウンド v${state.session.round}`;
  }

  function setBusy(busy) {
    state.busy = busy;
    el.busy.hidden = !busy;
  }

  // ---------------- 描画 ----------------

  let frame = 0;
  function draw() {
    if (!frame) frame = requestAnimationFrame(render);
  }

  function strokeLine(ctx, a, b, color, width) {
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.stroke();
  }

  function strokeContour(ctx, c, color, width) {
    if (!c || !c.length) return;
    ctx.beginPath();
    ctx.moveTo(c[0][0], c[0][1]);
    for (let i = 1; i < c.length; i++) ctx.lineTo(c[i][0], c[i][1]);
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  function otherKnees() {
    if (!state.analysis || !state.current) return [];
    const knees = state.cache.get(imageKey(state.current))?.knees || [];
    return knees.filter((k) => k.position !== state.analysis.position);
  }

  function render() {
    frame = 0;
    const ctx = el.canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    if (!state.img) return;

    const { s, ox, oy } = state.view;
    ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * ox, dpr * oy);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(state.img, 0, 0);

    // 線の太さは画面上のピクセルで一定にする
    const px = (n) => n / s;

    // 両膝の画像では、計測中でない膝の輪郭も薄く表示する
    for (const k of otherKnees()) strokeContour(ctx, k.contour, COLORS.otherContour, px(2));
    if (state.analysis) strokeContour(ctx, state.analysis.contour, COLORS.contour, px(2));

    const baseline = state.analysis
      ? state.analysis.baseline
      : state.data ? { P1: state.data.P1, P2: state.data.P2 } : null;
    if (baseline) strokeLine(ctx, baseline.P1, baseline.P2, COLORS.baseline, px(3));

    if (state.data) {
      strokeLine(ctx, state.data.Femur_L, state.data.Femur_R, COLORS.femur, px(3));
      strokeLine(ctx, state.data.Notch_L, state.data.Notch_R, COLORS.notch, px(4));
    }

    if (state.temp) {
      ctx.setLineDash([px(6), px(4)]);
      strokeLine(ctx, state.temp.start, state.temp.end, COLORS.temp, px(2));
      ctx.setLineDash([]);
    }
  }

  function resizeCanvas() {
    const r = el.stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    el.canvas.width = Math.max(1, Math.round(r.width * dpr));
    el.canvas.height = Math.max(1, Math.round(r.height * dpr));
    draw();
  }

  function fitView() {
    if (!state.img) return;
    const r = el.stage.getBoundingClientRect();
    const w = state.img.naturalWidth;
    const h = state.img.naturalHeight;
    const s = Math.min(r.width / w, r.height / h);
    state.view = { s, ox: (r.width - w * s) / 2, oy: (r.height - h * s) / 2 };
    draw();
  }

  function zoomAt(x, y, factor) {
    if (!state.img) return;
    const v = state.view;
    const s = Math.min(40, Math.max(0.02, v.s * factor));
    state.view = { s, ox: x - (x - v.ox) * (s / v.s), oy: y - (y - v.oy) * (s / v.s) };
    draw();
  }

  function toImage(e) {
    const r = el.canvas.getBoundingClientRect();
    const v = state.view;
    return [(e.clientX - r.left - v.ox) / v.s, (e.clientY - r.top - v.oy) / v.s];
  }

  function updateCursor() {
    el.canvas.style.cursor = state.pan ? 'grabbing' : state.lineMode ? 'crosshair' : 'default';
  }

  // ---------------- 画像の選択と AI 解析 ----------------

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }

  async function selectItem(item) {
    if (state.busy) return;
    state.current = item;
    state.analysis = null;
    state.data = null;
    state.lineMode = false;
    state.temp = null;
    el.start.hidden = true;
    el.empty.hidden = true;
    updateTable(null);
    renderList();
    updateCursor();

    let img;
    try {
      img = await loadImage(item.url);
    } catch (e) {
      if (state.current === item) setPrompt('画像の読み込みに失敗しました。', 'warning');
      return;
    }
    if (state.current !== item) return;
    state.img = img;
    fitView();

    const saved = results()[item.key];
    if (saved) {
      // 履歴あり: 保存済みの線を表示
      state.data = saved;
      state.phase = 'MEASURED';
      setNWI(saved.NWI);
      updateTable(saved);
      setPrompt('履歴データを表示中。修正可能です。', 'success');
    } else if (item.position) {
      // 両膝の画像を分けた項目は両膝として認識済みなので、プレビューを飛ばしてそのまま計測へ
      state.phase = 'PREVIEW';
      draw();
      startAnalysis();
      return;
    } else {
      // 履歴なし: プレビューのみ。AI はボタンを押してから
      state.phase = 'PREVIEW';
      setNWI(0);
      el.start.hidden = false;
      setPrompt('プレビュー中。計測を開始するにはボタンを押してください。', 'instruction');
    }
    draw();
  }

  async function requestAnalysis(item) {
    let res;
    try {
      if (item.kind === 'sample') {
        res = await fetch(`/api/analyze/sample/${encodeURIComponent(item.name)}`, { method: 'POST' });
      } else {
        const form = new FormData();
        form.append('file', item.blob, item.name);
        res = await fetch('/api/analyze', { method: 'POST', body: form });
      }
    } catch (e) {
      throw new Error('サーバに接続できませんでした。');
    }
    let body = null;
    try { body = await res.json(); } catch (e) { /* JSON 以外の応答 */ }
    if (!res.ok) throw new Error((body && body.detail) || `サーバエラー (${res.status})`);
    return body;
  }

  // AI の結果は画像単位。両膝の画像を分けた項目どうしは同じ結果を共有する
  function imageKey(item) {
    return item.imageKey || item.key;
  }

  // 両膝の画像を、リスト上で膝ごとの項目 (画面左・画面右) に置き換える
  function splitItem(item, knees) {
    const layout = knees.map((k) => ({ position: k.position, region: k.region }));
    const parts = layout.map((k) => Object.assign({}, item, {
      key: `${item.key}#${k.position}`,
      label: `${item.label}（${KNEE_NAMES[k.position]}）`,
      imageKey: item.key,
      position: k.position,
      region: k.region,
      layout,
    }));
    state.items.splice(state.items.indexOf(item), 1, ...parts);
    return parts;
  }

  // 両膝として保存済みの画像は、再読み込み後も膝ごとの項目で表示する
  function restoreSplits() {
    const res = results();
    for (const item of state.items.slice()) {
      if (item.position) continue;
      const saved = Object.keys(KNEE_NAMES).map((p) => res[`${item.key}#${p}`]).find((d) => d && d.Knee_Layout);
      if (saved) splitItem(item, saved.Knee_Layout);
    }
  }

  // AI 解析を実行して、ノッチ幅を引ける状態にする。成功したら true
  async function runAI() {
    let item = state.current;
    let response = state.cache.get(imageKey(item));
    if (!response) {
      setPrompt('AI解析を実行中...', 'normal');
      setBusy(true);
      try {
        response = await requestAnalysis(item);
        state.cache.set(imageKey(item), response);
      } catch (e) {
        state.lineMode = false;
        updateTable(null);
        setPrompt(e.message, 'warning');
        return false;
      } finally {
        setBusy(false);
      }
    }
    const knees = response.knees || [response];
    let note = '';
    if (knees.length > 1 && !item.position) {
      item = splitItem(item, knees)[0];
      state.current = item;
      note = '両膝を検出しました（薄い緑がもう一方の膝）。';
      renderList();
    }
    const analysis = item.position ? knees.find((k) => k.position === item.position) : knees[0];
    if (!analysis) {
      state.lineMode = false;
      setPrompt('この膝の AI 結果が見つかりません。画像を選び直してください。', 'warning');
      return false;
    }
    state.analysis = analysis;
    state.data = null;
    updateTable(null);
    state.lineMode = true;
    setNWI(0);
    const where = item.position ? `（${KNEE_NAMES[item.position]}の膝）` : '';
    setPrompt(`${note}AI解析完了${where}。ノッチ幅（橙色）を引いてください。`, 'success');
    updateCursor();
    draw();
    return true;
  }

  async function startAnalysis() {
    if (state.phase !== 'PREVIEW' || state.busy || !state.current) return;
    el.start.hidden = true;
    const ok = await runAI();
    // 失敗してもプレビュー状態のまま (別の画像を選ぶか、再読み込み)
    state.phase = ok ? 'MEASURED' : 'PREVIEW';
  }

  function finishNotch(start, end) {
    const data = G.measureNotch(state.analysis.baseline, state.analysis.contour, start, end);
    data.Laterality = state.analysis.laterality;
    state.data = data;
    state.lineMode = false;
    setNWI(data.NWI);
    updateTable(data);
    setPrompt('計算完了。保存してください。', 'success');
    updateCursor();
    draw();
  }

  // 修正 (Reset): ノッチ幅を引き直す
  function resetMeasurement() {
    if (!state.current || state.busy) return;
    state.temp = null;
    const hasBaseline = state.analysis || state.data;
    if (!hasBaseline) {
      setPrompt('AI基線データがありません。画像を再読み込みしてください。', 'warning');
      return;
    }
    state.data = null;
    updateTable(null);
    if (!state.analysis) {
      // 履歴表示中は輪郭がないので、AI 解析をやり直す
      setPrompt('修正のため、AI解析を再実行しています...', 'normal');
      runAI();
      return;
    }
    state.lineMode = true;
    setNWI('--');
    setPrompt('修正モード：ノッチ幅（橙色）を引き直してください。', 'instruction');
    updateCursor();
    draw();
  }

  function timestamp() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }

  function saveCurrent() {
    if (!state.current || !state.data || !(state.data.NWI > 0)) {
      setPrompt('計測データがありません。', 'warning');
      return;
    }
    const record = Object.assign({}, state.data, {
      ImageName: state.current.name,
      Timestamp: timestamp(),
      Mode: 'AI_SEMI',
      Knee_Position: state.current.position || 'single',
      Knee_Layout: state.current.layout,
    });
    results()[state.current.key] = record;
    state.data = record;
    saveSession();
    renderList();
    el.savedDialog.returnValue = '';
    el.savedDialog.showModal();
  }

  function loadNext() {
    const i = state.items.indexOf(state.current);
    if (i >= 0 && i < state.items.length - 1) selectItem(state.items[i + 1]);
  }

  function newRound() {
    if (state.busy) return;
    if (!window.confirm('新しい計測ラウンドを開始しますか？')) return;
    state.session.round += 1;
    state.session.rounds[state.session.round] = {};
    saveSession();
    state.current = null;
    state.img = null;
    state.analysis = null;
    state.data = null;
    state.lineMode = false;
    state.temp = null;
    state.phase = 'IDLE';
    el.start.hidden = true;
    el.empty.hidden = false;
    setPrompt(null);
    setNWI('--');
    updateTable(null);
    renderList();
    updateCursor();
    draw();
  }

  function csvCell(value) {
    const s = String(value);
    return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function downloadCSV() {
    const rows = Object.values(results());
    if (!rows.length) {
      setPrompt('保存済みのデータがありません。', 'warning');
      return;
    }
    const lines = [CSV_HEADERS.join(',')];
    for (const d of rows) {
      lines.push([
        csvCell(d.ImageName), d.Timestamp, d.Mode, d.NWI.toFixed(5),
        d.Femur_Px.toFixed(2), d.Notch_Px.toFixed(2),
        ...d.P1, ...d.P2, ...d.Femur_L, ...d.Femur_R, ...d.Notch_L, ...d.Notch_R,
        d.Laterality?.side || '', d.Laterality?.confidence ?? '', d.Laterality?.status || '',
        d.Knee_Position || 'single',
      ].join(','));
    }
    // Excel で文字化けしないよう BOM 付き UTF-8
    const blob = new Blob(['﻿' + lines.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `nwi_results_v${state.session.round}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ---------------- アップロード ----------------

  async function prepareUpload(file) {
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImage(url);
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(w, h));
      if (scale === 1 && file.size <= MAX_UPLOAD_BYTES) return file;
      const c = document.createElement('canvas');
      c.width = Math.round(w * scale);
      c.height = Math.round(h * scale);
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      return await new Promise((resolve, reject) => {
        c.toBlob((b) => (b ? resolve(b) : reject(new Error('encode failed'))), 'image/jpeg', 0.92);
      });
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function onUpload() {
    const file = el.fileInput.files[0];
    el.fileInput.value = '';
    if (!file || state.busy) return;
    try {
      const blob = await prepareUpload(file);
      const item = {
        key: `upload:${Date.now()}:${file.name}`,
        name: file.name,
        label: `📤 ${file.name}`,
        url: URL.createObjectURL(blob),
        kind: 'upload',
        blob,
      };
      state.items.unshift(item);
      selectItem(item);
    } catch (e) {
      setPrompt('画像を読み込めませんでした。JPEG / PNG を選択してください。', 'warning');
    }
  }

  // ---------------- 入力イベント ----------------

  function bindEvents() {
    const c = el.canvas;

    c.addEventListener('pointerdown', (e) => {
      if (!state.img) return;
      if (e.button === 1 || e.button === 2 || (e.button === 0 && e.altKey)) {
        state.pan = { id: e.pointerId, x: e.clientX, y: e.clientY };
        c.setPointerCapture(e.pointerId);
        updateCursor();
        return;
      }
      if (e.button !== 0 || !state.lineMode || !state.analysis) return;
      const p = toImage(e);
      if (!state.temp) {
        state.temp = { start: p, end: p };
        setPrompt('終点をクリックしてください（基線と平行に固定されます）。', 'instruction');
      } else {
        const start = state.temp.start;
        const end = G.projectOnBaseline(start, p, state.analysis.baseline);
        state.temp = null;
        finishNotch(start, end);
      }
      draw();
    });

    c.addEventListener('pointermove', (e) => {
      if (state.pan && e.pointerId === state.pan.id) {
        state.view.ox += e.clientX - state.pan.x;
        state.view.oy += e.clientY - state.pan.y;
        state.pan.x = e.clientX;
        state.pan.y = e.clientY;
        draw();
        return;
      }
      if (state.temp) {
        state.temp.end = G.projectOnBaseline(state.temp.start, toImage(e), state.analysis.baseline);
        draw();
      }
    });

    const endPan = (e) => {
      if (state.pan && e.pointerId === state.pan.id) {
        state.pan = null;
        updateCursor();
      }
    };
    c.addEventListener('pointerup', endPan);
    c.addEventListener('pointercancel', endPan);
    c.addEventListener('contextmenu', (e) => e.preventDefault());

    c.addEventListener('wheel', (e) => {
      if (!state.img) return;
      e.preventDefault();
      const r = c.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
    }, { passive: false });

    const zoomCenter = (f) => {
      const r = c.getBoundingClientRect();
      zoomAt(r.width / 2, r.height / 2, f);
    };
    $('btn-zoom-in').addEventListener('click', () => zoomCenter(1.25));
    $('btn-zoom-out').addEventListener('click', () => zoomCenter(0.8));
    $('btn-fit').addEventListener('click', fitView);

    el.start.addEventListener('click', startAnalysis);
    $('btn-save').addEventListener('click', saveCurrent);
    $('btn-reset').addEventListener('click', resetMeasurement);
    $('btn-new-round').addEventListener('click', newRound);
    $('btn-csv').addEventListener('click', downloadCSV);
    $('btn-help').addEventListener('click', () => el.welcome.showModal());
    el.fileInput.addEventListener('change', onUpload);
    el.savedDialog.addEventListener('close', () => {
      if (el.savedDialog.returnValue === 'next') loadNext();
    });

    document.addEventListener('keydown', (e) => {
      if (el.welcome.open || el.savedDialog.open) return;
      if (e.target.closest && e.target.closest('input, textarea, select')) return;
      if (e.key === 'Enter' && state.phase === 'PREVIEW' && !state.busy) {
        e.preventDefault();
        startAnalysis();
      } else if (e.key === ' ' && state.phase === 'MEASURED') {
        e.preventDefault();
        saveCurrent();
      } else if (e.key === 'Escape' && state.temp) {
        state.temp = null;
        setPrompt('AI解析完了。ノッチ幅（橙色）を引いてください。', 'success');
        draw();
      }
    });

    new ResizeObserver(resizeCanvas).observe(el.stage);
  }

  async function init() {
    bindEvents();
    resizeCanvas();
    renderList();
    el.welcome.showModal();
    try {
      const res = await fetch('/api/samples');
      const samples = await res.json();
      for (const s of samples) {
        state.items.push({ key: `sample:${s.name}`, name: s.name, label: s.name, url: s.url, kind: 'sample' });
      }
      restoreSplits();
      renderList();
    } catch (e) {
      setPrompt('サンプル一覧を取得できませんでした。', 'warning');
    }
  }

  init();
})();
