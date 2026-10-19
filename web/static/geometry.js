// 計測の幾何計算。デスクトップ版の algorithms.find_intersections と
// ui_components.ImageCanvas.handle_draw_finish をそのまま移植したもの。
// ブラウザでは window.NWIGeometry、Node (テスト) では module.exports として使う。
(function (root) {
  'use strict';

  function lineSegmentIntersection(p, d, q1, q2) {
    const sx = q2[0] - q1[0];
    const sy = q2[1] - q1[1];
    const dx = q1[0] - p[0];
    const dy = q1[1] - p[1];
    const cross = d[0] * sy - d[1] * sx;
    if (Math.abs(cross) < 1e-9) return null;
    const t = (dx * sy - dy * sx) / cross;
    const u = (dx * d[1] - dy * d[0]) / cross;
    if (u >= -1e-7 && u <= 1 + 1e-7) return [p[0] + t * d[0], p[1] + t * d[1]];
    return null;
  }

  // start から direction 方向に伸ばした半直線と閉じた輪郭の交点を、近い順に返す
  function findIntersections(start, direction, contour) {
    const norm = Math.hypot(direction[0], direction[1]);
    if (!contour || contour.length < 2 || norm < 1e-9) return [];
    const d = [direction[0] / norm, direction[1] / norm];
    const hits = [];
    for (let i = 0; i < contour.length; i++) {
      const hit = lineSegmentIntersection(start, d, contour[i], contour[(i + 1) % contour.length]);
      if (!hit) continue;
      const vx = hit[0] - start[0];
      const vy = hit[1] - start[1];
      const dist = Math.hypot(vx, vy);
      const dot = vx * d[0] + vy * d[1];
      if (dist >= 1 && dot >= -1e-6) hits.push([dist, hit]);
    }
    hits.sort((a, b) => a[0] - b[0]);
    return hits;
  }

  function baselineUnit(baseline) {
    const dx = baseline.P2[0] - baseline.P1[0];
    const dy = baseline.P2[1] - baseline.P1[1];
    const len = Math.hypot(dx, dy);
    return len > 0 ? [dx / len, dy / len] : [1, 0];
  }

  // ノッチ線はベースラインと平行に固定する: 始点から見たカーソル位置をベースライン方向に射影
  function projectOnBaseline(start, cursor, baseline) {
    const u = baselineUnit(baseline);
    const t = (cursor[0] - start[0]) * u[0] + (cursor[1] - start[1]) * u[1];
    return [start[0] + t * u[0], start[1] + t * u[1]];
  }

  // 自分で引いたノッチ線 (n1 -> n2) から、同じ高さの顆部幅と NWI を求める
  function measureNotch(baseline, contour, n1, n2) {
    const u = baselineUnit(baseline);
    const notchPx = Math.hypot(n1[0] - n2[0], n1[1] - n2[1]);
    const mid = [(n1[0] + n2[0]) / 2, (n1[1] + n2[1]) / 2];

    let femurL = n1;
    let femurR = n2;
    let femurPx = notchPx;
    if (contour && contour.length) {
      const pos = findIntersections(mid, u, contour);
      const neg = findIntersections(mid, [-u[0], -u[1]], contour);
      if (pos.length && neg.length) {
        femurL = neg[neg.length - 1][1];
        femurR = pos[pos.length - 1][1];
        femurPx = Math.hypot(femurR[0] - femurL[0], femurR[1] - femurL[1]);
      }
    }

    return {
      NWI: femurPx > 0 ? notchPx / femurPx : 0,
      Femur_Px: femurPx,
      Notch_Px: notchPx,
      P1: baseline.P1.slice(),
      P2: baseline.P2.slice(),
      Femur_L: femurL.slice(),
      Femur_R: femurR.slice(),
      Notch_L: n1.slice(),
      Notch_R: n2.slice(),
    };
  }

  const api = { findIntersections, projectOnBaseline, measureNotch };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.NWIGeometry = api;
})(typeof window !== 'undefined' ? window : globalThis);
