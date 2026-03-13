// app.js — Frontend logic for the Welding Simulation Web App

// ── Environment Routing ───────────────────────────────────────────────────────
// Detect if running locally or in production (Vercel -> Google Cloud Run)
const isCloud = window.location.hostname.includes('vercel.app') || window.location.hostname.includes('run.app');
const isLocal = !isCloud;
const CLOUD_RUN_URL = 'synarcs-welding-sim-915269781586.us-central1.run.app'; // Google Cloud Run backend

const API_BASE = isLocal ? '' : `https://${CLOUD_RUN_URL}`;
const WS_BASE = isLocal ? `ws://${location.host}` : `wss://${CLOUD_RUN_URL}`;

// ── State ─────────────────────────────────────────────────────────────────────
let currentStep = 1;
let pcData = null;   // cached point cloud response
let currentJointType = 'tee';

// ── Joint Definitions ─────────────────────────────────────────────────────────
const JOINT_DEFS = {
  tee: {
    name: 'Tee Joint',
    params: [
      { id: 'bw', label: 'Base Width', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 'bl', label: 'Base Length', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 'bt', label: 'Base Thickness', min: 0.005, max: 0.05, step: 0.001, default: 0.025, unit: 'm' },
      { id: 'sh', label: 'Stem Height', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 'st', label: 'Stem Thickness', min: 0.005, max: 0.05, step: 0.001, default: 0.025, unit: 'm' }
    ]
  },
  butt: {
    name: 'Butt Joint',
    params: [
      { id: 'w', label: 'Plate Width', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 'l', label: 'Plate Length', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 't', label: 'Plate Thickness', min: 0.005, max: 0.05, step: 0.001, default: 0.025, unit: 'm' },
      { id: 'gap', label: 'Root Gap', min: 0.0, max: 0.02, step: 0.001, default: 0.005, unit: 'm' }
    ]
  },
  lap: {
    name: 'Lap Joint',
    params: [
      { id: 'w', label: 'Plate Width', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 'l', label: 'Plate Length', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 't', label: 'Plate Thickness', min: 0.005, max: 0.05, step: 0.001, default: 0.025, unit: 'm' },
      { id: 'overlap', label: 'Overlap', min: 0.01, max: 0.20, step: 0.01, default: 0.05, unit: 'm' }
    ]
  },
  corner: {
    name: 'Corner Joint',
    params: [
      { id: 'w', label: 'Plate Width', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 'l', label: 'Plate Length', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 't', label: 'Plate Thickness', min: 0.005, max: 0.05, step: 0.001, default: 0.025, unit: 'm' },
      { id: 'type', label: 'Corner Type (0=Closed, 1=Open)', min: 0, max: 1, step: 1, default: 0, unit: '' }
    ]
  },
  edge: {
    name: 'Edge Joint',
    params: [
      { id: 'w', label: 'Plate Width', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 'l', label: 'Plate Length', min: 0.05, max: 0.40, step: 0.01, default: 0.15, unit: 'm' },
      { id: 't', label: 'Plate Thickness', min: 0.005, max: 0.05, step: 0.001, default: 0.025, unit: 'm' },
      { id: 'gap', label: 'Gap', min: 0.0, max: 0.02, step: 0.001, default: 0.002, unit: 'm' }
    ]
  }
};

// ── Dynamic UI Generation ─────────────────────────────────────────────────────

function updateJointUI() {
  currentJointType = document.getElementById('joint_type').value;
  const def = JOINT_DEFS[currentJointType];
  const wrapper = document.getElementById('params-wrapper');

  let html = `<div class="card"><h3>${def.name} Parameters</h3>`;

  def.params.forEach(p => {
    html += `
      <div class="param-row">
        <span class="param-label">${p.label}</span>
        <input type="range" id="s_${p.id}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}" oninput="syncParam('${p.id}',this.value)"/>
        <input type="number" id="n_${p.id}" value="${p.default}" step="${p.step}" min="${p.min}" max="${p.max}" onchange="syncParam('${p.id}',this.value)"/>
        <span class="param-unit">${p.unit}</span>
      </div>
    `;
  });

  html += `</div>`;
  wrapper.innerHTML = html;

  updatePreview();
}

function syncParam(id, val) {
  const v = parseFloat(val);
  const s_el = document.getElementById('s_' + id);
  const n_el = document.getElementById('n_' + id);
  if (s_el) s_el.value = v;
  if (n_el) {
    const stepStr = s_el.getAttribute('step');
    const decimals = stepStr.includes('.') ? stepStr.split('.')[1].length : 0;
    n_el.value = decimals > 0 ? v.toFixed(decimals) : v;
  }
  updatePreview();
}

function getParams() {
  const def = JOINT_DEFS[currentJointType];
  let params = { joint_type: currentJointType };

  def.params.forEach(p => {
    params[p.id] = parseFloat(document.getElementById('n_' + p.id).value);
  });

  // Global pose controls
  params.rotation = parseFloat(document.getElementById('n_rotation')?.value ?? 0);
  params.tilt = parseFloat(document.getElementById('n_tilt')?.value ?? 0);
  params.flip = _flipActive;

  // Engine selection
  const engineSelect = document.getElementById('sim_engine');
  if (engineSelect) {
    params.sim_engine = engineSelect.value;
  }

  return params;
}

// Flip toggle handler
let _flipActive = false;
function toggleFlip() {
  _flipActive = !_flipActive;
  const track = document.getElementById('flip-track');
  const thumb = document.getElementById('flip-thumb');
  const lbl = document.getElementById('flip-label');
  if (_flipActive) {
    track.style.background = 'var(--accent)';
    track.dataset.active = 'true';
    thumb.style.left = '25px';
    thumb.style.background = '#000';
    lbl.style.color = 'var(--accent)';
    lbl.textContent = 'On';
  } else {
    track.style.background = 'var(--bg3)';
    track.dataset.active = 'false';
    thumb.style.left = '3px';
    thumb.style.background = 'var(--muted)';
    lbl.style.color = 'var(--muted)';
    lbl.textContent = 'Off';
  }
  updatePreview();
}

function updateEngineWarning() {
  const select = document.getElementById('sim_engine');
  const warning = document.getElementById('isaac-warning');
  if (select && warning) {
    if (select.value === 'isaac_sim') {
      // Check if we are running locally (localhost, 127.0.0.1, 0.0.0.0, or a local network IP)
      // If we are NOT on the production Vercel domain or Cloud Run domain, assume local.
      const isCloud = window.location.hostname.includes('vercel.app') || window.location.hostname.includes('run.app');
      if (isCloud) {
        warning.style.display = 'block';
      } else {
        warning.style.display = 'none';
      }
    } else {
      warning.style.display = 'none';
    }
  }
}

// ── Plotly 3D Joint Preview ───────────────────────────────────────────────────
// Builds a Plotly mesh3d trace for a cuboid defined by min/max corners
function _boxTrace(x0, y0, z0, x1, y1, z1, color, opacity = 0.92, name = '') {
  // 8 vertices of the box
  const x = [x0, x1, x1, x0, x0, x1, x1, x0];
  const y = [y0, y0, y1, y1, y0, y0, y1, y1];
  const z = [z0, z0, z0, z0, z1, z1, z1, z1];
  // 12 triangles (2 per face × 6 faces)
  const i = [0, 0, 1, 1, 4, 4, 5, 5, 0, 0, 3, 3];
  const j = [1, 2, 2, 5, 5, 6, 6, 7, 4, 7, 7, 6];
  const k = [2, 3, 6, 3, 6, 7, 2, 4, 7, 3, 6, 2];
  // Proper tri-face index for a box
  return {
    type: 'mesh3d', x, y, z,
    i: [0, 0, 1, 1, 4, 4, 5, 5, 0, 0, 3, 3],
    j: [1, 2, 2, 5, 5, 6, 6, 7, 4, 7, 7, 6],
    k: [2, 3, 6, 3, 6, 7, 2, 4, 7, 3, 6, 2],
    color, opacity,
    flatshading: true,
    lighting: { ambient: 0.55, diffuse: 0.8, specular: 0.3, roughness: 0.5 },
    lightposition: { x: 300, y: 600, z: 500 },
    name, showlegend: false, hoverinfo: 'name'
  };
}

function _buildPlotlyTraces(p) {
  const traces = [];
  const A = '#5c7cfa'; // plate A — blue-purple
  const B = '#845ef7'; // plate B — purple
  const S = '#ff6b35'; // seam   — orange
  // Coordinate convention matches Isaac Sim:
  //   X = width,  Y = length (horizontal depth),  Z = height (up)
  // _boxTrace(x0,y0,z0, x1,y1,z1)

  if (p.joint_type === 'tee') {
    const bw = p.bw, bl = p.bl, bt = p.bt, sh = p.sh, st = p.st;
    // Base plate — flat on the table (Z is thin = bt)
    traces.push(_boxTrace(0, 0, 0, bw, bl, bt, A, 0.90, 'Base'));
    // Stem — standing vertically at center of base length (Y = bl/2)
    traces.push(_boxTrace(0, bl / 2 - st / 2, bt, bw, bl / 2 + st / 2, bt + sh, B, 0.90, 'Stem'));
    // Weld seam — thin highlight where stem meets base
    traces.push(_boxTrace(0, bl / 2 - st / 2, bt - 0.002, bw, bl / 2 + st / 2, bt + 0.002, S, 1.0, 'Seam'));

  } else if (p.joint_type === 'butt') {
    const w = p.w, t = p.t, l = p.l, g = p.gap;
    // Two plates placed end-to-end with a gap in the Y direction
    traces.push(_boxTrace(0, 0, 0, w, l, t, A, 0.90, 'Plate 1'));
    traces.push(_boxTrace(0, l + g, 0, w, l + g + l, t, B, 0.90, 'Plate 2'));
    // Weld gap — vertical fill between the two plates
    traces.push(_boxTrace(0, l, 0, w, l + g, t, S, 0.80, 'Weld Gap'));

  } else if (p.joint_type === 'lap') {
    const w = p.w, t = p.t, l = p.l, ov = p.overlap;
    // Bottom plate
    traces.push(_boxTrace(0, 0, 0, w, l, t, A, 0.90, 'Bottom Plate'));
    // Top plate shifted in Y (overlap region) and elevated by t in Z
    traces.push(_boxTrace(0, l - ov, t, w, l - ov + l, 2 * t, B, 0.88, 'Top Plate'));
    // Seam at the edge of the overlap
    traces.push(_boxTrace(0, l - ov, t - 0.002, w, l - ov + 0.003, t + 0.002, S, 1.0, 'Seam'));

  } else if (p.joint_type === 'corner') {
    const w = p.w, t = p.t, l = p.l;
    // Horizontal plate — flat on the table
    traces.push(_boxTrace(0, 0, 0, w, l, t, A, 0.90, 'Horiz. Plate'));
    // Vertical plate — standing at the far end of horizontal plate
    // Occupies: same X, Y=[l-t, l], Z=[t, t+l]
    traces.push(_boxTrace(0, l - t, t, w, l, t + l, B, 0.90, 'Vert. Plate'));
    // Seam along the inner edge (Y=l-t, Z=t)
    traces.push(_boxTrace(0, l - t, t - 0.002, w, l, t + 0.002, S, 1.0, 'Seam'));

  } else if (p.joint_type === 'edge') {
    const w = p.w, t = p.t, l = p.l, g = p.gap;
    // Two plates standing upright side-by-side:
    //   X=width, Y=thickness (thin), Z=height (tall)
    traces.push(_boxTrace(0, 0, 0, w, t, l, A, 0.90, 'Plate 1'));
    traces.push(_boxTrace(0, t + g, 0, w, 2 * t + g, l, B, 0.90, 'Plate 2'));
    // Seam along the top shared edge
    traces.push(_boxTrace(0, t, l - 0.003, w, t + g, l, S, 1.0, 'Seam'));
  }

  return traces;
}

let _previewInitialized = false;

// Apply rotation(Z=up), tilt(X=forward), flip(Y=long-axis) transforms.
// All traces rotate together as one rigid body around a shared global centroid.
function _applyPoseToTraces(traces, rotDeg, tiltDeg, flip) {
  const toRad = d => d * Math.PI / 180;
  const rotR = toRad(rotDeg);
  const tiltR = toRad(tiltDeg);
  const flipR = flip ? Math.PI : 0;

  const cosRz = Math.cos(rotR), sinRz = Math.sin(rotR);
  const cosRx = Math.cos(tiltR), sinRx = Math.sin(tiltR);
  const cosRy = Math.cos(flipR), sinRy = Math.sin(flipR);

  // Compute ONE global centroid across ALL trace vertices so the whole
  // assembly rotates as a rigid body around a single pivot point.
  let sumX = 0, sumY = 0, sumZ = 0, count = 0;
  for (const tr of traces) {
    if (!tr.x) continue;
    for (let i = 0; i < tr.x.length; i++) {
      sumX += tr.x[i]; sumY += tr.y[i]; sumZ += tr.z[i]; count++;
    }
  }
  const cx = sumX / count, cy = sumY / count, cz = sumZ / count;

  return traces.map(tr => {
    if (!tr.x) return tr;
    const nx = [], ny = [], nz = [];
    for (let i = 0; i < tr.x.length; i++) {
      let x = tr.x[i] - cx, y = tr.y[i] - cy, z = tr.z[i] - cz;
      // 1. Flip around Y axis (upside-down)
      let xf = x * cosRy + z * sinRy, yf = y, zf = -x * sinRy + z * cosRy;
      // 2. Tilt around X axis (forward lean)
      let xt = xf, yt = yf * cosRx - zf * sinRx, zt = yf * sinRx + zf * cosRx;
      // 3. Rotate around Z axis (spin on table)
      let xr = xt * cosRz - yt * sinRz, yr = xt * sinRz + yt * cosRz, zr = zt;
      nx.push(xr + cx); ny.push(yr + cy); nz.push(zr + cz);
    }
    return { ...tr, x: nx, y: ny, z: nz };
  });
}


function updatePreview() {
  const wrap = document.getElementById('preview-canvas-wrap');
  if (!wrap || typeof Plotly === 'undefined') return;

  const p = getParams();
  const lbl = document.getElementById('preview-joint-label');
  if (lbl) lbl.textContent = JOINT_DEFS[p.joint_type].name;

  let traces = _buildPlotlyTraces(p);
  traces = _applyPoseToTraces(traces, p.rotation || 0, p.tilt || 0, p.flip || false);

  const layout = {
    paper_bgcolor: 'rgba(5,5,15,0)',
    plot_bgcolor: 'rgba(5,5,15,0)',
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      bgcolor: 'rgba(5,5,15,1)',
      aspectmode: 'data',
      xaxis: { showticklabels: false, gridcolor: '#2e2e4a', zerolinecolor: '#2e2e4a', title: '' },
      yaxis: { showticklabels: false, gridcolor: '#2e2e4a', zerolinecolor: '#2e2e4a', title: '' },
      zaxis: { showticklabels: false, gridcolor: '#2e2e4a', zerolinecolor: '#2e2e4a', title: '' },
      camera: { eye: { x: 1.8, y: -1.6, z: 1.4 }, up: { x: 0, y: 0, z: 1 } }
    },
    modebar: { orientation: 'h', color: '#7878a0', bgcolor: 'transparent' }
  };

  const config = {
    displaylogo: false, responsive: true,
    modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'select3d', 'lasso3d', 'tableRotation', 'resetCameraLastSave3d']
  };

  if (!_previewInitialized) {
    Plotly.newPlot(wrap, traces, layout, config);
    _previewInitialized = true;
  } else {
    Plotly.react(wrap, traces, layout, config);
  }
}

// ── Step navigation ───────────────────────────────────────────────────────────
function goStep(n) {
  if (n === 1) resetSession();
  currentStep = n;
  [1, 2, 3, 4, 5].forEach(i => {
    document.getElementById(`step${i}`).classList.toggle('hidden', i !== n);
    document.querySelector(`[data-step="${i}"]`).classList.toggle('active', i === n);
  });
}

function resetSession() {
  // Reset scan page
  document.getElementById('scan-prog').style.width = '0%';
  document.getElementById('scan-count').textContent = 'Position 0 / 5';
  document.getElementById('scan-counter-display').innerHTML = '0<span> / 5</span>';
  const engineName = (document.getElementById('sim_engine')?.value === 'pybullet') ? 'PyBullet' : 'Isaac Sim';
  document.getElementById('scan-status-text').textContent = `Initializing ${engineName}…`;
  document.getElementById('scan-badge').textContent = '● Running';
  document.getElementById('scan-badge').className = 'badge badge-run';
  document.getElementById('scan-log').innerHTML = '';
  document.getElementById('scan-placeholder').style.display = 'flex';
  document.getElementById('scan-video').style.display = 'none';
  document.getElementById('scan-video').src = '';
  document.getElementById('scan-recording-badge').textContent = '● Recording…';
  document.getElementById('scan-recording-badge').className = 'badge badge-run';
  document.getElementById('scan-complete-msg').style.display = 'none';
  const btnProc = document.getElementById('btn-to-process');
  btnProc.disabled = true;
  btnProc.textContent = 'Process Point Clouds →';
  btnProc.classList.remove('btn-primary'); btnProc.classList.add('btn-secondary');
  document.getElementById('scan-images-section').classList.add('hidden');
  document.getElementById('scan-images').innerHTML = '';
  [1, 2, 3, 4, 5].forEach(i => {
    const el = document.getElementById(`pos-${i}`);
    if (el) el.classList.remove('active', 'done');
  });

  // Reset process page
  const btnMerge = document.getElementById('btn-process');
  if (btnMerge) { btnMerge.disabled = false; btnMerge.textContent = '▶ Merge Point Clouds'; }
  const procStatus = document.getElementById('proc-status-text');
  if (procStatus) procStatus.textContent = 'Click below to merge all 5 point cloud scans into one.';
  const progWrap = document.getElementById('proc-progress-wrap');
  if (progWrap) progWrap.style.display = 'none';
  const progBar = document.getElementById('proc-prog');
  if (progBar) progBar.style.width = '0%';
  document.getElementById('proc-badge')?.classList.add('hidden');
  document.getElementById('proc-log').innerHTML = '';
  const ph = document.getElementById('pc-placeholder');
  if (ph) ph.style.display = 'flex';
  const pcs = document.getElementById('pc-section');
  if (pcs) pcs.style.display = 'none';
  document.getElementById('pc-stats-card')?.classList.add('hidden');
  document.getElementById('density-card')?.classList.add('hidden');
  document.getElementById('btn-download')?.classList.add('hidden');
  const btnDetect = document.getElementById('btn-to-detect');
  if (btnDetect) { btnDetect.classList.add('hidden'); btnDetect.disabled = true; btnDetect.textContent = 'Detect Seams →'; btnDetect.classList.remove('btn-primary'); btnDetect.classList.add('btn-secondary'); }

  // Reset detect page
  const btnRunDetect = document.getElementById('btn-detect');
  if (btnRunDetect) { btnRunDetect.disabled = false; btnRunDetect.textContent = '▶ Detect Seams'; }
  const detStatus = document.getElementById('detect-status-text');
  if (detStatus) detStatus.textContent = 'Apply RANSAC plane fitting to detect weld seams.';
  const detProgWrap = document.getElementById('detect-progress-wrap');
  if (detProgWrap) detProgWrap.style.display = 'none';
  const detProgBar = document.getElementById('detect-prog');
  if (detProgBar) detProgBar.style.width = '0%';
  document.getElementById('detect-badge')?.classList.add('hidden');
  const dph = document.getElementById('detect-pc-placeholder');
  if (dph) dph.style.display = 'flex';
  const dpcs = document.getElementById('detect-pc-section');
  if (dpcs) dpcs.style.display = 'none';
  document.getElementById('seam-results-card')?.classList.add('hidden');
  document.getElementById('seam-planes-card')?.classList.add('hidden');
  const btnWeld = document.getElementById('btn-to-weld');
  if (btnWeld) { btnWeld.classList.add('hidden'); btnWeld.disabled = true; btnWeld.textContent = 'Proceed to Weld →'; btnWeld.classList.remove('btn-primary'); btnWeld.classList.add('btn-secondary'); }

  // Reset weld page (formerly step 4)
  document.getElementById('btn-restart')?.classList.add('hidden');
}



// ── Log helpers ───────────────────────────────────────────────────────────────
function appendLog(boxId, line) {
  const box = document.getElementById(boxId);
  const div = document.createElement('div');
  if (line.startsWith('[STEP]')) { div.className = 'log-step'; }
  else if (line.startsWith('[WARN]')) { div.className = 'log-warn'; }
  else if (line.startsWith('[ERROR]')) { div.className = 'log-error'; }
  else if (line.startsWith('[INFO]')) { div.className = 'log-info'; }
  else if (line.startsWith('[EXIT]')) { div.className = 'log-exit'; }
  div.textContent = line;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return line;
}

// ── Step 1 → 2: Save config and prepare to scan ───────────────────────────────
async function saveConfig() {
  const params = getParams();
  const btn = document.getElementById('btn-save-config');
  btn.disabled = true;
  btn.textContent = '⏳ Saving…';

  // Save config
  await fetch(`${API_BASE}/api/configure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  btn.style.display = 'none';
  document.getElementById('config-status').style.display = 'inline-block';
  const nextBtn = document.getElementById('btn-to-scan');
  nextBtn.classList.remove('hidden');
  nextBtn.classList.add('btn-primary');
  nextBtn.classList.remove('btn-secondary');
}

// ── Step 2: Start scan ────────────────────────────────────────────────────────
async function startScan() {
  const btn = document.getElementById('btn-start-scan');
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Scanning…';
  }
  const log = document.getElementById('scan-log');
  // Reset log and video state
  document.getElementById('scan-log').innerHTML = '';
  document.getElementById('scan-video').style.display = 'none';
  document.getElementById('scan-video').src = '';
  document.getElementById('scan-placeholder').style.display = 'flex';
  document.getElementById('scan-recording-badge').className = 'badge badge-run';
  document.getElementById('scan-recording-badge').textContent = '\u25cf Recording\u2026';

  // Connect WebSocket
  const ws = new WebSocket(`${WS_BASE}/ws/scan`);
  let scanCount = 0;
  const TOTAL_POS = 5;

  ws.onopen = () => {
    const p1 = document.getElementById('pos-1');
    if (p1) p1.classList.add('active');
  };
  ws.onmessage = e => {

    const line = e.data;
    appendLog('scan-log', line);

    if (line.includes('SCAN_POSITION_DONE')) {
      scanCount++;
      const pct = (scanCount / TOTAL_POS * 100).toFixed(0);
      document.getElementById('scan-prog').style.width = pct + '%';
      document.getElementById('scan-count').textContent = `Position ${scanCount} / ${TOTAL_POS}`;
      document.getElementById('scan-status-text').textContent = `Captured position ${scanCount} of ${TOTAL_POS}`;
      // Update big counter
      document.getElementById('scan-counter-display').innerHTML = `${scanCount}<span> / ${TOTAL_POS}</span>`;
      // Mark completed position as done, next as active
      for (let i = 1; i <= scanCount; i++) {
        const el = document.getElementById(`pos-${i}`);
        if (el) { el.classList.remove('active'); el.classList.add('done'); }
      }
      if (scanCount < TOTAL_POS) {
        const next = document.getElementById(`pos-${scanCount + 1}`);
        if (next) next.classList.add('active');
      }
    }
    if (line.includes('SCAN_COMPLETE')) {
      document.getElementById('scan-prog').style.width = '100%';
      document.getElementById('scan-badge').textContent = '✓ Complete';
      document.getElementById('scan-badge').className = 'badge badge-ok';
      document.getElementById('scan-status-text').textContent = 'Scan complete!';
      loadScanImages();
      // Enable the process button and show success message
      const btnProc = document.getElementById('btn-to-process');
      btnProc.disabled = false;
      btnProc.classList.add('btn-primary');
      btnProc.classList.remove('btn-secondary');
      btnProc.textContent = '▶ Process Point Clouds →';
      // Reveal success message above button
      const msg = document.getElementById('scan-complete-msg');
      if (msg) msg.style.display = 'flex';

      // Show the recorded video, hide placeholder
      document.getElementById('scan-recording-badge').textContent = '\u2713 Done';
      document.getElementById('scan-recording-badge').className = 'badge badge-ok';
      document.getElementById('scan-placeholder').style.display = 'none';
      const vid = document.getElementById('scan-video');
      vid.style.display = 'block';
      vid.src = `${API_BASE}/api/scan-video?t=${Date.now()}`;
      vid.play().catch(e => console.log('Auto-play prevented', e));
    }
    if (line.startsWith('[EXIT]') && !line.includes('code=0')) {
      document.getElementById('scan-badge').textContent = '✗ Error';
      document.getElementById('scan-badge').className = 'badge';
      document.getElementById('scan-badge').style.background = '#3a1a1a';
      document.getElementById('scan-badge').style.color = 'var(--red)';
    }
  };
  ws.onerror = () => appendLog('scan-log', '[ERROR] WebSocket connection failed');
}

async function loadScanImages() {
  const res = await fetch(`${API_BASE}/api/scan-images`);
  const data = await res.json();
  if (!data.images.length) return;
  const grid = document.getElementById('scan-images');
  grid.innerHTML = '';
  data.images.forEach(img => {
    grid.innerHTML += `<div class="thumb">
      <img src="${img.data}" alt="Scan ${img.index}"/>
      <span>Position ${img.index + 1}</span>
    </div>`;
  });
  document.getElementById('scan-images-section').classList.remove('hidden');
}

// ── Step 3: Process ───────────────────────────────────────────────────────────
async function runProcess() {
  const btn = document.getElementById('btn-process');
  btn.disabled = true;
  btn.textContent = '⏳ Merging…';
  document.getElementById('proc-log').innerHTML = '';
  document.getElementById('proc-status-text').textContent = 'Merging 5 point cloud scans…';

  // Show progress bar
  const progWrap = document.getElementById('proc-progress-wrap');
  const progBar = document.getElementById('proc-prog');
  const progLbl = document.getElementById('proc-progress-label');
  if (progWrap) progWrap.style.display = 'block';

  // Animate indeterminate progress until done
  let fakePct = 0;
  const fakeTimer = setInterval(() => {
    if (fakePct < 85) { fakePct += Math.random() * 4; if (progBar) progBar.style.width = fakePct + '%'; }
  }, 400);

  const ws = new WebSocket(`${WS_BASE}/ws/process`);

  ws.onmessage = e => {
    const line = e.data;
    appendLog('proc-log', line);
    if (line.trim() && !line.startsWith('[') && progLbl)
      progLbl.textContent = line.trim().substring(0, 60);

    if (line.includes('PROCESS_COMPLETE')) {
      clearInterval(fakeTimer);
      if (progBar) progBar.style.width = '100%';
      if (progLbl) progLbl.textContent = 'Done!';
      btn.textContent = '✓ Merged';
      document.getElementById('proc-status-text').textContent = 'All 5 scans merged successfully.';
      document.getElementById('proc-badge').classList.remove('hidden');
      // Hide placeholder, reveal 3D viewer
      const ph = document.getElementById('pc-placeholder');
      if (ph) ph.style.display = 'none';
      const pcs = document.getElementById('pc-section');
      if (pcs) pcs.style.display = 'flex';
      // Show stats + density cards
      document.getElementById('pc-stats-card')?.classList.remove('hidden');
      document.getElementById('density-card')?.classList.remove('hidden');
      // Footer detect button
      const btnDetect = document.getElementById('btn-to-detect');
      if (btnDetect) {
          btnDetect.classList.remove('hidden');
          btnDetect.disabled = false;
          btnDetect.classList.add('btn-primary');
          btnDetect.classList.remove('btn-secondary');
      }
      loadPointCloud(50000);
    }
  };
  ws.onerror = () => {
    clearInterval(fakeTimer);
    appendLog('proc-log', '[ERROR] WebSocket failed');
    btn.disabled = false;
    btn.textContent = '▶ Merge Point Clouds';
    document.getElementById('proc-status-text').textContent = 'Error — please retry.';
  };
}


// ── Point cloud loading ───────────────────────────────────────────────────────
function updateDensityLabel(val) {
  const n = parseInt(val);
  document.getElementById('density-label').textContent =
    n >= 500000 ? 'All points' : n.toLocaleString() + ' points';
}

function reloadPointCloud(val) {
  const max = parseInt(val) >= 500000 ? 0 : parseInt(val);
  loadPointCloud(max);
}

async function loadPointCloud(maxPts) {
  const url = maxPts > 0 ? `${API_BASE}/api/pointcloud?max_points=${maxPts}` : `${API_BASE}/api/pointcloud`;
  const res = await fetch(url);
  if (!res.ok) { appendLog('proc-log', '[ERROR] Could not load point cloud'); return; }
  pcData = await res.json();

  document.getElementById('pc-stats').textContent =
    `Showing ${pcData.shown_points.toLocaleString()} / ${pcData.total_points.toLocaleString()} points`;

  renderPointCloud(pcData);
}

function renderPointCloud(data) {
  const { points } = data;

  const traces = [
    {
      type: 'scatter3d',
      mode: 'markers',
      x: points.x, y: points.y, z: points.z,
      name: 'Point Cloud',
      marker: {
        size: 2.0,
        color: points.colors,
        opacity: 0.95,
      },
    },
  ];

  const layout = {
    paper_bgcolor: '#09090f',
    plot_bgcolor: '#09090f',
    font: { color: '#c0c0d8', family: 'Inter' },
    scene: {
      bgcolor: '#09090f',
      xaxis: { gridcolor: '#1e1e30', zerolinecolor: '#2e2e4a', title: 'X (m)' },
      yaxis: { gridcolor: '#1e1e30', zerolinecolor: '#2e2e4a', title: 'Y (m)' },
      zaxis: { gridcolor: '#1e1e30', zerolinecolor: '#2e2e4a', title: 'Z (m)' },
      camera: { eye: { x: 1.8, y: 1.8, z: 1.2 } },
    },
    margin: { l: 0, r: 0, t: 10, b: 0 },
  };

  Plotly.react('plotly-container', traces, layout, { responsive: true, displaylogo: false });
}

function downloadPointCloud() {
  window.open(`${API_BASE}/api/download-pcd`, '_blank');
}

// ── Step 4: Seam Detection ──────────────────────────────────────────────────
async function startSeamDetection() {
  const btn = document.getElementById('btn-detect');
  btn.disabled = true;
  btn.textContent = '⏳ Detecting…';
  document.getElementById('detect-log').innerHTML = '';
  document.getElementById('detect-status-text').textContent = 'Running RANSAC plane fitting…';

  const progWrap = document.getElementById('detect-progress-wrap');
  const progBar = document.getElementById('detect-prog');
  const progLbl = document.getElementById('detect-progress-label');
  if (progWrap) progWrap.style.display = 'block';

  let fakePct = 0;
  const fakeTimer = setInterval(() => {
    if (fakePct < 85) { fakePct += Math.random() * 8; if (progBar) progBar.style.width = fakePct + '%'; }
  }, 300);

  const ws = new WebSocket(`${WS_BASE}/ws/seam-detect`);

  ws.onmessage = async e => {
    const line = e.data;
    appendLog('detect-log', line);
    if (line.trim() && !line.startsWith('[') && progLbl)
      progLbl.textContent = line.trim().substring(0, 60);

    if (line.includes('SEAM_DETECT_COMPLETE')) {
        clearInterval(fakeTimer);
        if (progBar) progBar.style.width = '100%';
        if (progLbl) progLbl.textContent = 'Done!';
        btn.textContent = '✓ Detected';
        document.getElementById('detect-status-text').textContent = 'Weld seams successfully detected.';
        document.getElementById('detect-badge').classList.remove('hidden');
        
        const ph = document.getElementById('detect-pc-placeholder');
        if (ph) ph.style.display = 'none';
        const pcs = document.getElementById('detect-pc-section');
        if (pcs) pcs.style.display = 'flex';
        
        document.getElementById('seam-results-card')?.classList.remove('hidden');
        document.getElementById('seam-planes-card')?.classList.remove('hidden');
        
        const btnWeld = document.getElementById('btn-to-weld');
        if (btnWeld) {
            btnWeld.disabled = false;
            btnWeld.classList.remove('hidden');
            btnWeld.classList.add('btn-primary');
            btnWeld.classList.remove('btn-secondary');
            btnWeld.textContent = 'Proceed to Weld (Coming Soon) →';
        }
        
        // Load combined point cloud and seams
        await loadSeamResults();
    } else if (line.startsWith('[EXIT]')) {
        btn.disabled = false;
        clearInterval(fakeTimer);
        if (btn.textContent !== '✓ Detected') {
            btn.textContent = '▶ Retry Detection';
            
            const ph = document.getElementById('detect-pc-placeholder');
            if (ph) ph.style.display = 'none';
            const pcs = document.getElementById('detect-pc-section');
            if (pcs) pcs.style.display = 'flex';
            
            document.getElementById('seam-results-card')?.classList.remove('hidden');
            document.getElementById('seam-planes-card')?.classList.remove('hidden');
            
            await loadSeamResults();
        }
    }
  };
  ws.onerror = () => {
    clearInterval(fakeTimer);
    appendLog('detect-log', '[ERROR] WebSocket failed');
    btn.disabled = false;
    btn.textContent = '▶ Detect Seams';
    document.getElementById('detect-status-text').textContent = 'Error — please retry.';
  };
}

async function loadSeamResults() {
    // 1. Fetch Seam Results
    const resSeam = await fetch(`${API_BASE}/api/seam-results`);
    if (!resSeam.ok) { appendLog('detect-log', '[ERROR] Could not load seam results'); return; }
    const seamData = await resSeam.json();
    
    // 2. Fetch Point Cloud (subsampled to 20k points for performance)
    const resPC = await fetch(`${API_BASE}/api/pointcloud?max_points=20000`);
    if (!resPC.ok) { return; }
    const pcData = await resPC.json();
    
    // 3. Update UI Text
    if (seamData.error) {
        document.getElementById('seam-results').innerHTML = `<div style="color:#f87171"><strong>Error:</strong> ${seamData.error}</div>`;
        document.getElementById('seam-planes').innerHTML = '';
        document.getElementById('detect-status-text').textContent = 'Mathematical detection failed.';
    } else {
        if (seamData.seam1) {
            document.getElementById('seam-results').innerHTML = `
                <div style="margin-bottom:8px"><strong>Seam 1:</strong> <span style="font-family:monospace">L=${seamData.seam1.start.map(v=>v.toFixed(3)).join(',')} → R=${seamData.seam1.end.map(v=>v.toFixed(3)).join(',')}</span></div>
                <div><strong>Seam 2:</strong> <span style="font-family:monospace">L=${seamData.seam2.start.map(v=>v.toFixed(3)).join(',')} → R=${seamData.seam2.end.map(v=>v.toFixed(3)).join(',')}</span></div>
            `;
        }
        if (seamData.planes) {
            document.getElementById('seam-planes').innerHTML = `
                <div><strong>Base Plane:</strong> ${seamData.planes.base.inlier_count} pts</div>
                <div><strong>Stem Face 1:</strong> ${seamData.planes.stem1.inlier_count} pts</div>
                <div><strong>Stem Face 2:</strong> ${seamData.planes.stem2.inlier_count} pts</div>
            `;
        }
    }
    
    // 4. Render 3D Plot
    renderDetectVisualization(pcData, seamData);
}

function renderDetectVisualization(pcData, seamData) {
  const { points } = pcData;
  const traces = [
    {
      type: 'scatter3d', mode: 'markers',
      x: points.x, y: points.y, z: points.z,
      name: 'Point Cloud',
      marker: { size: 1.5, color: points.colors, opacity: 0.8 },
    }
  ];

  if (seamData.seam1) {
      // Seam 1
      traces.push({
          type: 'scatter3d', mode: 'lines',
          x: [seamData.seam1.start[0], seamData.seam1.end[0]],
          y: [seamData.seam1.start[1], seamData.seam1.end[1]],
          z: [seamData.seam1.start[2], seamData.seam1.end[2]],
          line: {color: '#f87171', width: 6}, name: 'Seam 1',
      });
      // Seam 1 Left Toolpath
      traces.push({
          type: 'scatter3d', mode: 'lines',
          x: [seamData.seam1.start_left[0], seamData.seam1.end_left[0]],
          y: [seamData.seam1.start_left[1], seamData.seam1.end_left[1]],
          z: [seamData.seam1.start_left[2], seamData.seam1.end_left[2]],
          line: {color: '#4ade80', width: 4, dash: 'dash'}, name: 'S1 Left',
      });
      // Seam 1 Right Toolpath
      traces.push({
          type: 'scatter3d', mode: 'lines',
          x: [seamData.seam1.start_right[0], seamData.seam1.end_right[0]],
          y: [seamData.seam1.start_right[1], seamData.seam1.end_right[1]],
          z: [seamData.seam1.start_right[2], seamData.seam1.end_right[2]],
          line: {color: '#fbbf24', width: 4, dash: 'dash'}, name: 'S1 Right',
      });
      
      // Seam 2
      traces.push({
          type: 'scatter3d', mode: 'lines',
          x: [seamData.seam2.start[0], seamData.seam2.end[0]],
          y: [seamData.seam2.start[1], seamData.seam2.end[1]],
          z: [seamData.seam2.start[2], seamData.seam2.end[2]],
          line: {color: '#60a5fa', width: 6}, name: 'Seam 2',
      });
      // Seam 2 Left Toolpath
      traces.push({
          type: 'scatter3d', mode: 'lines',
          x: [seamData.seam2.start_left[0], seamData.seam2.end_left[0]],
          y: [seamData.seam2.start_left[1], seamData.seam2.end_left[1]],
          z: [seamData.seam2.start_left[2], seamData.seam2.end_left[2]],
          line: {color: '#4ade80', width: 4, dash: 'dash'}, name: 'S2 Left',
      });
      // Seam 2 Right Toolpath
      traces.push({
          type: 'scatter3d', mode: 'lines',
          x: [seamData.seam2.start_right[0], seamData.seam2.end_right[0]],
          y: [seamData.seam2.start_right[1], seamData.seam2.end_right[1]],
          z: [seamData.seam2.start_right[2], seamData.seam2.end_right[2]],
          line: {color: '#fbbf24', width: 4, dash: 'dash'}, name: 'S2 Right',
      });
  }

  const layout = {
    paper_bgcolor: '#09090f', plot_bgcolor: '#09090f',
    font: { color: '#c0c0d8', family: 'Inter' },
    scene: {
      bgcolor: '#09090f',
      xaxis: { gridcolor: '#1e1e30', zerolinecolor: '#2e2e4a', title: 'X (m)' },
      yaxis: { gridcolor: '#1e1e30', zerolinecolor: '#2e2e4a', title: 'Y (m)' },
      zaxis: { gridcolor: '#1e1e30', zerolinecolor: '#2e2e4a', title: 'Z (m)' },
      camera: { eye: { x: 1.8, y: 1.8, z: 1.2 } },
    },
    legend: { bgcolor: 'rgba(24,24,40,0.8)', bordercolor: '#2e2e4a', borderwidth: 1 },
    margin: { l: 0, r: 0, t: 10, b: 0 },
  };

  if(!document.getElementById('detect-plotly-container').hasChildNodes()) {
      Plotly.newPlot('detect-plotly-container', traces, layout, { responsive: true, displaylogo: false });
  } else {
      Plotly.react('detect-plotly-container', traces, layout);
  }
}

// ── Step 5: Weld ──────────────────────────────────────────────────────────────
async function startWeld() {
  document.getElementById('btn-weld').disabled = true;
  document.getElementById('btn-weld').textContent = '⏳ Welding…';
  document.getElementById('weld-log').innerHTML = '';
  document.getElementById('weld-badge').textContent = '● Running';
  document.getElementById('weld-badge').className = 'badge badge-run';
  const weldEngineName = (document.getElementById('sim_engine')?.value === 'pybullet') ? 'PyBullet' : 'Isaac Sim';
  document.getElementById('weld-status-text').textContent = `${weldEngineName} opening…`;

  // Setup Video Box (Show recording state)
  document.getElementById('weld-stream-container').classList.remove('hidden');
  document.getElementById('weld-video').style.display = 'none';
  document.getElementById('weld-video').src = '';
  document.getElementById('weld-recording-badge').classList.remove('hidden');

  const ws = new WebSocket(`${WS_BASE}/ws/weld`);
  let wayCount = 0, wayTotal = 0;

  ws.onmessage = e => {
    const line = e.data;
    appendLog('weld-log', line);

    if (line.includes('WELD_START')) {
      const m = line.match(/total=(\d+)/);
      if (m) wayTotal = parseInt(m[1]);
    }
    if (line.includes('WELD_WAYPOINT_DONE')) {
      wayCount++;
      if (wayTotal > 0) {
        const pct = (wayCount / wayTotal * 100).toFixed(0);
        document.getElementById('weld-prog').style.width = pct + '%';
        document.getElementById('weld-status-text').textContent =
          `Waypoint ${wayCount} / ${wayTotal}`;
      }
    }
    if (line.includes('WELD_COMPLETE')) {
      document.getElementById('weld-prog').style.width = '100%';
      document.getElementById('weld-badge').textContent = '✓ Complete';
      document.getElementById('weld-badge').className = 'badge badge-ok';
      document.getElementById('weld-status-text').textContent = '✅ Welding complete!';
      document.getElementById('btn-weld').textContent = '✓ Done';
      document.getElementById('btn-restart').classList.remove('hidden');

      // Load and show the generated video
      document.getElementById('weld-recording-badge').classList.add('hidden');
      const vid = document.getElementById('weld-video');
      vid.style.display = 'block';
      vid.src = `${API_BASE}/api/weld-video?t=${Date.now()}`;
      vid.play().catch(e => console.log("Auto-play prevented", e));
    }
  };
  ws.onerror = () => appendLog('weld-log', '[ERROR] WebSocket failed');
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  updateJointUI();
});



