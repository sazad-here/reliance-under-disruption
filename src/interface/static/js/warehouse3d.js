// 3D warehouse view (three.js r169, MIT; vendored under static/vendor/three).
//
// A PICTURE OF THE NUMBERS, NOTHING MORE. Every data-driven object here is driven
// by a value the participant already sees in the number panel -- stock on hand,
// stock on order (TOTAL only), unmet orders, supplier and transport status -- so
// the view adds no cue about the AI's accuracy, Q*, delivery timing or future
// weeks (see the design rules at the top of task.html). There is ONE truck; it
// carries the on-order total and never shows when anything will arrive. The code
// never reads the experimental condition.
//
// Ambient life (a forklift on a fixed loop, a walking worker, a running conveyor
// belt, a slow camera drift) is scenery: it runs on the wall clock only, is the
// same every week and in both conditions, and never reads the data. It is off
// under prefers-reduced-motion and on slow machines (adaptive quality below).
//
// API (window.W3D, set when the module has loaded):
//   W3D.setState(s)                        snap to a state payload from /api/state
//   W3D.playWeek(before, outcome, after)   animate one week's result (~3.4 s), then snap to `after`;
//                                          a newer setState/playWeek cancels it at once
//   W3D.quality / W3D.setQuality(0|1|2)    adaptive-quality readout / manual override (testing)
// If WebGL is unavailable the container is hidden and the page works as before.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/three/OrbitControls.js';
import { RoomEnvironment } from '/static/vendor/three/RoomEnvironment.js';
import { RoundedBoxGeometry } from '/static/vendor/three/RoundedBoxGeometry.js';
import { mergeGeometries } from '/static/vendor/three/BufferGeometryUtils.js';

const UNITS_PER_BOX = 10;
const RACK_ROWS = 3, LEVELS = 3, BAYS = 8, DEPTH = 2;          // 144 rack slots = 1,440 units
const BOX = 0.55;
const RM = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
const reduced = () => RM.matches;

const boxesFor = units => (units > 0 ? Math.max(1, Math.round(units / UNITS_PER_BOX)) : 0);
const ease = t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const easeOut = t => 1 - Math.pow(1 - t, 3);
const easeIn = t => t * t * t;
const clamp01 = x => Math.min(1, Math.max(0, x));
// deterministic per-slot variation (tint, yaw, wrap): depends on the slot index only
const hash = i => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };
// seeded RNG, so every participant sees exactly the same textures
function rng(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Fixed choreography of one week (seconds). The total never depends on the data.
const T = {
  truckIn: 0.9,            // truck drives in through the dock door (only if it was not docked)
  arriveDocked: 0.15,      // arrivals start here when the truck is already at the dock
  arriveWin: 0.7, fly: 0.6,
  outStart: 1.4, outWin: 0.8, lift: 0.45, belt: 0.55, drop: 0.15,
  crates: 2.3,
  truckEnd: 2.75, truckEndDur: 0.6,
  total: 3.4,
};

// ---------------------------------------------------------------- textures
let ANISO = 4;
function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return [c, c.getContext('2d')]; }
function tex(c, color = true) {
  const t = new THREE.CanvasTexture(c);
  if (color) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = ANISO;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}
// scale a geometry's UVs so one texture tile covers `sx` x `sy` metres
function worldUV(geo, w, h, sx, sy) {
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * w / sx, uv.getY(i) * h / sy);
  return geo;
}

function floorMaps() {
  const S = 1024, R = rng(11);
  const [c, g] = canvas(S, S), [cr, gr] = canvas(S, S);
  g.fillStyle = '#6f7479'; g.fillRect(0, 0, S, S);
  gr.fillStyle = '#909090'; gr.fillRect(0, 0, S, S);
  for (let i = 0; i < 160; i++) {                                     // trowel mottling
    const x = R() * S, y = R() * S, r = 40 + R() * 240, dark = R() < 0.5;
    const paint = (ctx, rgb, a) => {
      const grd = ctx.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, `rgba(${rgb},${a})`); grd.addColorStop(1, `rgba(${rgb},0)`);
      ctx.fillStyle = grd; ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
    };
    paint(g, dark ? '38,40,44' : '160,165,170', 0.08);
    paint(gr, dark ? '70,70,70' : '190,190,190', 0.16);                 // polished patches reflect more
  }
  for (let i = 0; i < 26000; i++) {                                   // aggregate speckle
    const v = 70 + R() * 110 | 0;
    g.fillStyle = `rgba(${v},${v},${v + 4},${R() * 0.3})`;
    g.fillRect(R() * S, R() * S, 1 + R() * 1.5, 1 + R() * 1.5);
  }
  for (let i = 0; i < 12; i++) {                                      // faint tyre marks
    const x = R() * S, y = R() * S, dx = (R() - 0.5) * 900, dy = (R() - 0.5) * 900;
    for (const [ctx, st] of [[g, 'rgba(20,20,24,0.05)'], [gr, 'rgba(60,60,60,0.1)']]) {
      ctx.strokeStyle = st; ctx.lineWidth = 12 + R() * 8;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.bezierCurveTo(x + dx * 0.3, y + dy * 0.7, x + dx * 0.7, y + dy * 0.2, x + dx, y + dy); ctx.stroke();
    }
  }
  g.strokeStyle = 'rgba(40,40,40,0.28)'; g.lineWidth = 1;             // hairline cracks
  for (let i = 0; i < 5; i++) {
    let x = R() * S, y = R() * S; g.beginPath(); g.moveTo(x, y);
    for (let k = 0; k < 14; k++) { x += (R() - 0.5) * 40; y += (R() - 0.3) * 30; g.lineTo(x, y); }
    g.stroke();
  }
  // saw-cut slab joints on the tile edge (hides the wrap seam too)
  g.fillStyle = 'rgba(28,30,33,0.85)'; g.fillRect(0, 0, S, 4); g.fillRect(0, 0, 4, S);
  g.fillStyle = 'rgba(150,155,160,0.35)'; g.fillRect(0, 4, S, 2); g.fillRect(4, 0, 2, S);
  gr.fillStyle = '#e0e0e0'; gr.fillRect(0, 0, S, 6); gr.fillRect(0, 0, 6, S);
  return { map: tex(c), rough: tex(cr, false) };
}

function concreteTex() {                                              // precast plinth / dock
  const S = 256, R = rng(21); const [c, g] = canvas(S, S);
  g.fillStyle = '#8a8e92'; g.fillRect(0, 0, S, S);
  for (let i = 0; i < 5000; i++) { const v = 90 + R() * 90 | 0; g.fillStyle = `rgba(${v},${v},${v},${R() * 0.3})`; g.fillRect(R() * S, R() * S, 2, 2); }
  for (let i = 0; i < 14; i++) { g.fillStyle = `rgba(60,62,64,${0.05 + R() * 0.05})`; g.fillRect(R() * S, 0, 2 + R() * 6, S); }
  g.fillStyle = 'rgba(40,42,44,.7)'; g.fillRect(0, 0, 3, S);
  return tex(c);
}

function claddingMaps() {                                             // trapezoidal steel cladding
  const [c, g] = canvas(256, 256), [cb, gb] = canvas(256, 256);
  for (let x = 0; x < 256; x += 32) {
    const grd = g.createLinearGradient(x, 0, x + 32, 0);
    grd.addColorStop(0, '#9aa3ac'); grd.addColorStop(0.18, '#c9d0d6'); grd.addColorStop(0.5, '#d6dce1');
    grd.addColorStop(0.82, '#bcc4cb'); grd.addColorStop(1, '#949da6');
    g.fillStyle = grd; g.fillRect(x, 0, 32, 256);
    const gb2 = gb.createLinearGradient(x, 0, x + 32, 0);
    gb2.addColorStop(0, '#202020'); gb2.addColorStop(0.2, '#e0e0e0'); gb2.addColorStop(0.8, '#e0e0e0'); gb2.addColorStop(1, '#202020');
    gb.fillStyle = gb2; gb.fillRect(x, 0, 32, 256);
  }
  const R = rng(4);
  for (let i = 0; i < 40; i++) { g.fillStyle = `rgba(70,75,80,${R() * 0.05})`; g.fillRect(R() * 256, 0, 1 + R() * 3, 256); }
  return { map: tex(c), bump: tex(cb, false) };
}

function cartonTex(wrapped) {
  const S = 256, R = rng(wrapped ? 5 : 3); const [c, g] = canvas(S, S);
  g.fillStyle = '#b8895a'; g.fillRect(0, 0, S, S);
  for (let i = 0; i < 1800; i++) { g.fillStyle = `rgba(95,62,30,${R() * 0.1})`; g.fillRect(R() * S, R() * S, 3, 1); }
  for (let i = 0; i < 500; i++) { g.fillStyle = `rgba(225,195,155,${R() * 0.08})`; g.fillRect(R() * S, R() * S, 3, 1); }
  for (let cx = 0; cx < 2; cx++) for (let cy = 0; cy < 2; cy++) {       // 2 x 2 cartons per face
    const x0 = cx * 128, y0 = cy * 128;
    g.fillStyle = R() < 0.5 ? `rgba(255,235,210,${R() * 0.07})` : `rgba(80,50,20,${R() * 0.07})`;
    g.fillRect(x0, y0, 128, 128);
    g.fillStyle = 'rgba(205,168,120,.8)'; g.fillRect(x0 + 56, y0, 16, 128);       // tape
    g.fillStyle = 'rgba(120,85,45,.35)'; g.fillRect(x0 + 55, y0, 1, 128); g.fillRect(x0 + 72, y0, 1, 128);
    g.strokeStyle = 'rgba(60,38,16,.55)'; g.lineWidth = 3; g.strokeRect(x0 + 1.5, y0 + 1.5, 125, 125);
    g.strokeStyle = 'rgba(230,200,160,.25)'; g.lineWidth = 1; g.strokeRect(x0 + 4, y0 + 4, 120, 120);
  }
  g.fillStyle = '#f2efe8'; g.fillRect(14, 166, 36, 26);                   // shipping label + barcode
  g.fillStyle = '#222';
  for (let x = 17; x < 47; x += 2 + (R() * 2 | 0)) g.fillRect(x, 170, 1 + (R() * 1.6 | 0), 12);
  if (wrapped) {                                                       // stretch film
    g.fillStyle = 'rgba(214,226,236,.26)'; g.fillRect(0, 0, S, S);
    for (let y = 0; y < S; y += 22) { g.fillStyle = `rgba(255,255,255,${0.05 + R() * 0.07})`; g.fillRect(0, y, S, 8 + R() * 6); }
    for (let i = 0; i < 9; i++) {
      g.strokeStyle = `rgba(255,255,255,${0.12 + R() * 0.2})`; g.lineWidth = 1 + R() * 4;
      const x = R() * S; g.beginPath(); g.moveTo(x, 0); g.lineTo(x + 60 + R() * 60, S); g.stroke();
    }
  }
  return tex(c);
}

function woodTex() {
  const R = rng(8); const [c, g] = canvas(128, 128);
  g.fillStyle = '#b38a5c'; g.fillRect(0, 0, 128, 128);
  for (let i = 0; i < 60; i++) {
    g.strokeStyle = `rgba(${R() < 0.5 ? '90,58,28' : '210,175,130'},${0.1 + R() * 0.15})`; g.lineWidth = 1;
    const y = R() * 128; g.beginPath(); g.moveTo(0, y);
    for (let x = 0; x <= 128; x += 16) g.lineTo(x, y + Math.sin(x * 0.05 + i) * 2);
    g.stroke();
  }
  return tex(c);
}

function crateTex() {                                                 // red plastic tote
  const [c, g] = canvas(256, 256);
  g.fillStyle = '#c23b33'; g.fillRect(0, 0, 256, 256);
  g.fillStyle = '#d8554b'; g.fillRect(0, 0, 256, 18);                     // rim
  for (let y = 40; y < 256; y += 30) { g.fillStyle = 'rgba(90,15,12,.35)'; g.fillRect(8, y, 240, 5); g.fillStyle = 'rgba(255,140,130,.18)'; g.fillRect(8, y + 5, 240, 2); }
  for (let x = 8; x < 256; x += 60) { g.fillStyle = 'rgba(90,15,12,.3)'; g.fillRect(x, 20, 5, 236); }
  g.fillStyle = '#4b1410';                                                 // hand hole
  g.beginPath(); g.roundRect ? g.roundRect(84, 28, 88, 26, 12) : g.rect(84, 28, 88, 26); g.fill();
  g.strokeStyle = 'rgba(60,10,8,.6)'; g.lineWidth = 4; g.strokeRect(2, 2, 252, 252);
  return tex(c);
}

function uprightTex() {                                               // perforated rack upright
  const [c, g] = canvas(32, 512);
  g.fillStyle = '#2366b3'; g.fillRect(0, 0, 32, 512);
  g.fillStyle = 'rgba(255,255,255,.12)'; g.fillRect(3, 0, 3, 512);
  for (let y = 6; y < 512; y += 14) { g.fillStyle = '#0d2748'; g.fillRect(13, y, 6, 8); }
  return tex(c);
}

function deckTex() {                                                  // wire-mesh decking
  const [c, g] = canvas(64, 64);
  g.fillStyle = '#5d6369'; g.fillRect(0, 0, 64, 64);
  g.fillStyle = '#aab1b8';
  for (let i = 0; i < 64; i += 16) { g.fillRect(i, 0, 3, 64); g.fillRect(0, i, 64, 2); }
  return tex(c);
}

function beltTex() {
  const [c, g] = canvas(128, 32);
  g.fillStyle = '#2a2d31'; g.fillRect(0, 0, 128, 32);
  for (let x = 0; x < 128; x += 16) { g.fillStyle = '#3d4247'; g.fillRect(x, 0, 4, 32); g.fillStyle = '#1d1f22'; g.fillRect(x + 4, 0, 2, 32); }
  return tex(c);
}

function stripeTex() {                                                // yellow/black hazard tape
  const [c, g] = canvas(128, 32);
  g.fillStyle = '#f0bf2a'; g.fillRect(0, 0, 128, 32);
  g.fillStyle = '#1b1b1b';
  for (let x = -32; x < 160; x += 64) { g.beginPath(); g.moveTo(x, 32); g.lineTo(x + 32, 0); g.lineTo(x + 64, 0); g.lineTo(x + 32, 32); g.fill(); }
  return tex(c);
}

function grilleTex() {
  const [c, g] = canvas(64, 64);
  g.fillStyle = '#1f2226'; g.fillRect(0, 0, 64, 64);
  for (let y = 4; y < 64; y += 8) { g.fillStyle = '#50575e'; g.fillRect(3, y, 58, 3); }
  return tex(c);
}

function vestTex() {                                                  // hi-vis vest
  const [c, g] = canvas(64, 64);
  g.fillStyle = '#d7e83a'; g.fillRect(0, 0, 64, 64);
  g.fillStyle = '#c3c8cc'; g.fillRect(0, 22, 64, 6); g.fillRect(0, 40, 64, 6);
  return tex(c);
}

function signTex(text, { bg = '#23405f', fg = '#ffffff', w = 512, h = 128, size = 72, border = null } = {}) {
  const [c, g] = canvas(w, h);
  g.fillStyle = bg; g.fillRect(0, 0, w, h);
  if (border) { g.strokeStyle = border; g.lineWidth = 10; g.strokeRect(5, 5, w - 10, h - 10); }
  g.fillStyle = fg; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.font = `700 ${size}px "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`;
  g.fillText(text, w / 2, h / 2 + size * 0.04);
  const t = tex(c); t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; return t;
}

// soft shapes for contact shadows, glows and light pools (alpha only)
function softTex(inset, rgb = '0,0,0', power = 1.6) {
  const S = 64; const [c, g] = canvas(S, S); const img = g.createImageData(S, S);
  const [r, gg, b] = rgb.split(',').map(Number);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const u = Math.abs((x + 0.5) / S - 0.5), v = Math.abs((y + 0.5) / S - 0.5);
    const dx = Math.max(0, u - (0.5 - inset)) / inset, dy = Math.max(0, v - (0.5 - inset)) / inset;
    const a = Math.pow(clamp01(1 - Math.hypot(dx, dy)), power);
    const k = (y * S + x) * 4; img.data[k] = r; img.data[k + 1] = gg; img.data[k + 2] = b; img.data[k + 3] = a * 255;
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ---------------------------------------------------------------- geometry helpers
function prep(geo) {
  const g = geo.index ? geo.toNonIndexed() : geo.clone();
  for (const k of Object.keys(g.attributes)) if (k !== 'position' && k !== 'normal' && k !== 'uv') g.deleteAttribute(k);
  g.morphAttributes = {}; g.clearGroups();
  return g;
}
// Static batching: every part with the same material becomes one draw call.
function Batch() {
  const groups = new Map();
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), p = new THREE.Vector3(), sc = new THREE.Vector3();
  return {
    add(geo, mat, x, y, z, o = {}) {
      e.set(o.rx || 0, o.ry || 0, o.rz || 0); q.setFromEuler(e); p.set(x, y, z); sc.set(o.sx || 1, o.sy || 1, o.sz || 1);
      const g = prep(geo); g.applyMatrix4(m4.compose(p, q, sc));
      const key = mat.uuid + (o.cast === false ? '-' : '+');
      if (!groups.has(key)) groups.set(key, { mat, cast: o.cast !== false, geos: [] });
      groups.get(key).geos.push(g);
    },
    build(parent) {
      for (const { mat, cast, geos } of groups.values()) {
        const m = new THREE.Mesh(mergeGeometries(geos), mat);
        m.castShadow = cast; m.receiveShadow = true; parent.add(m);
      }
      groups.clear();
    },
  };
}
const box = (w, h, d) => new THREE.BoxGeometry(w, h, d);
const rbox = (w, h, d, r, seg = 2) => new RoundedBoxGeometry(w, h, d, seg, r);
const cyl = (r, h, n = 16, r2 = r) => new THREE.CylinderGeometry(r2, r, h, n);
const plane = (w, h) => new THREE.PlaneGeometry(w, h);
const flat = (w, h) => new THREE.PlaneGeometry(w, h).rotateX(-Math.PI / 2);   // lies on the floor

// a pallet load: wooden pallet + carton block, one geometry with two material groups
function loadGeometry() {
  const parts = [];
  const add = (g, x, y, z) => { const p = prep(g); p.translate(x, y, z); parts.push(p); };
  const pw = 0.58, top = -0.275 + 0.1;
  for (let i = 0; i < 5; i++) add(box(pw, 0.022, 0.095), 0, top - 0.011, -pw / 2 + 0.0475 + i * (pw - 0.095) / 4);
  for (const x of [-0.24, 0, 0.24]) for (const z of [-0.24, 0, 0.24]) add(box(0.09, 0.056, 0.09), x, top - 0.022 - 0.028, z);
  for (const x of [-0.24, 0, 0.24]) add(box(0.1, 0.022, pw), x, -0.275 + 0.011, 0);
  const pallet = mergeGeometries(parts);
  const carton = prep(rbox(0.54, 0.43, 0.54, 0.018)); carton.translate(0, top + 0.215 + 0.003, 0);
  return mergeGeometries([pallet, carton], true);
}

// ---------------------------------------------------------------- instanced unit loads
const UP = new THREE.Vector3(0, 1, 0);
class Units {
  constructor(parent, kinds, cap) {
    this.lists = []; this.dirty = true;
    this.meshes = kinds.map(k => {
      const m = new THREE.InstancedMesh(k.geo, k.mat, cap);
      m.count = 0; m.castShadow = true; m.receiveShadow = true; m.frustumCulled = false;
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      m.setColorAt(0, new THREE.Color(1, 1, 1));
      parent.add(m); return m;
    });
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion(); this._s = new THREE.Vector3(); this._c = new THREE.Color();
  }
  sync() {
    if (!this.dirty) return false;
    this.dirty = false;
    const n = this.meshes.map(() => 0);
    for (const list of this.lists) for (const it of list) {
      if (it.s < 0.02) continue;
      const m = this.meshes[it.kind], i = n[it.kind];
      if (i >= m.instanceMatrix.count) continue;
      n[it.kind]++;
      this._q.setFromAxisAngle(UP, it.ry); this._s.setScalar(it.s);
      m.setMatrixAt(i, this._m.compose(it.p, this._q, this._s));
      m.setColorAt(i, this._c.setRGB(it.tint, it.tint * 0.99, it.tint * 0.97));
    }
    this.meshes.forEach((m, k) => { m.count = n[k]; m.instanceMatrix.needsUpdate = true; m.instanceColor.needsUpdate = true; });
    return true;
  }
}
const CARD = 0, WRAP = 1, CRATE = 2;
const loadKind = i => (hash(i * 1.37 + 0.5) < 0.3 ? WRAP : CARD);

// ---------------------------------------------------------------- scene
function build(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  ANISO = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  container.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('aria-hidden', 'true');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#1d2126');
  scene.fog = new THREE.Fog('#1d2126', 30, 62);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.42;
  pmrem.dispose();

  const camera = new THREE.PerspectiveCamera(40, 2, 0.1, 140);
  camera.position.set(-6.6, 8.8, 13.6);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(1.5, 0.6, -0.6);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 7; controls.maxDistance = 22;
  controls.minPolarAngle = 0.35; controls.maxPolarAngle = 1.35;
  controls.minAzimuthAngle = -1.25; controls.maxAzimuthAngle = 0.55;   // stay inside the open (cut-away) sides
  controls.update();

  // lights: cool LED high-bay key with shadows, soft sky fill, dim warm bounce
  scene.add(new THREE.HemisphereLight('#d6e0ec', '#3b342c', 0.55));
  const sun = new THREE.DirectionalLight('#f4f1ea', 2.1);
  sun.position.set(-7, 15, 9);
  sun.target.position.set(2, 0, -2); scene.add(sun.target);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -17, right: 17, top: 13, bottom: -13, near: 2, far: 45 });
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02; sun.shadow.radius = 3;
  scene.add(sun);
  const fill = new THREE.DirectionalLight('#9fb6d6', 0.35);            // cool rim from the dock side
  fill.position.set(12, 6, -4); scene.add(fill);

  const floorM = floorMaps(), clad = claddingMaps();
  const std = o => new THREE.MeshStandardMaterial(o);
  const M = {
    floor: std({ map: floorM.map, roughnessMap: floorM.rough, roughness: 0.9, metalness: 0.0 }),
    wall: std({ map: clad.map, color: '#c9cfd5', bumpMap: clad.bump, bumpScale: 1.2, roughness: 0.5, metalness: 0.35 }),
    concrete: std({ map: concreteTex(), roughness: 0.85 }),
    steel: std({ color: '#5b6570', roughness: 0.45, metalness: 0.6 }),
    steelDark: std({ color: '#2c3136', roughness: 0.5, metalness: 0.55 }),
    upright: std({ map: uprightTex(), roughness: 0.4, metalness: 0.5 }),
    beam: std({ color: '#e0701d', roughness: 0.42, metalness: 0.4 }),
    deck: std({ map: deckTex(), roughness: 0.5, metalness: 0.7 }),
    card: std({ map: cartonTex(false), roughness: 0.88 }),
    wrap: std({ map: cartonTex(true), roughness: 0.3, metalness: 0.05 }),
    wood: std({ map: woodTex(), roughness: 0.92 }),
    crate: std({ map: crateTex(), roughness: 0.48 }),
    yellow: std({ color: '#f0b41c', roughness: 0.38, metalness: 0.15 }),
    paintY: std({ color: '#e6b422', roughness: 0.65, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }),
    paintW: std({ color: '#dfe3e6', roughness: 0.65, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }),
    walk: std({ color: '#3e6a55', roughness: 0.7, transparent: true, opacity: 0.55, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
    stripe: std({ map: stripeTex(), roughness: 0.6 }),
    stripeFloor: std({ map: stripeTex(), roughness: 0.65, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }),
    rubber: std({ color: '#1b1c1e', roughness: 0.95 }),
    tyre: std({ color: '#161719', roughness: 0.88 }),
    hub: std({ color: '#9aa1a8', roughness: 0.35, metalness: 0.8 }),
    cab: std({ color: '#e9ecef', roughness: 0.28, metalness: 0.3 }),
    trim: std({ color: '#3a3f45', roughness: 0.5, metalness: 0.4 }),
    glass: std({ color: '#3a4f66', roughness: 0.12, metalness: 0.85 }),
    grille: std({ map: grilleTex(), roughness: 0.4, metalness: 0.6 }),
    bed: std({ map: woodTex(), color: '#8c7a66', roughness: 0.85 }),
    counter: std({ color: '#47515b', roughness: 0.5, metalness: 0.3 }),
    lampHousing: std({ color: '#30353b', roughness: 0.45, metalness: 0.7 }),
    lampPanel: std({ color: '#ffffff', emissive: '#fff4e2', emissiveIntensity: 2.6 }),
    window: std({ color: '#223044', emissive: '#30455f', emissiveIntensity: 0.55, roughness: 0.15, metalness: 0.6 }),
    exit: std({ map: signTex('EXIT', { bg: '#1f8a4c', size: 84 }), emissive: '#ffffff', emissiveMap: signTex('EXIT', { bg: '#1f8a4c', size: 84 }), emissiveIntensity: 0.9 }),
    skin: std({ color: '#c9a088', roughness: 0.7 }),
    helmet: std({ color: '#f3f4f5', roughness: 0.35 }),
    trousers: std({ color: '#2c3440', roughness: 0.85 }),
    shirt: std({ color: '#3b4757', roughness: 0.8 }),
    vest: std({ map: vestTex(), roughness: 0.6 }),
    asphalt: std({ color: '#2a2c2f', roughness: 0.95 }),
    // status lights (emissive toggled by the controller)
    beacon: std({ color: '#7a5320', emissive: '#ffae3a', emissiveIntensity: 0, roughness: 0.3 }),
    signal: std({ color: '#5e4a2c', emissive: '#ffae3a', emissiveIntensity: 0, roughness: 0.3 }),
    hazard: std({ color: '#8f6428', emissive: '#ffae3a', emissiveIntensity: 0, roughness: 0.3 }),
    headlight: std({ color: '#dfe6ee', emissive: '#fffaf0', emissiveIntensity: 0.15, roughness: 0.1 }),
    reverse: std({ color: '#d9dde2', emissive: '#ffffff', emissiveIntensity: 0, roughness: 0.2 }),
  };
  const blobMat = (inset, opacity) => new THREE.MeshBasicMaterial({ map: softTex(inset), transparent: true, opacity, depthWrite: false, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 });
  const SHADOW = { rect: blobMat(0.22, 0.55), round: blobMat(0.5, 0.5) };
  const glowTex = softTex(0.5, '255,255,255', 2.2);
  const additive = (color, opacity) => new THREE.MeshBasicMaterial({ map: glowTex, color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  const B = Batch();
  const shadowPlane = (w, d, x, z, mat = SHADOW.rect, parent = scene, y = 0.008) => {
    const m = new THREE.Mesh(flat(w, d), mat); m.position.set(x, y, z); m.renderOrder = 1; parent.add(m); return m;
  };

  // ---- building shell (inward-facing planes: from outside you look straight in)
  B.add(worldUV(flat(30, 20), 30, 20, 6, 6), M.floor, 0, 0, 0, { cast: false });
  const wall = (w, h, x, y, z, ry) => B.add(worldUV(plane(w, h), w, h, 2, h), M.wall, x, y, z, { ry, cast: false });
  wall(24.2, 7, -2.9, 3.5, -7.2, 0);                     // back wall, left of the dock door
  wall(1.9, 7, 13.55, 3.5, -7.2, 0);                     // right of the door
  wall(3.4, 2.7, 10.9, 5.65, -7.2, 0);                   // lintel
  wall(17.2, 7, -12.2, 3.5, 1.4, Math.PI / 2);           // left wall
  wall(17.2, 7, 14.5, 3.5, 1.4, -Math.PI / 2);           // right wall
  const plinth = (w, x, z, ry) => B.add(worldUV(plane(w, 1.1), w, 1.1, 3, 1.1), M.concrete, x, 0.55, z, { ry, cast: false });
  plinth(24.2, -2.9, -7.17, 0); plinth(1.9, 13.55, -7.17, 0);
  plinth(17.2, -12.17, 1.4, Math.PI / 2); plinth(17.2, 14.47, 1.4, -Math.PI / 2);
  // high clerestory windows (night outside) and steel frame
  for (let x = -11; x <= 7; x += 2) {
    B.add(plane(1.8, 0.8), M.window, x, 5.9, -7.18, { cast: false });
    B.add(box(0.08, 0.95, 0.06), M.steelDark, x + 0.95, 5.9, -7.15, { cast: false });
  }
  B.add(box(20, 0.08, 0.08), M.steelDark, -2, 5.46, -7.15, { cast: false });
  B.add(box(20, 0.08, 0.08), M.steelDark, -2, 6.34, -7.15, { cast: false });
  const column = (x, z, ry = 0) => {                      // I-section columns
    B.add(box(0.32, 7, 0.04), M.steel, x, 3.5, z - 0.12, { ry, cast: false });
    B.add(box(0.32, 7, 0.04), M.steel, x, 3.5, z + 0.12, { ry, cast: false });
    B.add(box(0.04, 7, 0.24), M.steel, x, 3.5, z, { ry, cast: false });
    B.add(box(0.5, 0.05, 0.5), M.steelDark, x, 0.03, z, { cast: false });
  };
  for (const x of [-11.9, -6, 0, 6]) column(x, -6.95);
  column(14.25, -3, Math.PI / 2); column(14.25, 3, Math.PI / 2);
  B.add(box(27, 0.35, 0.22), M.steel, 1.2, 6.62, -6.95, { cast: false });     // eaves beam

  // high-bay LED lamps with warm-white pools on the floor
  const lampSpots = [[-8.5, -3.8], [-3, -3.8], [2.5, -3.8], [8, -3.8], [-8.5, 2.2], [-3, 2.2], [2.5, 2.2], [8, 2.2], [12.2, 2.2]];
  const poolMat = additive('#ffe2b8', 0.16);
  for (const [x, z] of lampSpots) {
    B.add(cyl(0.02, 1.0, 6), M.steelDark, x, 6.55, z, { cast: false });
    B.add(cyl(0.34, 0.16, 20, 0.26), M.lampHousing, x, 5.98, z, { cast: false });
    B.add(cyl(0.3, 0.02, 20), M.lampPanel, x, 5.895, z, { cast: false });
    const pool = new THREE.Mesh(flat(5.6, 5.6), poolMat); pool.position.set(x, 0.012, z); pool.renderOrder = 2; scene.add(pool);
  }

  // ---- floor markings
  const line = (x1, z1, x2, z2, mat = M.paintY, w = 0.1) => {
    const len = Math.hypot(x2 - x1, z2 - z1);
    B.add(flat(len, w), mat, (x1 + x2) / 2, 0.004, (z1 + z2) / 2, { ry: -Math.atan2(z2 - z1, x2 - x1), cast: false });
  };
  // pedestrian walkway (green) between racks and dispatch
  B.add(flat(16.6, 0.9), M.walk, -3.3, 0.003, 1.75, { cast: false });
  line(-11.6, 1.25, 5.0, 1.25); line(-11.6, 2.25, 5.0, 2.25);
  // forklift aisle next to the racks
  line(1.25, -6.9, 1.25, 1.05); line(5.2, -6.9, 5.2, 1.05); line(1.25, 1.05, 5.2, 1.05, M.stripeFloor, 0.16);
  // floor-stock bays in front of the racks
  line(-6.6, -1.3, -0.25, -1.3, M.paintW, 0.06); line(-6.6, 0.1, -0.25, 0.1, M.paintW, 0.06);
  line(-6.6, -1.3, -6.6, 0.1, M.paintW, 0.06); line(-0.25, -1.3, -0.25, 0.1, M.paintW, 0.06);
  // truck lane to the dock door
  line(9.55, -7.1, 9.55, 4.8); line(12.25, -7.1, 12.25, 4.8);

  // ---- pallet racking: blue perforated uprights, orange beams, wire decks
  const slots = [];
  const bayW = BOX + 0.12, levelH = 1.25, rackLen = BAYS * bayW;
  const rackZ = [-6.2, -4.3, -2.4];
  const upH = LEVELS * levelH + 0.4;
  const d = DEPTH * (BOX + 0.05);
  const x0 = -rackLen / 2 - 2.5;
  for (let r = 0; r < RACK_ROWS; r++) {
    const z0 = rackZ[r];
    for (let b = 0; b <= BAYS; b += 2) {
      const x = x0 + b * bayW;
      for (const dz of [-d / 2, d / 2]) {
        B.add(box(0.1, upH, 0.08), M.upright, x, upH / 2, z0 + dz);
        B.add(box(0.2, 0.02, 0.16), M.steelDark, x, 0.01, z0 + dz, { cast: false });
      }
      // frame bracing: horizontals + zig-zag diagonals between the two uprights
      for (let k = 0; k < 4; k++) {
        const y1 = 0.25 + k * 0.95, y2 = y1 + 0.95;
        B.add(box(0.03, 0.03, d), M.upright, x, y1, z0, { cast: false });
        const len = Math.hypot(d, 0.95);
        B.add(box(0.025, len, 0.025), M.upright, x, (y1 + y2) / 2, z0, { rx: (k % 2 ? 1 : -1) * Math.atan2(d, 0.95), cast: false });
      }
    }
    for (let l = 0; l < LEVELS; l++) {
      const y = 0.18 + l * levelH;
      for (const dz of [-d / 2, d / 2]) B.add(box(rackLen + 0.1, 0.11, 0.05), M.beam, x0 + rackLen / 2, y, z0 + dz);
      B.add(worldUV(box(rackLen, 0.02, d), rackLen, d, 0.3, 0.3), M.deck, x0 + rackLen / 2, y + 0.065, z0, { cast: false });
      for (let bb = 0; bb < BAYS; bb++)
        for (let k = 0; k < DEPTH; k++)
          slots.push({ l, r, pos: new THREE.Vector3(x0 + (bb + 0.5) * bayW, y + 0.08 + BOX / 2, z0 - d / 2 + (k + 0.5) * (BOX + 0.05)) });
    }
    shadowPlane(rackLen + 0.9, d + 0.9, x0 + rackLen / 2, z0);
  }
  // aisle signs on the rack ends
  ['C', 'B', 'A'].forEach((ch, r) => {
    const sgn = new THREE.Mesh(plane(0.42, 0.42), std({ map: signTex(ch, { w: 128, h: 128, size: 88, bg: '#2b3440' }), roughness: 0.6 }));
    sgn.position.set(x0 + rackLen + 0.02, upH - 0.3, rackZ[r] + d / 2 + 0.06); scene.add(sgn);
  });
  // fill order: lowest level first, front rack first -- stock visibly "rises"
  slots.sort((a, b) => (a.l - b.l) || (b.r - a.r));
  // overflow stock goes on the floor in front of the racks
  for (let i = 0; i < 60; i++) {
    const c = i % 10, row = (i / 10 | 0) % 2, h = i / 20 | 0;
    slots.push({ pos: new THREE.Vector3(-6.2 + c * 0.62, BOX / 2 + 0.005 + h * BOX, -0.9 + row * 0.62) });
  }

  // safety bollards
  const bollard = (x, z) => {
    B.add(cyl(0.11, 1.0, 16), M.yellow, x, 0.5, z);
    B.add(cyl(0.113, 0.1, 16), M.steelDark, x, 0.78, z, { cast: false });
    B.add(cyl(0.113, 0.1, 16), M.steelDark, x, 0.55, z, { cast: false });
  };
  for (const x of [-10, -7.4, -4.8, -2.2, 0.4]) bollard(x, 1.05);
  bollard(9.35, -6.75); bollard(12.45, -6.75);

  // ---- loading dock (side-loading platform for the flatbed)
  B.add(worldUV(box(3.4, 0.9, 5.6), 3.4, 5.6, 3, 3), M.concrete, 7.9, 0.45, -4.0);
  B.add(worldUV(box(0.04, 0.9, 5.6), 5.6, 0.9, 1.2, 0.9), M.stripe, 9.62, 0.45, -4.0, { ry: 0, cast: false });
  B.add(worldUV(box(3.4, 0.9, 0.04), 3.4, 0.9, 1.2, 0.9), M.stripe, 7.9, 0.45, -1.19, { cast: false });
  B.add(box(0.36, 0.02, 5.4), M.steel, 9.4, 0.91, -4.0, { cast: false });                 // leveller plate
  for (const z of [-6.0, -4.0, -2.0]) B.add(rbox(0.14, 0.34, 0.5, 0.04), M.rubber, 9.69, 0.62, z);
  for (let z = -6.4; z <= -1.8; z += 1.15) B.add(cyl(0.025, 1.0, 6), M.yellow, 6.35, 1.4, z, { cast: false });  // guard rail posts
  B.add(box(0.05, 0.05, 4.7), M.yellow, 6.35, 1.9, -4.1, { cast: false });
  B.add(box(0.05, 0.05, 4.7), M.yellow, 6.35, 1.45, -4.1, { cast: false });
  shadowPlane(4.2, 6.4, 7.9, -4.0);

  // ---- dock door with rubber seals, open roller shutter, status lights
  for (const x of [9.1, 12.7]) {
    B.add(box(0.22, 4.4, 0.36), M.steel, x, 2.2, -7.05);
    B.add(worldUV(box(0.3, 1.2, 0.3), 0.3, 1.2, 0.3, 0.6), M.stripe, x, 0.6, -6.85, { cast: false });
  }
  B.add(box(3.84, 0.26, 0.36), M.steel, 10.9, 4.42, -7.05);
  for (const x of [9.29, 12.51]) B.add(box(0.14, 4.3, 0.12), M.rubber, x, 2.15, -6.93, { cast: false });  // seals
  B.add(box(3.3, 0.14, 0.12), M.rubber, 10.9, 4.24, -6.93, { cast: false });
  B.add(rbox(3.9, 0.52, 0.5, 0.08), M.steelDark, 10.9, 4.85, -6.95);                                   // shutter drum
  B.add(worldUV(box(3.3, 0.12, 0.06), 3.3, 0.12, 0.6, 0.12), M.stripe, 10.9, 4.08, -7.12, { cast: false }); // shutter bottom rail
  const dockSign = new THREE.Mesh(plane(2.0, 0.54), std({ map: signTex('DOCK 1', { size: 80 }), roughness: 0.55 }));
  dockSign.position.set(7.45, 3.3, -7.16); scene.add(dockSign);
  // status column beside the door: supplier beacon (dome, top), transport signal (round lamp, below)
  B.add(box(0.5, 0.06, 0.34), M.steelDark, 8.8, 3.52, -7.0, { cast: false });
  B.add(cyl(0.2, 0.08, 20), M.steelDark, 8.8, 3.59, -6.95, { cast: false });
  B.add(new THREE.CylinderGeometry(0.16, 0.18, 0.22, 20), M.beacon, 8.8, 3.74, -6.95, { cast: false });
  B.add(new THREE.SphereGeometry(0.16, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2), M.beacon, 8.8, 3.85, -6.95, { cast: false });
  // transport signal lamp on the left jamb
  B.add(rbox(0.44, 0.6, 0.22, 0.05), M.steelDark, 8.8, 2.7, -6.92, { cast: false });
  B.add(cyl(0.16, 0.06, 24).rotateX(Math.PI / 2), M.signal, 8.8, 2.7, -6.79, { cast: false });
  B.add(box(0.4, 0.04, 0.14), M.steelDark, 8.8, 2.9, -6.74, { cast: false });
  const beaconGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: '#ffab3a', blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity: 0 }));
  beaconGlow.position.set(8.8, 3.8, -6.75); beaconGlow.scale.setScalar(2.2); scene.add(beaconGlow);
  const signalGlow = new THREE.Sprite(beaconGlow.material.clone());
  signalGlow.position.set(8.8, 2.7, -6.7); signalGlow.scale.setScalar(1.7); scene.add(signalGlow);
  // yard outside the door (seen through the opening)
  B.add(flat(10, 16), M.asphalt, 10.9, -0.005, -15.3, { cast: false });
  line(9.55, -8, 9.55, -22); line(12.25, -8, 12.25, -22);
  // exit sign on the right wall
  const exitSign = new THREE.Mesh(plane(0.9, 0.3), M.exit);
  exitSign.position.set(14.46, 3.2, 0.8); exitSign.rotation.y = -Math.PI / 2; scene.add(exitSign);

  // ---- dispatch: conveyor into the customer counter (unmet orders stack up here as red crates)
  const CONV = { x1: 0.7, x2: -2.95, z: 3.1, y: 0.82 };
  const convLen = CONV.x1 - CONV.x2, convX = (CONV.x1 + CONV.x2) / 2;
  for (const dz of [-0.43, 0.43]) B.add(box(convLen, 0.14, 0.05), M.steel, convX, CONV.y - 0.02, CONV.z + dz);
  for (const x of [CONV.x1 - 0.1, convX, CONV.x2 + 0.15]) for (const dz of [-0.38, 0.38])
    B.add(box(0.06, CONV.y - 0.05, 0.06), M.steelDark, x, (CONV.y - 0.05) / 2, CONV.z + dz);
  for (const x of [CONV.x1, CONV.x2]) B.add(cyl(0.06, 0.82, 12).rotateX(Math.PI / 2), M.hub, x, CONV.y - 0.03, CONV.z, { cast: false });
  const beltTexture = beltTex();
  const belt = new THREE.Mesh(worldUV(flat(convLen, 0.8), convLen, 0.8, 0.5, 0.8), std({ map: beltTexture, roughness: 0.8 }));
  belt.position.set(convX, CONV.y + 0.005, CONV.z); belt.receiveShadow = true; scene.add(belt);
  shadowPlane(convLen + 0.5, 1.4, convX, CONV.z);
  const dispatchSign = new THREE.Mesh(plane(1.9, 0.44), std({ map: signTex('DISPATCH', { bg: '#2b3440', size: 70 }), roughness: 0.55 }));
  dispatchSign.position.set(-1.1, 2.55, CONV.z); scene.add(dispatchSign);
  for (const x of [-1.85, -0.35]) B.add(cyl(0.008, 4.2, 4), M.steelDark, x, 4.9, CONV.z, { cast: false });
  B.add(box(4.6, 1.05, 0.9), M.counter, -5.3, 0.525, 3.1);
  B.add(box(4.7, 0.06, 1.0), M.steel, -5.3, 1.08, 3.1);
  B.add(worldUV(box(4.6, 0.1, 0.05), 4.6, 0.1, 0.4, 0.1), M.stripe, -5.3, 0.06, 3.57, { cast: false });
  shadowPlane(5.4, 1.8, -5.3, 3.1);
  // 7 x 2 crates per layer, at most 3 layers high, so the stack never hides the racks
  const unmetSlots = [];
  for (let i = 0; i < 40; i++) unmetSlots.push(new THREE.Vector3(-7.28 + (i % 7) * 0.66, 1.11 + BOX / 2 + ((i / 14) | 0) * BOX, 2.82 + ((i / 7 | 0) % 2) * 0.58));

  B.build(scene);

  // ---- ambient: forklift (scenery only; its forks stay empty)
  const forklift = new THREE.Group();
  {
    const F = Batch();
    F.add(rbox(1.05, 0.5, 1.75, 0.09), M.yellow, 0, 0.5, -0.05);
    F.add(rbox(1.08, 0.62, 0.5, 0.1), M.steelDark, 0, 0.72, -0.85);                 // counterweight
    F.add(box(0.95, 0.06, 0.9), M.steelDark, 0, 0.77, 0.25, { cast: false });
    for (const x of [-0.46, 0.46]) for (const z of [-0.6, 0.42]) F.add(box(0.06, 1.32, 0.06), M.steelDark, x, 1.44, z);
    F.add(box(1.0, 0.05, 1.12), M.steelDark, 0, 2.12, -0.09);                       // overhead guard
    F.add(rbox(0.5, 0.12, 0.45, 0.04), M.rubber, 0, 0.84, -0.3);                     // seat
    F.add(rbox(0.46, 0.42, 0.1, 0.04), M.rubber, 0, 1.07, -0.52);
    F.add(cyl(0.025, 0.5, 6), M.steelDark, 0, 1.0, 0.28, { rx: -0.5 });
    F.add(cyl(0.12, 0.03, 16), M.rubber, 0, 1.22, 0.36, { rx: 1.0 });
    for (const x of [-0.32, 0.32]) F.add(box(0.08, 2.05, 0.12), M.steelDark, x, 1.1, 0.95);   // mast
    F.add(box(0.72, 0.06, 0.1), M.steelDark, 0, 2.08, 0.95);
    F.add(box(0.8, 0.42, 0.06), M.steelDark, 0, 0.38, 1.06);                         // carriage
    for (const x of [-0.24, 0.24]) {
      F.add(box(0.11, 0.035, 1.05), M.steelDark, x, 0.08, 1.6);                      // forks
      F.add(box(0.11, 0.5, 0.035), M.steelDark, x, 0.33, 1.09);
    }
    const w1 = cyl(0.27, 0.22, 18).rotateZ(Math.PI / 2), w2 = cyl(0.22, 0.2, 18).rotateZ(Math.PI / 2);
    for (const x of [-0.5, 0.5]) { F.add(w1, M.tyre, x, 0.27, 0.55); F.add(w2, M.tyre, x, 0.22, -0.62); }
    // seated driver in a hi-vis vest
    F.add(new THREE.CapsuleGeometry(0.16, 0.3, 4, 10), M.vest, 0, 1.2, -0.3);
    F.add(new THREE.SphereGeometry(0.11, 14, 10), M.skin, 0, 1.58, -0.26);
    F.add(new THREE.SphereGeometry(0.125, 14, 8, 0, Math.PI * 2, 0, Math.PI / 2), M.helmet, 0, 1.62, -0.26);
    for (const x of [-0.1, 0.1]) F.add(new THREE.CapsuleGeometry(0.065, 0.35, 4, 8), M.trousers, x, 0.95, -0.05, { rx: Math.PI / 2 });
    for (const x of [-0.21, 0.21]) F.add(new THREE.CapsuleGeometry(0.05, 0.32, 4, 8), M.shirt, x, 1.25, -0.1, { rx: 1.1 });
    F.build(forklift);
    shadowPlane(1.5, 2.9, 0, 0.35, SHADOW.rect, forklift, 0.01);
    const spot = new THREE.Mesh(flat(0.55, 0.55), additive('#3b82ff', 0.55));      // blue safety spot
    spot.position.set(0, 0.016, 3.3); spot.renderOrder = 3; forklift.add(spot);
  }
  scene.add(forklift);
  const flPath = loopPath(3.22, -2.9, 1.02, 2.95, 0.95);

  // ---- ambient: a worker walking the pedestrian lane
  const worker = new THREE.Group();
  const limbs = {};
  {
    const legGeo = new THREE.CapsuleGeometry(0.07, 0.62, 4, 8).translate(0, -0.38, 0);
    const armGeo = new THREE.CapsuleGeometry(0.052, 0.46, 4, 8).translate(0, -0.27, 0);
    const mk = (geo, mat, x, y, z) => { const g = new THREE.Group(); const m = new THREE.Mesh(geo, mat); m.castShadow = true; g.add(m); g.position.set(x, y, z); worker.add(g); return g; };
    limbs.l = mk(legGeo, M.trousers, -0.1, 0.9, 0); limbs.r = mk(legGeo, M.trousers, 0.1, 0.9, 0);
    limbs.la = mk(armGeo, M.shirt, -0.24, 1.46, 0); limbs.ra = mk(armGeo, M.shirt, 0.24, 1.46, 0);
    const W = Batch();
    W.add(new THREE.CapsuleGeometry(0.17, 0.36, 4, 12), M.vest, 0, 1.24, 0, { sz: 0.75 });
    W.add(new THREE.SphereGeometry(0.105, 14, 10), M.skin, 0, 1.69, 0);
    W.add(new THREE.SphereGeometry(0.122, 14, 8, 0, Math.PI * 2, 0, Math.PI / 2), M.helmet, 0, 1.73, 0);
    W.add(box(0.26, 0.02, 0.1), M.helmet, 0, 1.74, 0.1, { cast: false });
    W.build(worker);
    shadowPlane(0.7, 0.7, 0, 0, SHADOW.round, worker, 0.011);
  }
  scene.add(worker);

  // ---- delivery truck (flatbed, so the on-order total is visible as cargo). ONE truck only.
  const truck = new THREE.Group();
  {
    const K = Batch();
    K.add(box(1.0, 0.24, 7.2), M.steelDark, 0, 0.52, 0.55);                             // chassis rails
    K.add(worldUV(box(2.4, 0.14, 5.4), 2.4, 5.4, 1.2, 2.4), M.bed, 0, 0.8, -0.1);        // deck (top 0.87)
    K.add(box(2.42, 0.1, 5.42), M.trim, 0, 0.7, -0.1, { cast: false });
    K.add(box(2.4, 1.55, 0.1), M.steel, 0, 1.65, 2.6);                                   // headboard
    for (let y = 1.0; y <= 2.3; y += 0.32) K.add(box(2.42, 0.05, 0.12), M.steelDark, 0, y, 2.6, { cast: false });
    for (let z = -2.6; z <= 2.3; z += 0.98) for (const x of [-1.17, 1.17]) K.add(box(0.06, 0.34, 0.06), M.steelDark, x, 1.03, z, { cast: false });
    K.add(rbox(2.4, 2.05, 1.8, 0.14, 3), M.cab, 0, 1.58, 3.55);                          // cab
    K.add(rbox(2.3, 0.45, 1.3, 0.12, 3), M.cab, 0, 2.75, 3.3);                           // roof fairing
    K.add(box(2.12, 0.9, 0.04), M.glass, 0, 2.05, 4.46, { rx: -0.08, cast: false });      // windscreen
    for (const x of [-1.215, 1.215]) K.add(box(0.02, 0.72, 0.85), M.glass, x, 2.08, 3.85, { cast: false });
    K.add(box(1.7, 0.6, 0.04), M.grille, 0, 1.12, 4.46, { cast: false });
    K.add(rbox(2.5, 0.32, 0.26, 0.06), M.trim, 0, 0.58, 4.42);                           // bumper
    K.add(box(2.4, 0.08, 0.02), M.trim, 0, 1.55, 4.46, { cast: false });                 // stripe
    for (const x of [-0.85, 0.85]) K.add(rbox(0.36, 0.16, 0.04, 0.03), M.headlight, x, 0.92, 4.47, { cast: false });
    for (const s of [-1, 1]) {
      K.add(box(0.26, 0.04, 0.04), M.trim, s * 1.32, 2.2, 4.1, { cast: false });           // mirrors
      K.add(rbox(0.08, 0.4, 0.2, 0.03), M.trim, s * 1.45, 2.05, 4.1);
      K.add(rbox(0.2, 0.14, 0.06, 0.03), M.hazard, s * 1.1, 0.93, 4.47, { cast: false });   // hazard lamps front
      K.add(rbox(0.2, 0.14, 0.04, 0.03), M.hazard, s * 1.08, 0.72, -2.83, { cast: false });
      K.add(rbox(0.1, 0.1, 0.34, 0.03), M.hazard, s * 1.21, 1.25, 3.2, { cast: false });   // side repeaters   // and rear
      K.add(rbox(0.14, 0.1, 0.04, 0.02), M.reverse, s * 0.8, 0.72, -2.83, { cast: false });
    }
    K.add(cyl(0.26, 0.9, 18).rotateX(Math.PI / 2), M.hub, -0.85, 0.62, 2.3);             // fuel tank
    const tg = cyl(0.45, 0.36, 22).rotateZ(Math.PI / 2), hg = cyl(0.24, 0.38, 16).rotateZ(Math.PI / 2);
    for (const [x, z] of [[-1.05, 3.7], [1.05, 3.7], [-1.05, -1.25], [1.05, -1.25], [-1.05, -2.2], [1.05, -2.2]]) {
      K.add(tg, M.tyre, x, 0.45, z); K.add(hg, M.hub, x, 0.45, z, { cast: false });
    }
    for (const x of [-1.05, 1.05]) K.add(box(0.44, 0.06, 2.1), M.trim, x, 0.96, -1.72, { cast: false });   // mudguards
    K.build(truck);
    shadowPlane(3.1, 8.2, 0, 0.7, SHADOW.rect, truck, 0.009);
  }
  const DOCK_Z = -2.2, OUT_Z = -12.6;
  truck.position.set(10.9, 0, DOCK_Z);
  scene.add(truck);
  const cargoSlots = [];
  for (let i = 0; i < 48; i++) cargoSlots.push(new THREE.Vector3(-0.7 + (i % 3) * 0.7, 0.875 + BOX / 2 + ((i / 3 | 0) % 3) * BOX, -2.3 + (i / 9 | 0) * 0.6));

  // ---- instanced unit loads (world: racks, floor, crates, flights; truck: cargo)
  const loadGeo = loadGeometry();
  const kinds = [
    { geo: loadGeo, mat: [M.wood, M.card] },
    { geo: loadGeo, mat: [M.wood, M.wrap] },
    { geo: rbox(0.55, 0.5, 0.55, 0.03), mat: M.crate },
  ];
  const world = new Units(scene, kinds, 420);
  const cargoUnits = new Units(truck, kinds, 60);

  // anchors for the HTML labels that sit over the scene
  const anchors = {
    stock: new THREE.Vector3(x0 + rackLen / 2, upH + 0.55, -4.3),
    order: new THREE.Vector3(10.9, 3.35, 2.9),
    unmet: new THREE.Vector3(-5.3, 2.3, 3.1),
  };
  return { renderer, scene, camera, controls, sun, M, slots, cargoSlots, unmetSlots, world, cargoUnits, anchors,
           truck, DOCK_Z, OUT_Z, forklift, flPath, worker, limbs, beltTexture, beaconGlow, signalGlow, CONV };
}

// closed rounded-rectangle path in the x/z plane, sampled by arc length
function loopPath(cx, cz, hx, hz, r) {
  const pts = [];
  const corners = [[cx + hx - r, cz - hz + r, -Math.PI / 2], [cx + hx - r, cz + hz - r, 0],
                   [cx - hx + r, cz + hz - r, Math.PI / 2], [cx - hx + r, cz - hz + r, Math.PI]];
  for (const [x, z, a0] of corners) for (let k = 0; k <= 10; k++) {
    const a = a0 + (k / 10) * Math.PI / 2; pts.push(new THREE.Vector2(x + r * Math.cos(a), z + r * Math.sin(a)));
  }
  pts.push(pts[0].clone());
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
  return { pts, cum, len: cum[cum.length - 1] };
}
function samplePath(P, s, out) {
  s = ((s % P.len) + P.len) % P.len;
  let i = 1; while (i < P.cum.length - 1 && P.cum[i] < s) i++;
  const a = P.pts[i - 1], b = P.pts[i], k = (s - P.cum[i - 1]) / (P.cum[i] - P.cum[i - 1] || 1);
  out.x = a.x + (b.x - a.x) * k; out.z = a.y + (b.y - a.y) * k; out.heading = Math.atan2(b.x - a.x, b.y - a.y);
  return out;
}

// ---------------------------------------------------------------- controller
function start(container) {
  const W = build(container);
  // soft vignette over the canvas (under the labels)
  const vig = document.createElement('div');
  vig.setAttribute('aria-hidden', 'true');
  vig.style.cssText = 'position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse 75% 70% at 50% 45%,rgba(0,0,0,0) 60%,rgba(8,10,14,.38) 100%)';
  container.appendChild(vig);
  const labels = {};
  for (const k of ['stock', 'order', 'unmet']) {
    const el = document.createElement('div');
    el.className = 'w3d-label w3d-' + k;
    container.appendChild(el); labels[k] = el;
  }
  const shelf = [], unmet = [], cargo = [];
  W.world.lists.push(shelf, unmet);
  W.cargoUnits.lists.push(cargo);

  let tweens = [], blink = { transport: false, supplier: false }, last = null, gen = 0, needsRender = true;
  let truckMove = 0;                                   // -1 reversing, 0 parked, 1 driving forward

  const item = () => ({ p: new THREE.Vector3(), s: 1, ry: 0, kind: CARD, tint: 1 });
  const dress = (it, i, kind, seed) => { it.s = 1; it.kind = kind; it.ry = (hash(i * 3.1 + seed) - 0.5) * 0.09; it.tint = 0.88 + hash(i * 7.7 + seed) * 0.17; };
  function fill(list, n, place, kindOf, seed, units) {
    list.length = Math.min(list.length, n);
    while (list.length < n) list.push(item());
    list.forEach((it, i) => { it.p.copy(place(i)); dress(it, i, kindOf(i), seed); });
    units.dirty = true;
  }
  const shelfPos = i => W.slots[Math.min(i, W.slots.length - 1)].pos;
  // loads enter and leave a slot from the open aisle in front of it, never through the steel
  const aisle = i => shelfPos(i).clone().add(new THREE.Vector3(0, 0.04, 0.74));
  const setTruck = present => { W.truck.position.z = present ? W.DOCK_Z : W.OUT_Z; W.truck.visible = present; truckMove = 0; };

  function snap(s) {
    gen += 1; tweens = [];
    fill(shelf, Math.min(boxesFor(s.inventory), W.slots.length), shelfPos, loadKind, 0, W.world);
    fill(cargo, Math.min(boxesFor(s.on_order), W.cargoSlots.length), i => W.cargoSlots[i], i => loadKind(i + 500), 500, W.cargoUnits);
    fill(unmet, Math.min(boxesFor(s.backorders), W.unmetSlots.length), i => W.unmetSlots[i], () => CRATE, 900, W.world);
    setTruck(s.on_order > 0);
    blink.transport = !s.transport_ok;
    blink.supplier = s.supplier_reliability < 0.999;
    labels.stock.innerHTML = `<b>${s.inventory.toLocaleString()}</b> in stock`;
    labels.order.innerHTML = `<b>${s.on_order.toLocaleString()}</b> on order${s.transport_ok ? '' : ' · <span class="w3d-warn">delayed</span>'}`;
    labels.unmet.innerHTML = `<b>${s.backorders.toLocaleString()}</b> unmet orders`;
    labels.unmet.classList.toggle('w3d-bad', s.backorders > 0);
    last = s; needsRender = true;
  }

  // generic tween: fn(k) with k in [0,1] over [t0, t0+dur]
  const tween = (t0, dur, fn) => tweens.push({ t0, dur, fn, started: false });
  const _a = new THREE.Vector3();
  function arc(it, from, to, t0, dur, lift, s0 = 1, s1 = 1, units = W.world) {
    const f = from.clone(), g = to.clone();
    tween(t0, dur, k => {
      const e = ease(k);
      it.p.lerpVectors(f, g, e); it.p.y += Math.sin(Math.PI * e) * lift;
      it.s = s0 + (s1 - s0) * e; units.dirty = true;
    });
  }
  function truckDrive(t0, dur, fromZ, toZ) {
    tween(t0, dur, k => {
      const e = fromZ < toZ ? 1 - (1 - k) * (1 - k) : ease(k);
      W.truck.visible = true;
      W.truck.position.z = fromZ + (toZ - fromZ) * e;
      truckMove = k >= 1 ? 0 : (toZ > fromZ ? 1 : -1);
      if (k >= 1 && toZ === W.OUT_Z) W.truck.visible = false;
    });
  }

  function playWeek(before, outcome, after) {
    if (reduced()) { snap(after); return Promise.resolve(); }
    snap(before);
    const g = gen;
    const now = performance.now() / 1000;
    const nArr = boxesFor(outcome.arrivals);
    let arriveAt = now + T.arriveDocked;
    if (nArr > 0 && cargo.length === 0) {
      // arrivals but no truck at the dock: it drives in through the door first
      fill(cargo, Math.min(nArr, W.cargoSlots.length), i => W.cargoSlots[i], i => loadKind(i + 500), 500, W.cargoUnits);
      W.truck.position.z = W.OUT_Z; W.truck.visible = true;
      truckDrive(now, T.truckIn, W.OUT_Z, W.DOCK_Z);
      arriveAt = now + T.truckIn;
    }
    // Phase 1: arrivals leave the truck (top/back of the load first) for the next free slots.
    const arrive = Math.min(nArr, cargo.length);
    const startN = shelf.length;
    const landAt = [];
    W.truck.updateMatrixWorld(true);
    const truckDocked = new THREE.Matrix4().makeTranslation(10.9, 0, W.DOCK_Z);
    for (let i = 0; i < arrive && startN + i < W.slots.length; i++) {
      const src = cargo[cargo.length - 1 - i];
      const from = src.p.clone().applyMatrix4(truckDocked);
      const it = item(); dress(it, startN + i, loadKind(startN + i), 0);
      it.p.copy(from); it.s = 0;                                     // hidden until its flight starts
      shelf.push(it);
      const t0 = arriveAt + i * (T.arriveWin / Math.max(arrive, 1));
      tween(t0, 0.001, () => { src.s = 0; it.s = 1; W.cargoUnits.dirty = true; W.world.dirty = true; });
      const slot = shelfPos(startN + i), front = aisle(startN + i);
      arc(it, from, front, t0, T.fly * 0.72, 1.9);
      arc(it, front, slot, t0 + T.fly * 0.72, T.fly * 0.28, 0.05);
      landAt[startN + i] = t0 + T.fly;
    }
    // Phase 2: sold loads leave the racks, ride the conveyor and go out through dispatch.
    const closeN = Math.min(boxesFor(outcome.inventory_close), W.slots.length);
    const leaving = Math.max(0, shelf.length - closeN);
    const C = W.CONV, beltY = C.y + 0.19;
    const inlet = new THREE.Vector3(C.x1 - 0.2, beltY, C.z), outlet = new THREE.Vector3(C.x2 + 0.15, beltY, C.z);
    for (let j = 0; j < leaving; j++) {
      const idx = shelf.length - 1 - j, it = shelf[idx];
      const t0 = Math.max(now + T.outStart + j * (T.outWin / Math.max(leaving, 1)), (landAt[idx] || 0) + 0.05);
      const from = shelfPos(idx), front = aisle(idx);
      arc(it, from, front, t0, T.lift * 0.3, 0.03);
      arc(it, front, inlet, t0 + T.lift * 0.3, T.lift * 0.7, 1.1, 1, 0.7);
      tween(t0 + T.lift, T.belt, k => { it.p.lerpVectors(inlet, outlet, k); it.s = 0.7; W.world.dirty = true; });
      tween(t0 + T.lift + T.belt, T.drop, k => { it.p.copy(outlet); it.p.x -= k * 0.25; it.s = 0.7 * (1 - k); W.world.dirty = true; });
    }
    // Phase 3: unmet-order crates on the counter follow the week's result.
    const crateN = Math.min(boxesFor(outcome.backorders_close), W.unmetSlots.length);
    const c0 = unmet.length;
    for (let i = crateN; i < c0; i++) {
      const it = unmet[i];
      tween(now + 1.2, 0.35, k => { it.s = 1 - easeIn(k); W.world.dirty = true; });
    }
    for (let i = c0; i < crateN; i++) {
      const it = item(); dress(it, i, CRATE, 900); it.p.copy(W.unmetSlots[i]); it.s = 0; unmet.push(it);
      const to = W.unmetSlots[i].clone(), from = to.clone(); from.y += 0.5;
      tween(now + T.crates + (i - c0) * (0.4 / Math.max(crateN - c0, 1)), 0.3, k => {
        const e = easeOut(k); it.p.lerpVectors(from, to, e); it.s = e; W.world.dirty = true;
      });
    }
    // Phase 4: the truck leaves when nothing is on order, or comes in with the new order.
    const hasAfter = after.on_order > 0, hasNow = cargo.length > 0 || W.truck.visible;
    if (hasNow && !hasAfter) truckDrive(now + T.truckEnd, T.truckEndDur, W.DOCK_Z, W.OUT_Z);
    else if (!hasNow && hasAfter) {
      tween(now + T.truckEnd, 0.001, () => {
        fill(cargo, Math.min(boxesFor(after.on_order), W.cargoSlots.length), i => W.cargoSlots[i], i => loadKind(i + 500), 500, W.cargoUnits);
      });
      truckDrive(now + T.truckEnd, T.truckEndDur, W.OUT_Z, W.DOCK_Z);
    }
    needsRender = true;
    // a newer snap (the participant already pressed Continue) wins over this one
    return new Promise(res => setTimeout(() => { if (g === gen) snap(after); res(); }, T.total * 1000));
  }

  // ------------------------------------------------ adaptive quality (2 high, 1 medium, 0 low)
  const Q = { tier: 2, locked: false, samples: [], warm: 25, frames: 0, time: 0, done: false };
  function applyTier(t) {
    Q.tier = t;
    const dpr = window.devicePixelRatio || 1;
    W.renderer.setPixelRatio(t === 2 ? Math.min(dpr, 1.5) : t === 1 ? Math.min(dpr, 1) : Math.min(dpr, 0.75));
    const shadows = t >= 1;
    if (W.renderer.shadowMap.enabled !== shadows) {
      W.renderer.shadowMap.enabled = shadows; W.sun.castShadow = shadows;
      W.scene.traverse(o => { if (o.material) [].concat(o.material).forEach(m => { m.needsUpdate = true; }); });
    }
    const ms = t === 2 ? 2048 : 1024;
    if (W.sun.shadow.mapSize.x !== ms) { W.sun.shadow.mapSize.set(ms, ms); if (W.sun.shadow.map) { W.sun.shadow.map.dispose(); W.sun.shadow.map = null; } }
    resize(); needsRender = true;
  }
  function measure(dt) {
    if (Q.done || Q.locked) return;
    if (Q.warm > 0) { Q.warm--; return; }                 // skip shader-compile hitches
    Q.frames++; Q.time += dt;
    if (Q.time >= 1.5) {
      const fps = Q.frames / Q.time;
      Q.samples.push({ tier: Q.tier, fps: Math.round(fps) });
      if (fps < 40 && Q.tier > 0) { applyTier(Q.tier - 1); Q.frames = 0; Q.time = 0; Q.warm = 20; }
      else Q.done = true;
    }
  }
  const ambientOn = () => Q.tier >= 1 && !reduced();

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    W.renderer.setSize(w, h, false);
    W.camera.aspect = w / h; W.camera.updateProjectionMatrix();
    needsRender = true;
  }
  new ResizeObserver(resize).observe(container);
  resize();

  // ------------------------------------------------ ambient life (wall clock only, never the data)
  const fp = { x: 0, z: 0, heading: 0 };
  let userMoved = false;
  W.controls.addEventListener('start', () => { userMoved = true; });
  W.controls.addEventListener('change', () => { needsRender = true; });
  const camBase = new THREE.Spherical().setFromVector3(W.camera.position.clone().sub(W.controls.target));
  const _off = new THREE.Vector3();
  function ambient(t) {
    samplePath(W.flPath, t * 1.25, fp);
    W.forklift.position.set(fp.x, 0, fp.z); W.forklift.rotation.y = fp.heading;
    // worker: walk 9.6 m east, turn, walk back, turn (22 s cycle)
    const u = t % 22, xa = -9.4, xb = 0.2, walk = xb - xa, speed = walk / 10.2;
    let x, ry, swing = 0;
    if (u < 10.2) { x = xa + u * speed; ry = Math.PI / 2; swing = u; }
    else if (u < 11) { x = xb; ry = Math.PI / 2 - Math.PI * ease((u - 10.2) / 0.8); }
    else if (u < 21.2) { x = xb - (u - 11) * speed; ry = -Math.PI / 2; swing = u; }
    else { x = xa; ry = -Math.PI / 2 - Math.PI * ease((u - 21.2) / 0.8); }
    W.worker.position.set(x, 0, 1.75); W.worker.rotation.y = ry;
    const a = swing ? Math.sin(swing * Math.PI * 1.9) * 0.45 : 0;
    W.limbs.l.rotation.x = a; W.limbs.r.rotation.x = -a; W.limbs.la.rotation.x = -a * 0.8; W.limbs.ra.rotation.x = a * 0.8;
    W.worker.position.y = swing ? Math.abs(Math.sin(swing * Math.PI * 1.9)) * 0.025 : 0;
    W.beltTexture.offset.x = -(t * 0.7) % 1;
    if (!userMoved) {                                   // gentle idle drift; stops for good on first touch
      _off.setFromSphericalCoords(camBase.radius, camBase.phi + Math.sin(t * 0.09) * 0.025, camBase.theta + Math.sin(t * 0.11) * 0.06);
      W.camera.position.copy(W.controls.target).add(_off);
    }
  }
  ambient(0);

  // ------------------------------------------------ render loop, paused when hidden / offscreen
  const v = new THREE.Vector3();
  function place(el, anchor) {
    v.copy(anchor).project(W.camera);
    el.style.visibility = v.z > 1 ? 'hidden' : '';
    el.style.transform = `translate(-50%,-100%) translate(${(v.x * 0.5 + 0.5) * container.clientWidth}px,${(-v.y * 0.5 + 0.5) * container.clientHeight}px)`;
  }
  let lastT = performance.now() / 1000, lastBlink = null;
  function frame() {
    const t = performance.now() / 1000, dt = Math.min(0.25, t - lastT); lastT = t;
    measure(dt);
    tweens = tweens.filter(tw => {
      const k = clamp01((t - tw.t0) / tw.dur);
      if (t < tw.t0) return true;
      tw.fn(k); return k < 1;
    });
    const animating = tweens.length > 0 || truckMove !== 0;
    if (ambientOn()) { ambient(t); needsRender = true; }
    // status lights: blink while disrupted; steady under reduced motion (no flashing)
    const on = reduced() ? true : Math.sin(t * 6) > 0;
    const key = `${blink.transport && on}|${blink.supplier && on}|${truckMove}`;
    if (key !== lastBlink) {
      lastBlink = key; needsRender = true;
      const tr = blink.transport && on, su = blink.supplier && on;
      W.M.hazard.emissiveIntensity = tr ? 2.4 : 0;
      W.M.signal.emissiveIntensity = tr ? 2.4 : 0;
      W.signalGlow.material.opacity = tr ? 0.75 : 0;
      W.M.beacon.emissiveIntensity = su ? 2.6 : (blink.supplier ? 0.35 : 0);
      W.beaconGlow.material.opacity = su ? 0.9 : 0;
      W.M.headlight.emissiveIntensity = truckMove !== 0 ? 2.2 : 0.15;
      W.M.reverse.emissiveIntensity = truckMove < 0 ? 2.2 : 0;
    }
    if (W.controls.update()) needsRender = true;
    if (W.world.sync()) needsRender = true;
    if (W.cargoUnits.sync()) needsRender = true;
    if (needsRender || animating || !Q.done) {
      W.renderer.render(W.scene, W.camera);
      place(labels.stock, W.anchors.stock); place(labels.order, W.anchors.order); place(labels.unmet, W.anchors.unmet);
      needsRender = false;
    }
  }
  let running = false, onScreen = true;
  function updateRun() {
    const run = onScreen && !document.hidden;
    if (run === running) return;
    running = run;
    W.renderer.setAnimationLoop(run ? frame : null);
    if (run) { lastT = performance.now() / 1000; needsRender = true; if (!Q.done) { Q.warm = Math.max(Q.warm, 10); } }
  }
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(es => { onScreen = es[es.length - 1].isIntersecting; updateRun(); }).observe(container);
  }
  document.addEventListener('visibilitychange', updateRun);
  updateRun();
  if (RM.addEventListener) RM.addEventListener('change', () => { if (reduced() && last) snap(last); ambient(0); needsRender = true; });

  return {
    setState: snap, playWeek,
    get state() { return last; },
    get quality() {
      const r = W.renderer.info.render;
      return { tier: Q.tier, samples: Q.samples.slice(), settled: Q.done || Q.locked, ambient: ambientOn(),
               pixelRatio: W.renderer.getPixelRatio(), shadows: W.renderer.shadowMap.enabled, drawCalls: r.calls, triangles: r.triangles, framesRendered: r.frame };
    },
    setQuality(t) { Q.locked = true; applyTier(Math.max(0, Math.min(2, t | 0))); if (!ambientOn()) ambient(0); },
  };
}

try {
  const el = document.getElementById('scene3d');
  const probe = document.createElement('canvas');
  const gl = probe.getContext('webgl2') || probe.getContext('webgl');
  if (!el || !gl) throw new Error('no-webgl');
  window.W3D = start(el);
  document.body.classList.add('has-3d');
  window.dispatchEvent(new Event('w3d-ready'));
} catch (e) {
  // No WebGL (or a driver problem): keep the page exactly as it works without 3D.
  const box = document.getElementById('scene3dCard');
  if (box) box.classList.add('hide');
  if (e && e.message !== 'no-webgl') throw e;   // real bugs still reach the error reporter
}
