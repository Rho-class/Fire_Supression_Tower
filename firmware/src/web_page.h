// Control page served by the ESP32 at http://192.168.4.1 (or http://stepper.local
// when joined to your home Wi-Fi). Talks to the ESP32 over a WebSocket on port 81
// using the same one-letter commands as the USB serial port.
#pragma once

const char WEB_PAGE[] PROGMEM = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Stepper control</title>
<style>
  :root { --bg:#f6f5f1; --card:#fff; --ink:#22211e; --muted:#6b6a64; --line:#dddbd3;
          --accent:#2f6fd0; --on:#1d9e75; --warn:#d85a30; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#1b1b19; --card:#262624; --ink:#eceae3; --muted:#a3a198; --line:#3b3a36;
            --accent:#6ea4f0; --on:#3ccf9b; --warn:#f0866a; } }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body { margin:0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; background:var(--bg);
         color:var(--ink); padding:16px; user-select:none; -webkit-user-select:none; }
  header { display:flex; justify-content:space-between; align-items:center; max-width:720px; margin:0 auto 12px; }
  h1 { font-size:20px; font-weight:600; margin:0; }
  #conn { font-size:13px; padding:4px 10px; border-radius:999px; background:var(--warn); color:#fff; }
  #conn.ok { background:var(--on); }
  main { display:grid; gap:12px; max-width:720px; margin:0 auto; grid-template-columns:1fr 1fr; }
  @media (max-width:560px) { main { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; }
  .card h2 { font-size:15px; margin:0 0 2px; font-weight:600; }
  .state { font-size:13px; color:var(--muted); margin-bottom:10px; min-height:18px; }
  .state b { color:var(--ink); font-weight:600; }
  .row { display:flex; gap:8px; margin-top:8px; }
  button { flex:1; font:inherit; font-size:15px; padding:14px 8px; border-radius:10px; border:1px solid var(--line);
           background:var(--bg); color:var(--ink); touch-action:none; cursor:pointer; }
  button.small { padding:10px 8px; font-size:14px; }
  button.active { background:var(--accent); border-color:var(--accent); color:#fff; }
  #pumpBtn.on { background:var(--on); border-color:var(--on); color:#fff; }
  #stopBtn { background:var(--warn); border-color:var(--warn); color:#fff; font-weight:600; }
  .wide { grid-column:1 / -1; }
  kbd { font-family:inherit; font-size:12px; border:1px solid var(--line); border-radius:4px; padding:0 4px; color:var(--muted); }
  button.active kbd, #pumpBtn.on kbd, #stopBtn kbd { color:#fff; border-color:rgba(255,255,255,.7); }
  .hint { font-size:12px; color:var(--muted); text-align:center; margin:12px auto 0; max-width:720px; line-height:1.7; }
</style>
</head>
<body>
<header><h1>Stepper control</h1><span id="conn">Connecting…</span></header>
<main>
  <section class="card">
    <h2>Motor 1</h2>
    <div class="state" id="m1state">–</div>
    <div class="row">
      <button data-motor="0" data-cmd="L">◀ <kbd>A</kbd></button>
      <button data-motor="0" data-cmd="R"><kbd>D</kbd> ▶</button>
    </div>
    <div class="row">
      <button class="small" data-tap="D">Slower <kbd>S</kbd></button>
      <button class="small" data-tap="U">Faster <kbd>W</kbd></button>
    </div>
  </section>
  <section class="card">
    <h2>Motor 2</h2>
    <div class="state" id="m2state">–</div>
    <div class="row">
      <button data-motor="1" data-cmd="B">◀ <kbd>←</kbd></button>
      <button data-motor="1" data-cmd="F"><kbd>→</kbd> ▶</button>
    </div>
    <div class="row">
      <button class="small" data-tap="M">Slower <kbd>↓</kbd></button>
      <button class="small" data-tap="P">Faster <kbd>↑</kbd></button>
    </div>
  </section>
  <section class="card wide">
    <div class="row" style="margin-top:0">
      <button id="pumpBtn" data-tap="E">Pump: off <kbd>E</kbd></button>
      <button id="stopBtn" data-tap="S">Stop all <kbd>Space</kbd></button>
    </div>
  </section>
</main>
<p class="hint">Hold a turn button (or key) to move; let go to stop.<br>
Motors stop if this page loses focus or the connection drops; the pump switches off if the connection drops.</p>
<script>
const TURN_KEYS = { a:[0,"L"], d:[0,"R"], ArrowLeft:[1,"B"], ArrowRight:[1,"F"] };
const TAP_KEYS  = { w:"U", s:"D", ArrowUp:"P", ArrowDown:"M", e:"E", " ":"S" };
const STOP_CMD  = ["X", "Y"];
const held = [null, null];
let ws = null;

function send(text) { if (ws && ws.readyState === 1) ws.send(text); }

function press(motor, cmd) {
  if (held[motor] === cmd) return;
  held[motor] = cmd;
  send(cmd);
  highlight();
}
function release(motor, cmd) {
  if (held[motor] !== cmd) return;
  held[motor] = null;
  send(STOP_CMD[motor]);
  highlight();
}
function releaseAll() {
  for (let m = 0; m < 2; m++) if (held[m]) { held[m] = null; send(STOP_CMD[m]); }
  highlight();
}
function highlight() {
  document.querySelectorAll("button[data-cmd]").forEach(b => {
    b.classList.toggle("active", held[+b.dataset.motor] === b.dataset.cmd);
  });
}

function keyName(e) { return e.key.length === 1 ? e.key.toLowerCase() : e.key; }
addEventListener("keydown", e => {
  const k = keyName(e);
  if (TURN_KEYS[k]) { e.preventDefault(); press(...TURN_KEYS[k]); }
  else if (TAP_KEYS[k]) { e.preventDefault(); if (!e.repeat) send(TAP_KEYS[k]); }
});
addEventListener("keyup", e => {
  const k = keyName(e);
  if (TURN_KEYS[k]) { e.preventDefault(); release(...TURN_KEYS[k]); }
});
addEventListener("blur", releaseAll);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseAll(); });

document.querySelectorAll("button[data-cmd]").forEach(b => {
  const m = +b.dataset.motor, c = b.dataset.cmd;
  b.addEventListener("pointerdown", e => { b.setPointerCapture(e.pointerId); press(m, c); });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach(t => b.addEventListener(t, () => release(m, c)));
  b.addEventListener("contextmenu", e => e.preventDefault());
});
document.querySelectorAll("button[data-tap]").forEach(b => {
  b.addEventListener("click", () => send(b.dataset.tap));
});

// Heartbeat: tells the ESP32 the page is still here, and keeps held motors moving.
setInterval(() => send("." + held.filter(Boolean).join("")), 100);

const WORDS = { left:"turning ◀", right:"turning ▶", stopped:"stopped" };
function showStatus(text) {
  const p = text.trim().split(/\s+/);
  if (p.length !== 6 || p[0] !== "STATUS") return;
  document.getElementById("m1state").innerHTML = "<b>" + (WORDS[p[1]] || p[1]) + "</b> · " + p[2] + " RPM";
  document.getElementById("m2state").innerHTML = "<b>" + (WORDS[p[3]] || p[3]) + "</b> · " + p[4] + " RPM";
  const pump = document.getElementById("pumpBtn");
  pump.classList.toggle("on", p[5] === "on");
  pump.firstChild.textContent = "Pump: " + (p[5] === "on" ? "ON " : "off ");
}

function connect() {
  const conn = document.getElementById("conn");
  ws = new WebSocket("ws://" + location.hostname + ":81/");
  ws.onopen = () => { conn.textContent = "Connected"; conn.classList.add("ok"); send("?"); };
  ws.onmessage = ev => showStatus(String(ev.data));
  ws.onclose = () => {
    conn.textContent = "Disconnected – retrying"; conn.classList.remove("ok");
    held[0] = held[1] = null; highlight();
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
}
connect();
</script>
</body>
</html>)HTML";
