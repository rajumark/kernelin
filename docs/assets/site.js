// Kernelin site: shared behaviour for the landing page and the docs.
(function () {
  "use strict";

  // ---------- tiny syntax highlighter for Kotlin, WGSL, Gradle and shell ----------
  var KEYWORDS = {
    kotlin: "fun val var if else when for while do in until downTo step return break continue true false null object class package import plugins id version implementation suspend is as",
    wgsl: "fn let var if else for while loop return break continue struct true false",
  };
  var TYPES = "Int Float Boolean Unit FloatArray IntArray String Double Long KernelSource GpuData WebGpu i32 u32 f32 bool array vec3 uniform storage read read_write private";
  function escapeHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function highlight(code, rawLang) {
    var lang = rawLang === "wgsl" ? "wgsl" : "kotlin";
    var kw = new Set(KEYWORDS[lang].split(" "));
    var ty = new Set(TYPES.split(" "));
    var re = /(\/\/[^\n]*|#[^\n]*$)|("(?:[^"\\\n]|\\.)*")|(@[A-Za-z_]\w*)|(\b\d+(?:\.\d+)?(?:[fiuLe]\b|f)?)|([A-Za-z_]\w*)(?=\s*\()|([A-Za-z_]\w*)/gm;
    var out = "", last = 0, m;
    while ((m = re.exec(code))) {
      out += escapeHtml(code.slice(last, m.index));
      var t = escapeHtml(m[0]);
      if (m[1]) out += ((rawLang === "bash") !== (m[1][0] === "#")) ? t : '<span class="tok-c">' + t + "</span>";
      else if (m[2]) out += '<span class="tok-s">' + t + "</span>";
      else if (m[3]) out += '<span class="tok-a">' + t + "</span>";
      else if (m[4]) out += '<span class="tok-n">' + t + "</span>";
      else if (m[5]) out += kw.has(m[5]) ? '<span class="tok-k">' + t + "</span>" : ty.has(m[5]) ? '<span class="tok-t">' + t + "</span>" : '<span class="tok-f">' + t + "</span>";
      else out += kw.has(m[6]) ? '<span class="tok-k">' + t + "</span>" : ty.has(m[6]) ? '<span class="tok-t">' + t + "</span>" : t;
      last = re.lastIndex;
    }
    return out + escapeHtml(code.slice(last));
  }
  window.KernelinSite = { highlight: highlight, escapeHtml: escapeHtml };

  document.querySelectorAll("code[data-lang]").forEach(function (el) {
    el.innerHTML = highlight(el.textContent, el.getAttribute("data-lang"));
  });

  // ---------- mobile menu ----------
  var nav = document.querySelector(".nav");
  var menuBtn = document.querySelector(".menu-btn");
  if (nav && menuBtn) {
    menuBtn.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("ul a").forEach(function (a) {
      a.addEventListener("click", function () { nav.classList.remove("open"); menuBtn.setAttribute("aria-expanded", "false"); });
    });
  }

  // ---------- copy buttons ----------
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var target = document.getElementById(btn.getAttribute("data-copy"));
      var text = target ? target.textContent : btn.getAttribute("data-copy-text") || "";
      var label = btn.textContent;
      (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(function () {
        btn.textContent = "Copied";
      }, function () {
        btn.textContent = "Press ⌘C";
      }).then(function () { setTimeout(function () { btn.textContent = label; }, 1400); });
    });
  });

  // ---------- footer: local time and WebGPU support ----------
  var clock = document.getElementById("clock");
  if (clock) {
    var tick = function () {
      var d = new Date();
      clock.textContent = String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
    };
    tick();
    setInterval(tick, 15000);
  }
  var gpuState = document.getElementById("gpu-state");
  if (gpuState) {
    var has = typeof navigator !== "undefined" && !!navigator.gpu;
    gpuState.textContent = has ? "WebGPU ready in this browser" : "No WebGPU in this browser";
    var dot = document.getElementById("gpu-dot");
    if (dot && has) dot.classList.add("on");
  }

  // ---------- "logs": private build log behind a PIN ----------
  // Note: this is a light gate for a static site, not real security. The PIN is stored as a SHA-256 hash.
  var PIN_HASH = "0ffe1abd1a08215353c233d6e009613e95eec4253832a761af28ff37ac5a150c";
  var gate = document.getElementById("gate");
  if (!gate) return;
  var pin = document.getElementById("pin");
  var cells = Array.prototype.slice.call(gate.querySelectorAll(".pins .cell"));
  var msg = gate.querySelector(".msg");
  var lastFocus = null, checking = false;

  function paint() {
    var n = pin.value.length, focused = document.activeElement === pin;
    cells.forEach(function (c, i) {
      c.classList.toggle("filled", i < n);
      c.classList.toggle("active", focused && (i === n || (n === cells.length && i === n - 1)));
    });
  }
  function openGate() {
    lastFocus = document.activeElement;
    gate.classList.add("open");
    gate.classList.remove("bad");
    msg.textContent = "";
    pin.value = "";
    setTimeout(function () { pin.focus(); paint(); }, 30);
  }
  function closeGate() {
    gate.classList.remove("open");
    if (location.hash === "#logs") history.replaceState(null, "", location.pathname + location.search);
    if (lastFocus) lastFocus.focus();
  }
  function sha256(text) {
    if (!(window.crypto && crypto.subtle)) return Promise.resolve(null);
    return crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)).then(function (buf) {
      return Array.prototype.map.call(new Uint8Array(buf), function (b) { return b.toString(16).padStart(2, "0"); }).join("");
    });
  }
  function check() {
    if (checking || pin.value.length < cells.length) return;
    checking = true;
    sha256(pin.value).then(function (hash) {
      checking = false;
      if (hash === PIN_HASH) {
        try { sessionStorage.setItem("kernelin-logs", "ok"); } catch (e) {}
        location.href = "logs.html";
        return;
      }
      gate.classList.remove("bad");
      void gate.offsetWidth;
      gate.classList.add("bad");
      msg.textContent = hash === null ? "Open this page over https to unlock." : "Wrong PIN. Try again.";
      pin.value = "";
      pin.focus();
      paint();
    });
  }

  pin.addEventListener("input", function () {
    pin.value = pin.value.replace(/\D/g, "").slice(0, cells.length);
    if (pin.value) { gate.classList.remove("bad"); msg.textContent = ""; }
    paint();
    check();
  });
  pin.addEventListener("keydown", function (e) { if (e.key === "Enter") check(); });
  pin.addEventListener("focus", paint);
  pin.addEventListener("blur", paint);
  document.querySelectorAll("[data-open-logs]").forEach(function (b) { b.addEventListener("click", openGate); });
  gate.querySelector("[data-close]").addEventListener("click", closeGate);
  gate.addEventListener("click", function (e) { if (e.target === gate) closeGate(); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && gate.classList.contains("open")) closeGate(); });
  if (location.hash === "#logs") openGate();
})();
