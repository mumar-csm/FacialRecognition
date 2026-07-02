(function () {
  "use strict";

  // ── DOM refs ──
  var video = document.getElementById("camera");
  var canvas = document.getElementById("snapshot");
  var ctx = canvas.getContext("2d");
  var enrollBtn = document.getElementById("enroll-btn");
  var firstNameInput = document.getElementById("first-name");
  var lastNameInput = document.getElementById("last-name");
  var posIdInput = document.getElementById("pos-employee-id");
  var storeLabel = document.getElementById("store-label");
  var pinGate = document.getElementById("pin-gate");
  var pinGateInput = document.getElementById("pin-gate-input");
  var pinGateSubmit = document.getElementById("pin-gate-submit");
  var pinGateError = document.getElementById("pin-gate-error");
  var mainContainer = document.getElementById("main-container");

  // ── PIN cache (per-tab session only) ──
  var PIN_STORAGE_KEY = "kiosk_manager_pin";
  var cachedPin = null;
  function getCachedPin() {
    try { return sessionStorage.getItem(PIN_STORAGE_KEY); } catch (e) { return null; }
  }
  function setCachedPin(pin) {
    cachedPin = pin;
    try { sessionStorage.setItem(PIN_STORAGE_KEY, pin); } catch (e) { /* non-fatal */ }
  }
  function clearCachedPin() {
    cachedPin = null;
    try { sessionStorage.removeItem(PIN_STORAGE_KEY); } catch (e) { /* non-fatal */ }
  }
  var resultCard = document.getElementById("result-card");
  var resultIcon = document.getElementById("result-icon");
  var resultName = document.getElementById("result-name");
  var resultAction = document.getElementById("result-action");
  var lightingWarning = document.getElementById("lighting-warning");
  var nameHint = document.getElementById("name-hint");
  var statusDot = document.getElementById("status-dot");
  var statusText = document.getElementById("status-text");

  // ── Config ──
  var RESULT_DISPLAY_MS = 4000;
  var LIGHTING_CHECK_MS = 2000;
  var BRIGHTNESS_MIN = 50;
  var BRIGHTNESS_MAX = 220;

  var cameraReady = false;
  var lightingOk = true;

  // ── Server config (store label) ──
  async function loadServerConfig() {
    try {
      var health = await fetch("/api/health").then(function (r) { return r.json(); });
      if (storeLabel) storeLabel.textContent = health.store_id || "";
      return health;
    } catch (e) {
      return {};
    }
  }

  // ── Verify PIN against server ──
  async function verifyPin(pin) {
    try {
      var resp = await fetch("/api/verify-pin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pin: pin }),
      });
      var data = await resp.json();
      return data;
    } catch (e) {
      return { valid: false, protected: true };
    }
  }

  // ── PIN gate flow ──
  function showPinGate(errMsg) {
    pinGate.classList.remove("hidden");
    mainContainer.classList.add("hidden");
    pinGateError.textContent = errMsg || "";
    pinGateInput.value = "";
    setTimeout(function () { pinGateInput.focus(); }, 50);
  }

  function hidePinGate() {
    pinGate.classList.add("hidden");
    mainContainer.classList.remove("hidden");
  }

  async function handlePinSubmit() {
    var entered = pinGateInput.value.trim();
    if (!entered) {
      pinGateError.textContent = "Please enter a PIN.";
      return;
    }
    pinGateSubmit.disabled = true;
    pinGateSubmit.textContent = "Verifying...";
    var result = await verifyPin(entered);
    pinGateSubmit.disabled = false;
    pinGateSubmit.textContent = "Unlock";
    if (result.valid) {
      setCachedPin(entered);
      hidePinGate();
      bootMain();
    } else {
      pinGateError.textContent = "Incorrect PIN.";
      pinGateInput.value = "";
      pinGateInput.focus();
    }
  }

  pinGateSubmit.addEventListener("click", handlePinSubmit);
  pinGateInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") handlePinSubmit();
  });

  // ── Camera init ──
  async function initCamera() {
    try {
      var stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      cameraReady = true;
      updateButtonState();
      setStatus("idle", "Ready \u2014 enter name and capture");
      setInterval(checkLighting, LIGHTING_CHECK_MS);
    } catch (err) {
      setStatus("error", "Camera access denied: " + err.message);
    }
  }

  // ── Brightness check on center region ──
  function checkLighting() {
    if (!cameraReady) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Sample the center 40% of the frame (where the face guide is)
    var cw = canvas.width;
    var ch = canvas.height;
    var regionW = Math.floor(cw * 0.4);
    var regionH = Math.floor(ch * 0.4);
    var startX = Math.floor((cw - regionW) / 2);
    var startY = Math.floor((ch - regionH) / 2);

    var imageData = ctx.getImageData(startX, startY, regionW, regionH);
    var data = imageData.data;
    var totalBrightness = 0;
    var pixelCount = data.length / 4;

    for (var i = 0; i < data.length; i += 4) {
      totalBrightness += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    var avgBrightness = totalBrightness / pixelCount;

    if (avgBrightness < BRIGHTNESS_MIN) {
      lightingWarning.textContent = "Too dark \u2014 find better lighting";
      lightingWarning.className = "lighting-warning";
      lightingOk = false;
    } else if (avgBrightness > BRIGHTNESS_MAX) {
      lightingWarning.textContent = "Too bright \u2014 reduce glare";
      lightingWarning.className = "lighting-warning";
      lightingOk = false;
    } else {
      lightingWarning.className = "lighting-warning hidden";
      lightingOk = true;
    }

    updateButtonState();
  }

  // ── Name sanitization (mirrors server-side logic) ──
  function sanitizeName(raw) {
    return raw.replace(/[^a-zA-Z\s-]/g, "").trim().toLowerCase().replace(/\s+/g, "_");
  }

  // ── POS ID validation (mirrors server-side regex in kiosk_server.py) ──
  var POS_ID_RE = /^\d{7}$/;
  function isValidPosId(raw) {
    return POS_ID_RE.test(raw);
  }

  // ── Button state + name/POS-ID validation ──
  function updateButtonState() {
    var rawFirst = firstNameInput.value.trim();
    var rawLast = lastNameInput.value.trim();
    var cleanFirst = sanitizeName(rawFirst);
    var cleanLast = sanitizeName(rawLast);
    var hasName = cleanFirst && cleanLast;
    var rawPosId = posIdInput.value.trim();
    var posIdOk = isValidPosId(rawPosId);

    // Build the hint: name + POS ID feedback, whichever applies.
    var hintParts = [];
    if ((rawFirst && rawFirst !== cleanFirst.replace(/_/g, " ")) ||
        (rawLast && rawLast !== cleanLast.replace(/_/g, " "))) {
      hintParts.push("Letters, spaces, and hyphens only. Will save as: " + cleanFirst + "_" + cleanLast);
    } else if (hasName) {
      hintParts.push("Will save as: " + cleanFirst + "_" + cleanLast);
    }
    if (rawPosId && !posIdOk) {
      hintParts.push("POS Employee ID must be exactly 7 digits.");
    }
    if (hintParts.length) {
      nameHint.textContent = hintParts.join(" — ");
      nameHint.className = "name-hint";
    } else {
      nameHint.className = "name-hint hidden";
    }

    enrollBtn.disabled = !cameraReady || !lightingOk || !hasName || !posIdOk;
  }

  firstNameInput.addEventListener("input", updateButtonState);
  lastNameInput.addEventListener("input", updateButtonState);
  posIdInput.addEventListener("input", updateButtonState);

  // ── Enroll ──
  enrollBtn.addEventListener("click", async function () {
    var firstName = firstNameInput.value.trim();
    var lastName = lastNameInput.value.trim();
    var posId = posIdInput.value.trim();

    if (!firstName || !lastName) {
      showResult({ icon: "\u26A0", name: "", action: "Enter first and last name", cardClass: "result-warning" });
      return;
    }
    if (!isValidPosId(posId)) {
      showResult({ icon: "\u26A0", name: "", action: "Enter a valid POS Employee ID", cardClass: "result-warning" });
      return;
    }

    enrollBtn.disabled = true;
    enrollBtn.textContent = "Processing...";
    setStatus("scanning", "Capturing and enrolling...");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    var dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    var base64Data = dataUrl.split(",")[1];

    try {
      var pinValue = cachedPin || getCachedPin();
      var resp = await fetch("/api/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64Data,
          first_name: firstName,
          last_name: lastName,
          pos_employee_id: posId,
          pin: pinValue,
        }),
      });
      var data = await resp.json();
      handleResult(data);
    } catch (err) {
      showResult({ icon: "\u2716", name: "", action: "Server error: " + err.message, cardClass: "result-spoof" });
    }

    enrollBtn.textContent = "Capture & Enroll";
    updateButtonState();
  });

  // ── Handle result ──
  function handleResult(data) {
    switch (data.status) {
      case "enrolled":
        showResult({
          icon: "\u2713",
          name: data.employee_name,
          action: "Enrolled Successfully!",
          cardClass: "result-clock-in",
        });
        firstNameInput.value = "";
        lastNameInput.value = "";
        posIdInput.value = "";
        setStatus("idle", "Ready \u2014 enter name and capture");
        break;

      case "unauthorized":
        // Cached PIN was rejected (likely rotated server-side). Re-gate the page.
        clearCachedPin();
        showResult({ icon: "\u26D4", name: "", action: "Session expired \u2014 re-enter PIN", cardClass: "result-spoof" });
        setTimeout(function () { showPinGate("Please re-enter the manager PIN."); }, 1200);
        break;

      case "no_face":
        showResult({ icon: "?", name: "", action: data.message, cardClass: "result-unknown" });
        break;

      case "multiple_faces":
        showResult({ icon: "\u26A0", name: "", action: data.message, cardClass: "result-warning" });
        break;

      case "spoof_detected":
        showResult({ icon: "\u26D4", name: "", action: data.message, cardClass: "result-spoof" });
        break;

      case "low_light":
        // Not a spoof \u2014 recoverable lighting problem. Amber warning, not red.
        showResult({ icon: "\u26A0", name: "", action: data.message, cardClass: "result-warning" });
        break;

      case "duplicate_face":
        // Same physical face already belongs to an active employee. Hard stop \u2014
        // flash a prominent red error and keep the form filled so the manager
        // can review who it matched.
        showResult({
          icon: "\u26D4",
          name: "This Face Already Exists",
          action: data.message || "This face is already enrolled to another employee.",
          cardClass: "result-duplicate",
        });
        setStatus("error", "Enrollment blocked \u2014 duplicate face");
        break;

      default:
        showResult({ icon: "\u2716", name: "", action: data.message || "Unknown error", cardClass: "result-spoof" });
        break;
    }
  }

  // ── Result display ──
  function showResult(opts) {
    resultIcon.textContent = opts.icon;
    resultName.textContent = opts.name;
    resultAction.textContent = opts.action;
    resultCard.className = "result-card " + opts.cardClass;
    setTimeout(hideResult, RESULT_DISPLAY_MS);
  }

  function hideResult() {
    resultCard.className = "result-card hidden";
  }

  // ── Status bar ──
  function setStatus(state, text) {
    statusText.textContent = text;
    statusDot.className = "status-dot dot-" + state;
  }

  // ── Boot ──
  async function bootMain() {
    await loadServerConfig();
    initCamera();
  }

  async function boot() {
    var health = await loadServerConfig();
    if (!health.enrollment_protected) {
      // No PIN configured — open enrollment, skip gate entirely
      hidePinGate();
      bootMain();
      return;
    }
    // Try cached PIN first (manager unlocked earlier in this tab session)
    var existing = getCachedPin();
    if (existing) {
      var result = await verifyPin(existing);
      if (result.valid) {
        cachedPin = existing;
        hidePinGate();
        bootMain();
        return;
      }
      clearCachedPin();
    }
    showPinGate();
  }

  boot();
})();
