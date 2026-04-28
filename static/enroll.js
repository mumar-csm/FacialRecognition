(function () {
  "use strict";

  // ── DOM refs ──
  var video = document.getElementById("camera");
  var canvas = document.getElementById("snapshot");
  var ctx = canvas.getContext("2d");
  var enrollBtn = document.getElementById("enroll-btn");
  var firstNameInput = document.getElementById("first-name");
  var lastNameInput = document.getElementById("last-name");
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

  // ── Button state + name validation ──
  function updateButtonState() {
    var rawFirst = firstNameInput.value.trim();
    var rawLast = lastNameInput.value.trim();
    var cleanFirst = sanitizeName(rawFirst);
    var cleanLast = sanitizeName(rawLast);
    var hasName = cleanFirst && cleanLast;

    // Show hint if characters were stripped
    if ((rawFirst && rawFirst !== cleanFirst.replace(/_/g, " ")) ||
        (rawLast && rawLast !== cleanLast.replace(/_/g, " "))) {
      nameHint.textContent = "Letters, spaces, and hyphens only. Will save as: " + cleanFirst + "_" + cleanLast;
      nameHint.className = "name-hint";
    } else if (hasName) {
      nameHint.textContent = "Will save as: " + cleanFirst + "_" + cleanLast;
      nameHint.className = "name-hint";
    } else {
      nameHint.className = "name-hint hidden";
    }

    enrollBtn.disabled = !cameraReady || !lightingOk || !hasName;
  }

  firstNameInput.addEventListener("input", updateButtonState);
  lastNameInput.addEventListener("input", updateButtonState);

  // ── Enroll ──
  enrollBtn.addEventListener("click", async function () {
    var firstName = firstNameInput.value.trim();
    var lastName = lastNameInput.value.trim();

    if (!firstName || !lastName) {
      showResult({ icon: "\u26A0", name: "", action: "Enter first and last name", cardClass: "result-warning" });
      return;
    }

    enrollBtn.disabled = true;
    enrollBtn.textContent = "Processing...";
    setStatus("scanning", "Capturing and enrolling...");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    var dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    var base64Data = dataUrl.split(",")[1];

    try {
      var resp = await fetch("/api/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Data, first_name: firstName, last_name: lastName }),
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
        setStatus("idle", "Ready \u2014 enter name and capture");
        break;

      case "already_exists":
        showResult({ icon: "\u26A0", name: "", action: data.message, cardClass: "result-warning" });
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
  initCamera();
})();
