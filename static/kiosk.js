(function () {
  "use strict";

  // ── DOM refs ──
  const video = document.getElementById("camera");
  const canvas = document.getElementById("snapshot");
  const ctx = canvas.getContext("2d");
  const resultCard = document.getElementById("result-card");
  const resultIcon = document.getElementById("result-icon");
  const resultName = document.getElementById("result-name");
  const resultAction = document.getElementById("result-action");
  const resultTime = document.getElementById("result-time");
  const scanStatus = document.getElementById("scan-status");
  const challengePrompt = document.getElementById("challenge-prompt");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const clockEl = document.getElementById("clock");

  // ── State ──
  const CAPTURE_INTERVAL_MS = 2000;
  const VERIFY_INTERVAL_MS = 800;   // Faster capture during verification
  const CHALLENGE_INTERVAL_MS = 200; // Fast capture during liveness challenge
  const RESULT_DISPLAY_MS = 4000;
  const RECOGNIZED_LOCKOUT_MS = 2000; // suppress stale responses after a success
  let capturing = false;
  let paused = false;
  let captureTimer = null;
  let verifying = false;
  let challenging = false;
  let recognizedAt = 0;
  let inFlight = false;

  // ── Clock ──
  function updateClock() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ── Camera init ──
  async function initCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 320 }, height: { ideal: 240 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      setStatus("idle", "Ready — look at the camera");
      startCapture();
    } catch (err) {
      setStatus("error", "Camera access denied: " + err.message);
    }
  }

  // ── Capture loop ──
  function startCapture() {
    if (capturing) return;
    capturing = true;
    captureTimer = setInterval(captureAndRecognize, CAPTURE_INTERVAL_MS);
  }

  function setCaptureSpeed(mode) {
    // mode: "normal", "verify", "challenge"
    var interval = CAPTURE_INTERVAL_MS;
    if (mode === "verify") interval = VERIFY_INTERVAL_MS;
    if (mode === "challenge") interval = CHALLENGE_INTERVAL_MS;

    if (captureTimer) clearInterval(captureTimer);
    captureTimer = setInterval(captureAndRecognize, interval);
    verifying = mode === "verify";
    challenging = mode === "challenge";

    // Show/hide challenge prompt
    if (mode !== "challenge" && challengePrompt) {
      challengePrompt.className = "challenge-prompt hidden";
    }
  }

  function setVerifyMode(active) {
    setCaptureSpeed(active ? "verify" : "normal");
  }

  function pauseCapture() {
    paused = true;
    setCaptureSpeed("normal");
  }

  function resumeCapture() {
    paused = false;
    hideResult();
    setStatus("idle", "Ready — look at the camera");
  }

  async function captureAndRecognize() {
    if (paused || inFlight) return;
    inFlight = true;

    // Draw current video frame to hidden canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    const base64Data = dataUrl.split(",")[1];

    setStatus("scanning", "Scanning...");

    try {
      const resp = await fetch("/api/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Data }),
      });
      const data = await resp.json();
      handleResult(data);
    } catch (err) {
      setStatus("error", "Server error: " + err.message);
    } finally {
      inFlight = false;
    }
  }

  // ── Handle recognition result ──
  function handleResult(data) {
    // pauseCapture() stops new captures but in-flight fetches still arrive.
    // After a successful recognition the server engages the cooldown, so
    // those stale responses come back as "cooldown" and would overwrite the
    // success card. Suppress non-success statuses briefly to let the success
    // card stay visible.
    if (data.status !== "recognized" && Date.now() - recognizedAt < RECOGNIZED_LOCKOUT_MS) {
      return;
    }

    switch (data.status) {
      case "recognized":
        recognizedAt = Date.now();
        setCaptureSpeed("normal");
        showResult({
          icon: data.is_clock_in ? "\u2713" : "\u2190",
          name: data.identity,
          action: data.is_clock_in ? "Clocked In" : "Clocked Out",
          cardClass: data.is_clock_in ? "result-clock-in" : "result-clock-out",
        });
        break;

      case "verifying":
        setVerifyMode(true);
        setStatus("scanning",
          "Verifying " + data.identity + "... (" +
          data.consensus_progress + "/" + data.consensus_required + ")");
        return;

      case "liveness_challenge":
        setCaptureSpeed("challenge");
        // Show challenge prompt overlay
        if (challengePrompt) {
          challengePrompt.textContent = data.challenge_instruction;
          challengePrompt.className = "challenge-prompt";
        }
        setStatus("challenge",
          data.identity + " — " + data.challenge_instruction +
          " (" + Math.ceil(data.challenge_time_remaining) + "s)");
        return;

      case "cooldown":
        showResult({
          icon: "\u23F3",
          name: data.identity,
          action: data.message,
          cardClass: "result-cooldown",
        });
        break;

      case "spoof_detected":
        setCaptureSpeed("normal");
        showResult({
          icon: "\u26D4",
          name: "",
          action: "Spoof Detected",
          cardClass: "result-spoof",
        });
        break;

      case "unknown":
        setCaptureSpeed("normal");
        showResult({
          icon: "?",
          name: "",
          action: "Face Not Recognized",
          cardClass: "result-unknown",
        });
        break;

      case "no_face":
        setCaptureSpeed("normal");
        // Stay in scanning mode, don't show a result card
        setStatus("idle", "No face detected — position your face in the frame");
        return;

      case "multiple_faces":
        showResult({
          icon: "\u26A0",
          name: "",
          action: "One person at a time, please",
          cardClass: "result-warning",
        });
        break;

      case "error":
        setStatus("error", data.message || "Recognition error");
        return;

      default:
        return;
    }
  }

  function showResult({ icon, name, action, cardClass }) {
    pauseCapture();

    resultIcon.textContent = icon;
    resultName.textContent = name;
    resultAction.textContent = action;
    resultTime.textContent = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    // Apply card color class
    resultCard.className = "result-card " + cardClass;
    scanStatus.textContent = "";

    // Auto-reset after display period
    setTimeout(resumeCapture, RESULT_DISPLAY_MS);
  }

  function hideResult() {
    resultCard.className = "result-card hidden";
    scanStatus.textContent = "Position your face in the frame";
  }

  // ── Status bar ──
  function setStatus(state, text) {
    statusText.textContent = text;
    statusDot.className = "status-dot dot-" + state;
  }

  // ── Boot ──
  fetch("/api/health").then(function (r) { return r.json(); }).then(function (h) {
    var lbl = document.getElementById("store-label");
    if (lbl) lbl.textContent = h.store_id || "";
  }).catch(function () {});
  initCamera();
})();
