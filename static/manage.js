(function () {
  "use strict";

  // ── DOM refs ──
  var pinGate         = document.getElementById("pin-gate");
  var pinGateInput    = document.getElementById("pin-gate-input");
  var pinGateSubmit   = document.getElementById("pin-gate-submit");
  var pinGateError    = document.getElementById("pin-gate-error");
  var mainContainer   = document.getElementById("main-container");
  var storeLabel      = document.getElementById("store-label");
  var tableWrapper    = document.getElementById("table-wrapper");
  var tbody           = document.getElementById("employees-tbody");
  var emptyState      = document.getElementById("empty-state");
  var statusDot       = document.getElementById("status-dot");
  var statusText      = document.getElementById("status-text");
  var modalBackdrop   = document.getElementById("modal-backdrop");
  var modalText       = document.getElementById("modal-text");
  var modalPhotoSlot  = document.getElementById("modal-photo-slot");
  var modalCancel     = document.getElementById("modal-cancel");
  var modalConfirm    = document.getElementById("modal-confirm");
  var photoBackdrop   = document.getElementById("photo-backdrop");
  var photoSlot       = document.getElementById("photo-slot");
  var photoName       = document.getElementById("photo-name");
  var photoIdEl       = document.getElementById("photo-id");
  var photoClose      = document.getElementById("photo-close");
  var toast           = document.getElementById("toast");

  // ── PIN cache (shared key with enroll page) ──
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

  // ── Helpers ──
  function setStatus(state, text) {
    statusText.textContent = text;
    statusDot.className = "status-dot dot-" + state;
  }

  function fmtName(raw) {
    return raw.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function fmtDate(ts) {
    if (!ts) return "—";
    return ts.slice(0, 10);
  }

  function showToast(message, kind) {
    toast.textContent = message;
    toast.className = "toast toast-" + (kind || "success");
    setTimeout(function () { toast.className = "toast hidden"; }, 3000);
  }

  // ── Server config ──
  async function loadServerConfig() {
    try {
      var health = await fetch("/api/health").then(function (r) { return r.json(); });
      if (storeLabel) storeLabel.textContent = health.store_id || "";
      return health;
    } catch (e) {
      return {};
    }
  }

  // ── PIN verify ──
  async function verifyPin(pin) {
    try {
      var resp = await fetch("/api/verify-pin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pin: pin }),
      });
      return await resp.json();
    } catch (e) {
      return { valid: false, protected: true };
    }
  }

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

  // ── Employee list ──
  async function loadEmployees() {
    setStatus("scanning", "Loading employees...");
    try {
      var data = await fetch("/api/employees").then(function (r) { return r.json(); });
      renderEmployees(data.employees || []);
      setStatus("idle", data.count + " employee" + (data.count === 1 ? "" : "s"));
    } catch (err) {
      setStatus("error", "Failed to load: " + err.message);
    }
  }

  function renderEmployees(employees) {
    tbody.innerHTML = "";
    if (employees.length === 0) {
      tableWrapper.style.display = "none";
      emptyState.style.display = "block";
      return;
    }
    emptyState.style.display = "none";
    tableWrapper.style.display = "block";
    employees.forEach(function (emp) {
      var tr = document.createElement("tr");
      var displayName = emp.name || fmtName(emp.id);

      var nameTd = document.createElement("td");
      nameTd.textContent = displayName;

      var idTd = document.createElement("td");
      idTd.textContent = emp.id;
      idTd.style.color = "#94a3b8";
      idTd.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
      idTd.style.fontSize = "13px";

      var posIdTd = document.createElement("td");
      posIdTd.textContent = emp.pos_employee_id || "—";
      posIdTd.style.color = emp.pos_employee_id ? "#e2e8f0" : "#64748b";
      posIdTd.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
      posIdTd.style.fontSize = "13px";
      if (!emp.pos_employee_id) {
        posIdTd.title = "No POS ID on file (enrolled before this field existed). Delete + re-enroll to set one.";
      }

      var dateTd = document.createElement("td");
      dateTd.textContent = fmtDate(emp.enrolled_at);

      var actionTd = document.createElement("td");
      actionTd.style.textAlign = "right";
      var actionWrap = document.createElement("div");
      actionWrap.className = "action-btns";

      var viewBtn = document.createElement("button");
      viewBtn.className = "action-btn view-btn";
      viewBtn.textContent = "View";
      viewBtn.disabled = !emp.has_photo;
      viewBtn.title = emp.has_photo ? "View enrollment photo" : "No photo on file";
      viewBtn.addEventListener("click", function () { openPhoto(emp); });

      var delBtn = document.createElement("button");
      delBtn.className = "action-btn delete-btn";
      delBtn.textContent = "Remove";
      delBtn.addEventListener("click", function () { confirmDelete(emp); });

      actionWrap.appendChild(viewBtn);
      actionWrap.appendChild(delBtn);
      actionTd.appendChild(actionWrap);

      tr.appendChild(nameTd);
      tr.appendChild(idTd);
      tr.appendChild(posIdTd);
      tr.appendChild(dateTd);
      tr.appendChild(actionTd);
      tbody.appendChild(tr);
    });
  }

  // ── Delete flow ──
  var pendingDelete = null;

  function photoUrl(empId) {
    var pin = cachedPin || getCachedPin() || "";
    return "/api/employee/" + encodeURIComponent(empId) + "/photo?pin=" + encodeURIComponent(pin);
  }

  function initialsFor(emp) {
    var raw = (emp.name || emp.id || "").trim();
    var parts = raw.split(/[\s_]+/).filter(Boolean);
    var first = (parts[0] || "?").charAt(0);
    var second = parts.length > 1 ? parts[parts.length - 1].charAt(0) : "";
    return (first + second).toUpperCase();
  }

  function confirmDelete(emp) {
    pendingDelete = emp;
    var displayName = emp.name || fmtName(emp.id);

    modalPhotoSlot.innerHTML = "";
    if (emp.has_photo) {
      var img = document.createElement("img");
      img.className = "confirm-photo";
      img.alt = displayName;
      img.src = photoUrl(emp.id);
      img.addEventListener("error", function () {
        modalPhotoSlot.innerHTML = "";
        var ph = document.createElement("div");
        ph.className = "confirm-photo-placeholder";
        ph.textContent = initialsFor(emp);
        modalPhotoSlot.appendChild(ph);
      });
      modalPhotoSlot.appendChild(img);
    } else {
      var ph = document.createElement("div");
      ph.className = "confirm-photo-placeholder";
      ph.textContent = initialsFor(emp);
      modalPhotoSlot.appendChild(ph);
    }

    modalText.innerHTML =
      "This will remove <strong>" + displayName + "</strong> from recognition. " +
      "They will not be able to clock in until re-enrolled. This cannot be undone.";
    modalBackdrop.classList.remove("hidden");
  }

  // ── Photo viewer ──
  function openPhoto(emp) {
    var displayName = emp.name || fmtName(emp.id);
    photoName.textContent = displayName;
    photoIdEl.textContent = emp.id;
    photoSlot.innerHTML = "";

    if (!emp.has_photo) {
      var miss = document.createElement("div");
      miss.className = "photo-missing";
      miss.textContent = "No photo on file for this employee.";
      photoSlot.appendChild(miss);
    } else {
      var img = document.createElement("img");
      img.alt = displayName;
      img.src = photoUrl(emp.id);
      img.addEventListener("error", function () {
        photoSlot.innerHTML = "";
        var miss = document.createElement("div");
        miss.className = "photo-missing";
        miss.textContent = "Failed to load photo.";
        photoSlot.appendChild(miss);
      });
      photoSlot.appendChild(img);
    }
    photoBackdrop.classList.remove("hidden");
  }

  function closePhoto() {
    photoBackdrop.classList.add("hidden");
    photoSlot.innerHTML = "";
  }

  photoClose.addEventListener("click", closePhoto);
  photoBackdrop.addEventListener("click", function (e) {
    if (e.target === photoBackdrop) closePhoto();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      if (!photoBackdrop.classList.contains("hidden")) closePhoto();
      else if (!modalBackdrop.classList.contains("hidden")) closeModal();
    }
  });

  function closeModal() {
    pendingDelete = null;
    modalBackdrop.classList.add("hidden");
    modalConfirm.disabled = false;
    modalConfirm.textContent = "Remove";
  }

  modalCancel.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", function (e) {
    if (e.target === modalBackdrop) closeModal();
  });

  modalConfirm.addEventListener("click", async function () {
    if (!pendingDelete) return;
    var emp = pendingDelete;
    var displayName = emp.name || fmtName(emp.id);

    modalConfirm.disabled = true;
    modalConfirm.textContent = "Removing...";

    var pin = cachedPin || getCachedPin();
    var url = "/api/enroll/" + encodeURIComponent(emp.id) + "?pin=" + encodeURIComponent(pin || "");

    try {
      var resp = await fetch(url, { method: "DELETE" });
      var data = await resp.json();

      if (data.status === "deleted") {
        closeModal();
        showToast(displayName + " removed", "success");
        loadEmployees();
      } else if (data.status === "unauthorized") {
        clearCachedPin();
        closeModal();
        showToast("Session expired — re-enter PIN", "error");
        setTimeout(function () { showPinGate("Please re-enter the manager PIN."); }, 800);
      } else if (data.status === "not_found") {
        closeModal();
        showToast("Employee not found", "error");
        loadEmployees();
      } else {
        modalConfirm.disabled = false;
        modalConfirm.textContent = "Remove";
        showToast("Error: " + (data.message || "unknown"), "error");
      }
    } catch (err) {
      modalConfirm.disabled = false;
      modalConfirm.textContent = "Remove";
      showToast("Server error: " + err.message, "error");
    }
  });

  // ── Boot ──
  async function bootMain() {
    await loadServerConfig();
    loadEmployees();
  }

  async function boot() {
    var health = await loadServerConfig();
    if (!health.enrollment_protected) {
      hidePinGate();
      bootMain();
      return;
    }
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
