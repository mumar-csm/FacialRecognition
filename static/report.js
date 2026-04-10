(function () {
  "use strict";

  var startInput     = document.getElementById("start-date");
  var endInput       = document.getElementById("end-date");
  var employeeSelect = document.getElementById("employee-filter");
  var loadBtn        = document.getElementById("load-btn");
  var csvBtn         = document.getElementById("csv-btn");
  var summaryRow     = document.getElementById("summary-row");
  var tableWrapper   = document.getElementById("table-wrapper");
  var tbody          = document.getElementById("report-tbody");
  var emptyState     = document.getElementById("empty-state");
  var statusDot      = document.getElementById("status-dot");
  var statusText     = document.getElementById("status-text");

  function setStatus(state, text) {
    statusText.textContent = text;
    statusDot.className = "status-dot dot-" + state;
  }

  function fmtDate(ts)       { return ts.slice(0, 10); }
  function fmtTime(ts)       { return ts.slice(11, 19); }
  function fmtConfidence(d)  { return Math.max(0, (1 - d) * 100).toFixed(1) + "%"; }

  function fmtName(raw) {
    return raw.replace(/_/g, " ").replace(/\b\w/g, function (c) {
      return c.toUpperCase();
    });
  }

  function initDates() {
    var now = new Date();
    var end = now.toISOString().slice(0, 10);
    var s = new Date(now);
    s.setDate(s.getDate() - 6);
    startInput.value = s.toISOString().slice(0, 10);
    endInput.value   = end;
  }

  function loadEmployees() {
    fetch("/api/employees")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        data.employees.forEach(function (emp) {
          var opt = document.createElement("option");
          opt.value = emp.id;
          opt.textContent = emp.name || fmtName(emp.id);
          employeeSelect.appendChild(opt);
        });
      })
      .catch(function () { /* non-fatal — dropdown stays as "All Employees" */ });
  }

  function loadReport() {
    var start = startInput.value;
    var end   = endInput.value;
    var emp   = employeeSelect.value;

    if (!start || !end) {
      setStatus("error", "Please select a start and end date");
      return;
    }

    setStatus("scanning", "Loading...");
    loadBtn.disabled = true;
    csvBtn.disabled  = true;
    summaryRow.style.display  = "none";
    tableWrapper.style.display = "none";
    emptyState.style.display  = "none";

    var url = "/api/report?start=" + start + "&end=" + end;
    if (emp) url += "&employee=" + encodeURIComponent(emp);

    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        renderSummary(data.summary);
        renderTable(data.records);
        loadBtn.disabled = false;
        if (data.count > 0) csvBtn.disabled = false;
        setStatus("idle", data.count + " record" + (data.count === 1 ? "" : "s") + " loaded");
      })
      .catch(function (err) {
        setStatus("error", "Failed to load: " + err.message);
        loadBtn.disabled = false;
      });
  }

  function renderSummary(summary) {
    document.getElementById("sum-clockins").textContent  = summary.total_clock_ins;
    document.getElementById("sum-clockouts").textContent = summary.total_clock_outs;
    document.getElementById("sum-employees").textContent = summary.unique_employees;
    document.getElementById("sum-spoof").textContent     = summary.spoof_attempts_count;
    summaryRow.style.display = "flex";
  }

  function renderTable(records) {
    tbody.innerHTML = "";
    if (records.length === 0) {
      emptyState.style.display = "block";
      return;
    }
    records.forEach(function (rec) {
      var tr        = document.createElement("tr");
      var typeClass = rec.is_clock_in ? "badge-clock-in" : "badge-clock-out";
      var typeText  = rec.is_clock_in ? "Clock In" : "Clock Out";
      var name      = rec.employee_name || fmtName(rec.employee_id);
      tr.innerHTML =
        "<td>" + fmtDate(rec.timestamp) + "</td>" +
        "<td>" + fmtTime(rec.timestamp) + "</td>" +
        "<td>" + name + "</td>" +
        "<td><span class='badge " + typeClass + "'>" + typeText + "</span></td>" +
        "<td>" + fmtConfidence(rec.distance) + "</td>" +
        "<td>" + rec.camera_id + "</td>";
      tbody.appendChild(tr);
    });
    tableWrapper.style.display = "block";
  }

  function exportCsv() {
    var start = startInput.value;
    var end   = endInput.value;
    var emp   = employeeSelect.value;
    var url   = "/api/report/csv?start=" + start + "&end=" + end;
    if (emp) url += "&employee=" + encodeURIComponent(emp);
    window.location.href = url;
  }

  loadBtn.addEventListener("click", loadReport);
  csvBtn.addEventListener("click", exportCsv);

  initDates();
  loadEmployees();
})();
