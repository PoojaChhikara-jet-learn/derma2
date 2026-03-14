/**
 * DermaWeb — script.js
 * Full interaction logic:
 *   - Float overlay open/close
 *   - Step navigation (choose → camera | upload)
 *   - Camera: start, snap, retake, stop
 *   - Upload: file pick, drag-drop, clear
 *   - Save & Analyse: POST to Flask /analyze
 *   - Result rendering inside the float
 */

"use strict";

const SERVER = "http://localhost:5000";

// ── State ─────────────────────────────────────────────────────────────────────
let cameraStream   = null;
let capturedBase64 = null;   // camera capture
let uploadedBase64 = null;   // file upload

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const show = (id, displayType = "flex") => { $(id).style.display = displayType; };
const hide = id => { $(id).style.display = "none"; };

// ── On load: health check ─────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  checkServer();
  bindUpload();
});

async function checkServer() {
  try {
    const r = await fetch(`${SERVER}/health`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    if (!d.api_ready) {
      console.warn("[DermaWeb] Server up but ANTHROPIC_API_KEY not set.");
    }
  } catch {
    console.warn("[DermaWeb] Server not reachable at", SERVER);
  }
}

// ═══════════════════════════════════════════════════════════
//  FLOAT OVERLAY
// ═══════════════════════════════════════════════════════════
function openAnalyser() {
  $("floatOverlay").classList.add("open");
  document.body.style.overflow = "hidden";
  goStep("choose");
}

function closeAnalyser() {
  $("floatOverlay").classList.remove("open");
  document.body.style.overflow = "";
  stopCamera();
}

function handleOverlayClick(e) {
  if (e.target === $("floatOverlay")) closeAnalyser();
}

// ═══════════════════════════════════════════════════════════
//  STEP NAVIGATION
// ═══════════════════════════════════════════════════════════
function goStep(target) {
  // Hide all steps
  ["step0", "stepCamera", "stepUpload"].forEach(id => {
    $(id).classList.remove("active");
  });

  if (target === "choose") {
    $("floatTitleText").textContent = "Skin Analyser";
    $("step0").classList.add("active");
    stopCamera();

  } else if (target === "camera") {
    $("floatTitleText").textContent = "Camera Capture";
    $("stepCamera").classList.add("active");
    resetCameraUI();

  } else if (target === "upload") {
    $("floatTitleText").textContent = "Upload Image";
    $("stepUpload").classList.add("active");
    resetUploadUI();
  }
}

// ═══════════════════════════════════════════════════════════
//  CAMERA
// ═══════════════════════════════════════════════════════════
function resetCameraUI() {
  $("camIdleMsg").style.display  = "flex";
  $("cameraFeed").style.display  = "none";
  $("capturedImg").style.display = "none";
  $("camScanLine").style.display = "none";
  show("camCtrlsStart", "flex");
  hide("camCtrlsLive");
  hide("camCtrlsPreview");
  hide("camLoading");
  hide("camError");
  $("camResult").innerHTML = "";
  capturedBase64 = null;
}

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    const feed = $("cameraFeed");
    feed.srcObject = cameraStream;
    feed.style.display = "block";
    $("camIdleMsg").style.display  = "none";
    $("camScanLine").style.display = "block";
    hide("camCtrlsStart");
    show("camCtrlsLive", "flex");
  } catch (err) {
    showFloatError("camError", "Camera access denied: " + err.message);
  }
}

function snapPhoto() {
  const video  = $("cameraFeed");
  const canvas = $("snapCanvas");
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0);

  capturedBase64 = canvas.toDataURL("image/jpeg", 0.92);

  // Show captured image
  const img = $("capturedImg");
  img.src = capturedBase64;
  img.style.display = "block";
  video.style.display = "none";
  $("camScanLine").style.display = "none";

  // Flash effect
  const fl = $("flashLayer");
  fl.classList.remove("go");
  void fl.offsetWidth;
  fl.classList.add("go");

  stopCameraTrack();
  hide("camCtrlsLive");
  show("camCtrlsPreview", "flex");
}

function retakeCamera() {
  $("capturedImg").style.display = "none";
  $("capturedImg").src = "";
  capturedBase64 = null;
  hide("camCtrlsPreview");
  hide("camError");
  $("camResult").innerHTML = "";
  startCamera();
}

function stopCamera() {
  stopCameraTrack();
  $("cameraFeed").style.display  = "none";
  $("cameraFeed").srcObject = null;
  $("camScanLine").style.display = "none";
}

function stopCameraTrack() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
}

// ═══════════════════════════════════════════════════════════
//  UPLOAD
// ═══════════════════════════════════════════════════════════
function bindUpload() {
  const fileInput  = $("fileInput");
  const uploadDrop = $("uploadDrop");

  fileInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) loadUploadFile(file);
  });

  uploadDrop.addEventListener("dragover",  e => { e.preventDefault(); uploadDrop.style.borderColor = "var(--sage)"; });
  uploadDrop.addEventListener("dragleave", () => { uploadDrop.style.borderColor = ""; });
  uploadDrop.addEventListener("drop", e => {
    e.preventDefault();
    uploadDrop.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) loadUploadFile(file);
  });
}

function loadUploadFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    uploadedBase64 = e.target.result;
    const prev = $("uploadPreview");
    prev.src = uploadedBase64;
    show("uploadPreview", "block");
    hide("uploadDrop");
    show("upCtrlsPreview", "flex");
    hide("upError");
    $("upResult").innerHTML = "";
  };
  reader.readAsDataURL(file);
}

function clearUpload() {
  uploadedBase64 = null;
  $("uploadPreview").style.display = "none";
  $("uploadPreview").src = "";
  $("fileInput").value = "";
  show("uploadDrop", "block");
  hide("upCtrlsPreview");
  hide("upError");
  $("upResult").innerHTML = "";
}

function resetUploadUI() {
  clearUpload();
  hide("upLoading");
}

// ═══════════════════════════════════════════════════════════
//  SAVE & ANALYSE  →  POST to Flask
// ═══════════════════════════════════════════════════════════
async function saveAndAnalyse(source) {
  const imageData = source === "camera" ? capturedBase64 : uploadedBase64;
  if (!imageData) { showFloatError(source === "camera" ? "camError" : "upError", "No image available."); return; }

  const loadingId = source === "camera" ? "camLoading"    : "upLoading";
  const errorId   = source === "camera" ? "camError"      : "upError";
  const resultId  = source === "camera" ? "camResult"     : "upResult";
  const saveBtn   = source === "camera" ? "btnSaveCam"    : "btnSaveUp";
  const previewCtrl = source === "camera" ? "camCtrlsPreview" : "upCtrlsPreview";

  // Show loading
  hide(previewCtrl);
  hide(errorId);
  $(resultId).innerHTML = "";
  show(loadingId, "flex");
  $(saveBtn).disabled = true;

  try {
    const res  = await fetch(`${SERVER}/analyze`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: imageData, save: true }),
    });

    const data = await res.json();
    hide(loadingId);

    if (!res.ok || !data.success) {
      throw new Error(data.error || "Analysis failed.");
    }

    // Render result
    $(resultId).innerHTML = buildResultHTML(data, imageData);
    animateConfBar(data.confidence);

  } catch (err) {
    hide(loadingId);
    const msg = err.message.includes("fetch")
      ? `Server unreachable at ${SERVER}. Make sure app.py is running.`
      : err.message;
    showFloatError(errorId, msg);
    show(previewCtrl, "flex");
  } finally {
    $(saveBtn).disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════
//  BUILD RESULT HTML
// ═══════════════════════════════════════════════════════════
function buildResultHTML(d, imageData) {
  const sevClass   = (d.severity || "normal").toLowerCase().replace(" ", "-");
  const urgClass   = (d.urgency  || "routine").toLowerCase();
  const urgIcon    = { routine: "🟢", soon: "🟡", urgent: "🔴" }[urgClass] || "🔵";

  const obsList = (d.observations    || []).map(o => `
    <li><span class="li-bullet">👁</span>${o}</li>`).join("");

  const recList = (d.recommendations || []).map((r, i) => `
    <li><span class="li-bullet">${i + 1}</span>${r}</li>`).join("");

  // Thumbnail of the analysed image
  const thumb = imageData
    ? `<img src="${imageData}" alt="Analysed" style="width:100%;border-radius:12px;max-height:180px;object-fit:cover;margin-bottom:1.25rem;border:1px solid var(--border);">`
    : "";

  return `
    <hr class="result-divider"/>

    ${thumb}

    <!-- Hero -->
    <div class="result-hero ${sevClass}">
      <div class="d-flex align-items-start justify-content-between flex-wrap gap-2">
        <div>
          <div class="result-condition">${d.condition || "Unknown"}</div>
          <div class="result-conf">Confidence: ${(d.confidence || 0).toFixed(1)}%</div>
          <span class="sev-badge ${sevClass}">
            ${getSevIcon(sevClass)} ${d.severity || "—"}
          </span>
        </div>
        <span class="urgency-chip ${urgClass}">
          ${urgIcon} ${d.urgency || "Routine"}
        </span>
      </div>

      <!-- Confidence bar -->
      <div class="conf-bar-wrap mt-3">
        <div class="conf-bar-fill" id="confBarFill" style="width:0%;"></div>
      </div>

      <p style="font-size:.84rem;color:var(--ink-md);margin-top:.75rem;line-height:1.6;">
        ${d.summary || ""}
      </p>
    </div>

    <!-- Observations -->
    ${obsList ? `
    <div class="result-section-title">Observations</div>
    <ul class="result-list">${obsList}</ul>` : ""}

    <!-- Recommendations -->
    ${recList ? `
    <div class="result-section-title">Recommendations</div>
    <ul class="result-list">${recList}</ul>` : ""}

    <!-- Disclaimer -->
    <div class="disclaimer-box">
      <span>⚠️</span>
      <span>${d.disclaimer || "This analysis is for informational purposes only. Please consult a dermatologist."}</span>
    </div>

    ${d.saved ? `<p style="font-size:.7rem;color:var(--sage);margin-top:.75rem;text-align:right;font-family:'JetBrains Mono',monospace;">
      <i class="bi bi-check-circle-fill me-1"></i>Image saved to server
    </p>` : ""}
  `;
}

function getSevIcon(cls) {
  const icons = { normal: "✅", mild: "🟡", moderate: "🟠", severe: "🔴" };
  return icons[cls] || "ℹ️";
}

function animateConfBar(pct) {
  // small delay so DOM is painted
  setTimeout(() => {
    const bar = document.getElementById("confBarFill");
    if (bar) bar.style.width = Math.min(pct, 100) + "%";
  }, 100);
}

// ═══════════════════════════════════════════════════════════
//  ERROR DISPLAY
// ═══════════════════════════════════════════════════════════
function showFloatError(id, msg) {
  const el = $(id);
  if (!el) return;
  el.textContent = "⚠ " + msg;
  el.style.display = "block";
}
