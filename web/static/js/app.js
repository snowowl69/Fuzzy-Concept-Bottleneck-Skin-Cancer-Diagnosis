/**
 * Fuzzy Concept-Bottleneck Skin Cancer Diagnosis
 * Frontend Application Logic
 */

// ==============================================================
// State
// ==============================================================
let currentConcepts = [];
let currentResults = null;
let lastCounterfactualText = "";
let isIntervened = false;

const CLASS_NAMES = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
];

const CONCEPT_LABELS = [
    "Asymmetry Score",
    "Border Irregularity",
    "Color Variation",
    "Diameter Score",
];

// ==============================================================
// DOM References
// ==============================================================
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const preprocessToggle = document.getElementById("preprocess-toggle");
const uploadSection = document.getElementById("upload-section");
const loadingSection = document.getElementById("loading-section");
const resultsSection = document.getElementById("results-section");
const overlayImg = document.getElementById("overlay-img");
const predClassEl = document.getElementById("pred-class");
const predConfidenceEl = document.getElementById("pred-confidence");
const probsContainer = document.getElementById("probabilities");
const featuresBody = document.getElementById("features-body");
const explanationBox = document.getElementById("explanation-box");
const interventionBadge = document.getElementById("intervention-badge");
const cfVisualComparison = document.getElementById("cf-visual-comparison");
const cfOrigImg = document.getElementById("cf-orig-img");
const cfRenderedImg = document.getElementById("cf-rendered-img");
const reportModal = document.getElementById("report-modal");
const reportContent = document.getElementById("report-content");

// ==============================================================
// Upload Zone - Drag & Drop + Click
// ==============================================================
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        uploadImage(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        uploadImage(e.target.files[0]);
    }
});

// ==============================================================
// Image Upload & Prediction
// ==============================================================
async function uploadImage(file) {
    // Validate file type
    if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file (JPG, PNG).");
        return;
    }

    // Show loading
    uploadSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    explanationBox.classList.add("hidden");
    if (cfVisualComparison) cfVisualComparison.classList.add("hidden");
    isIntervened = false;

    const formData = new FormData();
    formData.append("image", file);
    const applyPreprocess = preprocessToggle ? preprocessToggle.checked : true;
    formData.append("preprocess", applyPreprocess ? "true" : "false");

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (data.error) {
            alert("Prediction Error: " + data.error);
            resetToUpload();
            return;
        }

        currentResults = data;
        displayResults(data);
    } catch (err) {
        console.error("Upload failed:", err);
        alert("Upload failed. Please try again.");
        resetToUpload();
    }
}

function resetToUpload() {
    uploadSection.classList.remove("hidden");
    loadingSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
}

// ==============================================================
// Display Prediction Results
// ==============================================================
function displayResults(data) {
    // Segmentation overlay
    overlayImg.src = "data:image/png;base64," + data.overlay_img_b64;

    // Prediction class
    predClassEl.textContent = data.predicted_class;
    predConfidenceEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    interventionBadge.classList.add("hidden");

    // Probability bars
    renderProbabilities(data.probabilities, false);

    // Features table
    renderFeatures(data.features);

    // Setup concept sliders
    currentConcepts = data.concept_vector;
    setupSliders(currentConcepts);

    // Show results
    loadingSection.classList.add("hidden");
    resultsSection.classList.remove("hidden");
}

// ==============================================================
// Render Probability Bars
// ==============================================================
function renderProbabilities(probs, intervened) {
    let html = "";
    let idx = 0;

    for (const key in probs) {
        const p = probs[key] * 100;
        const barClass = intervened
            ? `prob-bar-fill--${idx} prob-bar-fill--intervened`
            : `prob-bar-fill--${idx}`;
        html += `
            <div class="prob-item">
                <div class="prob-item__header">
                    <span class="prob-item__name">${key}</span>
                    <span class="prob-item__value">${p.toFixed(1)}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill ${barClass}" style="width: ${p}%"></div>
                </div>
            </div>
        `;
        idx++;
    }

    probsContainer.innerHTML = html;
}

// ==============================================================
// Render Features Table
// ==============================================================
function renderFeatures(features) {
    let html = "";
    for (const key in features) {
        if (typeof features[key] === "number") {
            const displayName = key
                .replace(/_/g, " ")
                .replace(/\b\w/g, (c) => c.toUpperCase());
            html += `
                <tr>
                    <td>${displayName}</td>
                    <td>${features[key].toFixed(4)}</td>
                </tr>
            `;
        }
    }
    featuresBody.innerHTML = html;
}

// ==============================================================
// Concept Sliders
// ==============================================================
function setupSliders(concepts) {
    for (let i = 0; i < 4; i++) {
        const slider = document.getElementById(`slider_${i}`);
        const valueEl = document.getElementById(`val_${i}`);
        if (slider && valueEl) {
            slider.value = concepts[i];
            valueEl.textContent = concepts[i].toFixed(2);
        }
    }
}

// Debounced intervention
let interventionTimeout = null;

function onSliderInput() {
    for (let i = 0; i < 4; i++) {
        const slider = document.getElementById(`slider_${i}`);
        const valueEl = document.getElementById(`val_${i}`);
        if (slider && valueEl) {
            valueEl.textContent = parseFloat(slider.value).toFixed(2);
        }
    }

    if (interventionTimeout) clearTimeout(interventionTimeout);
    interventionTimeout = setTimeout(runIntervention, 200);
}

async function runIntervention() {
    const interventions = {};
    for (let i = 0; i < 4; i++) {
        const slider = document.getElementById(`slider_${i}`);
        if (slider) {
            interventions[i] = parseFloat(slider.value);
        }
    }

    const payload = {
        concepts: currentConcepts,
        interventions: interventions,
    };

    try {
        const res = await fetch("/api/intervene", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        if (!data.error) {
            predClassEl.textContent = data.intervened_diagnosis;
            predConfidenceEl.textContent = "Concept Intervention Active";
            interventionBadge.classList.remove("hidden");
            isIntervened = true;

            renderProbabilities(data.intervened_probabilities, true);
        }
    } catch (err) {
        console.error("Intervention failed:", err);
    }
}

// ==============================================================
// Counterfactual Generation & Visualizer
// ==============================================================
async function generateCounterfactual() {
    if (!currentResults || !currentConcepts || currentConcepts.length !== 4) {
        alert("Please upload and analyze a dermoscopic image first before generating a counterfactual explanation.");
        return;
    }

    const targetSelect = document.getElementById("target-class");
    const targetClass = targetSelect.value;

    const cfBtn = document.getElementById("cf-btn");
    cfBtn.textContent = "Generating...";
    cfBtn.disabled = true;

    const payload = {
        concepts: currentConcepts,
        target_class: targetClass,
        original_image_b64: currentResults ? currentResults.original_img_b64 : null,
    };

    try {
        const res = await fetch("/api/counterfactual", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        let data;
        try {
            data = await res.json();
        } catch (parseErr) {
            throw new Error(`Server returned non-JSON response (${res.status})`);
        }

        if (!res.ok || data.error) {
            const errorMsg = data.error || `Request failed with status ${res.status}`;
            alert("Counterfactual generation notice: " + errorMsg);
            return;
        }

        lastCounterfactualText = data.explanation_text;
        explanationBox.innerHTML = `
            <strong>🎯 Target Class:</strong> ${data.target_class_name}<br><br>
            <strong>📝 Clinical Explanation:</strong> ${data.explanation_text}
        `;
        explanationBox.classList.remove("hidden");

        // Display visual comparison if counterfactual image was rendered
        if (data.counterfactual_image_b64 && currentResults && cfVisualComparison) {
            cfOrigImg.src = "data:image/png;base64," + currentResults.original_img_b64;
            cfRenderedImg.src = "data:image/png;base64," + data.counterfactual_image_b64;
            cfVisualComparison.classList.remove("hidden");
        }
    } catch (err) {
        console.error("Counterfactual failed:", err);
        alert("Counterfactual generation notice: " + (err.message || "An unexpected error occurred."));
    } finally {
        cfBtn.textContent = "✨ Generate Explanation";
        cfBtn.disabled = false;
    }
}

// ==============================================================
// Clinical Report Generation
// ==============================================================
async function exportClinicalReport() {
    if (!currentResults) {
        alert("No active diagnosis to export.");
        return;
    }

    try {
        const res = await fetch("/api/report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                predicted_class: currentResults.predicted_class,
                confidence: currentResults.confidence,
                fuzzy_concepts: currentResults.fuzzy_concepts,
                features: currentResults.features,
                probabilities: currentResults.probabilities,
                counterfactual_summary: lastCounterfactualText || "No counterfactual explanation requested.",
            }),
        });
        const report = await res.json();

        renderReportModal(report);
    } catch (e) {
        console.error("Report generation failed:", e);
        alert("Failed to generate report.");
    }
}

function renderReportModal(report) {
    let conceptsHtml = "";
    for (const [k, v] of Object.entries(report.fuzzy_concepts)) {
        conceptsHtml += `<li><strong>${k}:</strong> ${Number(v).toFixed(3)}</li>`;
    }

    let probsHtml = "";
    for (const [k, v] of Object.entries(report.probabilities)) {
        probsHtml += `<tr><td>${k}</td><td>${(Number(v) * 100).toFixed(1)}%</td></tr>`;
    }

    reportContent.innerHTML = `
        <div class="report-meta-grid">
            <div><strong>Report ID:</strong> ${report.report_id}</div>
            <div><strong>Timestamp:</strong> ${report.timestamp}</div>
            <div><strong>Primary Diagnosis:</strong> <span style="color: var(--color-primary); font-weight: 700;">${report.predicted_class}</span></div>
            <div><strong>Model Confidence:</strong> ${(report.confidence * 100).toFixed(1)}%</div>
        </div>

        <div class="report-images-preview mt-4">
            <div>
                <h4>Original Lesion</h4>
                <img src="data:image/png;base64,${currentResults.original_img_b64}" class="report-img" alt="Original">
            </div>
            <div>
                <h4>Segmentation Overlay</h4>
                <img src="data:image/png;base64,${currentResults.overlay_img_b64}" class="report-img" alt="Overlay">
            </div>
        </div>

        <div class="mt-4">
            <h4>Fuzzy ABCDE Concept Memberships</h4>
            <ul class="report-concepts-list">
                ${conceptsHtml}
            </ul>
        </div>

        <div class="mt-4">
            <h4>Diagnosis Class Probabilities</h4>
            <table class="features-table">
                <thead><tr><th>Diagnosis Class</th><th>Probability</th></tr></thead>
                <tbody>${probsHtml}</tbody>
            </table>
        </div>

        <div class="mt-4">
            <h4>Counterfactual Clinical Explanation</h4>
            <p class="report-cf-text">${report.counterfactual_summary}</p>
        </div>

        <div class="report-disclaimer-box mt-4">
            <strong>⚠️ Research Prototype Notice:</strong> ${report.disclaimer}
        </div>
    `;

    reportModal.classList.remove("hidden");
}

function closeReportModal() {
    reportModal.classList.add("hidden");
}

// ==============================================================
// New Analysis Button
// ==============================================================
function newAnalysis() {
    resultsSection.classList.add("hidden");
    explanationBox.classList.add("hidden");
    if (cfVisualComparison) cfVisualComparison.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    fileInput.value = "";
    currentConcepts = [];
    currentResults = null;
    lastCounterfactualText = "";
    isIntervened = false;
}
