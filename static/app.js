document.addEventListener("DOMContentLoaded", () => {
    // State
    let availableFeatures = [];
    let selectedFeatures = [];
    let isCustom = false;
    let currentTheme = "dark";

    // Elements
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeIcon = document.getElementById("themeIcon");
    const themeText = document.getElementById("themeText");

    const radioDataSources = document.querySelectorAll('input[name="dataSource"]');
    const sampleControlBox = document.getElementById("sampleControlBox");
    const uploadControlBox = document.getElementById("uploadControlBox");
    const regenerateBtn = document.getElementById("regenerateBtn");
    const csvFileInput = document.getElementById("csvFileInput");
    const uploadBtn = document.getElementById("uploadBtn");
    const uploadStatus = document.getElementById("uploadStatus");

    const featureCheckboxesDiv = document.getElementById("featureCheckboxes");
    const kSlider = document.getElementById("kSlider");
    const kValDisplay = document.getElementById("kValDisplay");
    const seedInput = document.getElementById("seedInput");
    const runClusterBtn = document.getElementById("runClusterBtn");

    const cfgFeatures = document.getElementById("cfgFeatures");
    const cfgK = document.getElementById("cfgK");
    const cfgStudents = document.getElementById("cfgStudents");

    const loadingSpinner = document.getElementById("loadingSpinner");
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    // Initialize Theme
    themeToggleBtn.addEventListener("click", () => {
        currentTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", currentTheme);
        themeIcon.textContent = currentTheme === "dark" ? "☀️" : "🌙";
        themeText.textContent = currentTheme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode";
        
        // Relayout active plotly charts for theme compatibility
        const plotlyDivs = document.querySelectorAll(".plotly-chart-container");
        plotlyDivs.forEach(div => {
            if (div.data) {
                const bg = currentTheme === "dark" ? "#1E293B" : "#FFFFFF";
                const fontColor = currentTheme === "dark" ? "#F1F5F9" : "#0F172A";
                Plotly.relayout(div, {
                    "paper_bgcolor": bg,
                    "plot_bgcolor": bg,
                    "font.color": fontColor
                });
            }
        });
    });

    // Tab Switcher
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            navTabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            tab.classList.add("active");
            const target = tab.getAttribute("data-tab");
            document.getElementById(target).classList.add("active");

            // Resize Plotly charts when switching tabs
            window.dispatchEvent(new Event('resize'));
        });
    });

    // Data Source Toggle
    radioDataSources.forEach(radio => {
        radio.addEventListener("change", (e) => {
            isCustom = e.target.value === "custom";
            if (isCustom) {
                sampleControlBox.classList.add("hidden");
                uploadControlBox.classList.remove("hidden");
            } else {
                sampleControlBox.classList.remove("hidden");
                uploadControlBox.classList.add("hidden");
                fetchDatasetInfo();
            }
        });
    });

    // Regenerate Sample Data
    regenerateBtn.addEventListener("click", async () => {
        showLoading(true);
        try {
            const res = await fetch("/api/regenerate", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                await fetchDatasetInfo();
                await executeClustering();
            }
        } catch (err) {
            console.error(err);
        } finally {
            showLoading(false);
        }
    });

    // Custom CSV Upload
    uploadBtn.addEventListener("click", async () => {
        if (!csvFileInput.files || csvFileInput.files.length === 0) {
            uploadStatus.textContent = "Please choose a CSV file first.";
            uploadStatus.className = "status-msg error";
            return;
        }

        const formData = new FormData();
        formData.append("file", csvFileInput.files[0]);

        showLoading(true);
        try {
            const res = await fetch("/api/upload", { method: "POST", body: formData });
            const data = await res.json();
            if (data.success) {
                uploadStatus.textContent = data.message;
                uploadStatus.className = "status-msg success";
                await fetchDatasetInfo();
                await executeClustering();
            } else {
                uploadStatus.textContent = data.error || "Upload failed.";
                uploadStatus.className = "status-msg error";
            }
        } catch (err) {
            uploadStatus.textContent = "Error uploading file.";
            uploadStatus.className = "status-msg error";
        } finally {
            showLoading(false);
        }
    });

    // Slider & Controls
    kSlider.addEventListener("input", (e) => {
        kValDisplay.textContent = e.target.value;
        cfgK.textContent = e.target.value;
    });

    runClusterBtn.addEventListener("click", () => {
        executeClustering();
    });

    // Fetch Initial Dataset Metadata
    async function fetchDatasetInfo() {
        showLoading(true);
        try {
            const res = await fetch(`/api/dataset?use_custom=${isCustom}`);
            const data = await res.json();
            if (data.success) {
                availableFeatures = data.numeric_features || [];
                cfgStudents.textContent = data.total_students;
                document.getElementById("metricTotalStudents").textContent = data.total_students;
                document.getElementById("metricNumericFeatures").textContent = availableFeatures.length;

                renderFeatureCheckboxes();
                renderPreviewTable(data.columns, data.preview);
                renderStatsTable(data.summary_stats);
            }
        } catch (err) {
            console.error("Failed to load dataset info:", err);
        } finally {
            showLoading(false);
        }
    }

    function renderFeatureCheckboxes() {
        featureCheckboxesDiv.innerHTML = "";
        selectedFeatures = availableFeatures.slice(0, 2); // Default first 2

        availableFeatures.forEach(feat => {
            const label = document.createElement("label");
            label.className = "checkbox-label";

            const input = document.createElement("input");
            input.type = "checkbox";
            input.value = feat;
            if (selectedFeatures.includes(feat)) input.checked = true;

            input.addEventListener("change", () => {
                updateSelectedFeatures();
            });

            const span = document.createElement("span");
            span.textContent = feat.replace(/_/g, " ");

            label.appendChild(input);
            label.appendChild(span);
            featureCheckboxesDiv.appendChild(label);
        });

        updateSelectedFeatures();
    }

    function updateSelectedFeatures() {
        const checked = featureCheckboxesDiv.querySelectorAll('input[type="checkbox"]:checked');
        selectedFeatures = Array.from(checked).map(c => c.value);

        if (selectedFeatures.length < 2) {
            runClusterBtn.disabled = true;
            cfgFeatures.textContent = "Select ≥ 2 features";
        } else {
            runClusterBtn.disabled = false;
            cfgFeatures.textContent = selectedFeatures.map(f => f.replace(/_/g, " ")).join(", ");
        }

        renderPredictorInputs();
    }

    function renderPreviewTable(cols, rows) {
        const table = document.getElementById("previewTable");
        const thead = table.querySelector("thead");
        const tbody = table.querySelector("tbody");

        thead.innerHTML = `<tr>${cols.map(c => `<th>${c.replace(/_/g, " ")}</th>`).join("")}</tr>`;
        tbody.innerHTML = rows.map(r => `<tr>${cols.map(c => `<td>${r[c] !== undefined ? r[c] : ""}</td>`).join("")}</tr>`).join("");
    }

    function renderStatsTable(stats) {
        const tbody = document.getElementById("statsTable").querySelector("tbody");
        tbody.innerHTML = stats.map(s => `
            <tr>
                <td><strong>${s.feature.replace(/_/g, " ")}</strong></td>
                <td>${Number(s.mean).toFixed(2)}</td>
                <td>${Number(s.std).toFixed(2)}</td>
                <td>${Number(s.min).toFixed(2)}</td>
                <td>${Number(s.median).toFixed(2)}</td>
                <td>${Number(s.max).toFixed(2)}</td>
            </tr>
        `).join("");
    }

    // Execute Clustering API Call
    async function executeClustering() {
        if (selectedFeatures.length < 2) return;

        showLoading(true);
        const payload = {
            use_custom: isCustom,
            features: selectedFeatures,
            n_clusters: parseInt(kSlider.value),
            random_state: parseInt(seedInput.value) || 42
        };

        try {
            const res = await fetch("/api/cluster", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.success) {
                renderOverviewMetrics(data);
                renderOptimalK(data);
                renderVisualizerPlots(data);
                renderPersonasAndTable(data);
            } else {
                alert("Clustering Error: " + (data.error || "Unknown error"));
            }
        } catch (err) {
            console.error("Clustering API execution failed:", err);
        } finally {
            showLoading(false);
        }
    }

    function renderOverviewMetrics(data) {
        if (selectedFeatures.length >= 2) {
            document.getElementById("metricFeature1Title").textContent = selectedFeatures[0].replace(/_/g, " ");
            document.getElementById("metricFeature2Title").textContent = selectedFeatures[1].replace(/_/g, " ");

            const stats = data.clustered_sample;
            if (stats && stats.length > 0) {
                const avg1 = stats.reduce((acc, cur) => acc + (cur[selectedFeatures[0]] || 0), 0) / stats.length;
                const avg2 = stats.reduce((acc, cur) => acc + (cur[selectedFeatures[1]] || 0), 0) / stats.length;
                document.getElementById("metricFeature1Val").textContent = avg1.toFixed(1);
                document.getElementById("metricFeature2Val").textContent = avg2.toFixed(1) + (selectedFeatures[1].includes("Percentage") ? "%" : "");
            }
        }
        renderPlotlyConfigured("plotlyCorrMatrix", data.corr_fig_json);
    }

    function renderOptimalK(data) {
        renderPlotlyConfigured("plotlyOptimalK", data.fig_eval_json);
        document.getElementById("recKVal").textContent = data.best_k;
        document.getElementById("recSilVal").textContent = data.best_silhouette_score;
    }

    function renderVisualizerPlots(data) {
        renderPlotlyConfigured("plotly2D", data.fig_2d_json);
        renderPlotlyConfigured("plotly3D", data.fig_3d_json);

        const container = document.getElementById("boxplotsContainer");
        container.innerHTML = "";
        data.boxplots_json.forEach((bpJson, idx) => {
            const bpDiv = document.createElement("div");
            bpDiv.className = "plotly-chart-container";
            bpDiv.id = `boxplotChart_${idx}`;
            container.appendChild(bpDiv);
            renderPlotlyConfigured(`boxplotChart_${idx}`, bpJson);
        });
    }

    function renderPersonasAndTable(data) {
        const grid = document.getElementById("personaCardsGrid");
        grid.innerHTML = data.personas.map(p => `
            <div class="persona-card" style="border-left-color: ${p.color};">
                <div class="persona-title">Cluster ${p.cluster_id}: ${p.name}</div>
                <div class="persona-stats">
                    <span class="persona-count" style="color: ${p.color};">${p.count}</span>
                    <span class="persona-pct">(${p.percentage}% of students)</span>
                </div>
                <div class="persona-desc">${p.description}</div>
            </div>
        `).join("");

        // Clustered Dataset Table
        const table = document.getElementById("clusteredDataTable");
        const cols = Object.keys(data.clustered_sample[0] || {});
        table.querySelector("thead").innerHTML = `<tr>${cols.map(c => `<th>${c.replace(/_/g, " ")}</th>`).join("")}</tr>`;
        table.querySelector("tbody").innerHTML = data.clustered_sample.map(r => `
            <tr>${cols.map(c => `<td>${r[c] !== undefined ? r[c] : ""}</td>`).join("")}</tr>
        `).join("");
    }

    function renderPredictorInputs() {
        const container = document.getElementById("dynamicPredictInputs");
        container.innerHTML = selectedFeatures.map(feat => `
            <div class="form-group" style="margin-bottom: 12px;">
                <label for="input_${feat}">${feat.replace(/_/g, " ")}:</label>
                <input type="number" step="0.1" id="input_${feat}" class="text-input" value="${feat.toLowerCase().includes('study') ? '20' : '80'}">
            </div>
        `).join("");
    }

    // Prediction Form Submit
    document.getElementById("predictionForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const studentData = {};
        selectedFeatures.forEach(feat => {
            const val = parseFloat(document.getElementById(`input_${feat}`).value) || 0;
            studentData[feat] = val;
        });

        const payload = {
            use_custom: isCustom,
            features: selectedFeatures,
            student_data: studentData,
            n_clusters: parseInt(kSlider.value),
            random_state: parseInt(seedInput.value) || 42
        };

        showLoading(true);
        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.success) {
                const panel = document.getElementById("predictorResult");
                panel.classList.remove("hidden");
                
                document.getElementById("resultClusterBadge").textContent = `Cluster ${data.cluster_id}`;
                document.getElementById("resultClusterBadge").style.backgroundColor = data.color;
                document.getElementById("resultPersonaName").textContent = data.persona_name;
                document.getElementById("resultPersonaDesc").textContent = data.persona_description;

                const adviceList = document.getElementById("resultAdviceList");
                adviceList.innerHTML = data.advice.map(item => `<li>${item}</li>`).join("");

                panel.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert("Prediction failed: " + data.error);
            }
        } catch (err) {
            console.error("Predict API error:", err);
        } finally {
            showLoading(false);
        }
    });

    // Helper to render Plotly chart with theme adjustment
    function renderPlotlyConfigured(elementId, figureJson) {
        const bg = currentTheme === "dark" ? "#1E293B" : "#FFFFFF";
        const fontColor = currentTheme === "dark" ? "#F1F5F9" : "#0F172A";

        figureJson.layout = figureJson.layout || {};
        figureJson.layout.paper_bgcolor = bg;
        figureJson.layout.plot_bgcolor = bg;
        figureJson.layout.font = figureJson.layout.font || {};
        figureJson.layout.font.color = fontColor;
        figureJson.layout.autosize = true;

        Plotly.newPlot(elementId, figureJson.data, figureJson.layout, { responsive: true });
    }

    function showLoading(show) {
        if (show) loadingSpinner.classList.remove("hidden");
        else loadingSpinner.classList.add("hidden");
    }

    // Startup Initialization
    (async () => {
        await fetchDatasetInfo();
        await executeClustering();
    })();
});
