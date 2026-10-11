const form = document.querySelector("#measurementForm");
const submitButton = document.querySelector("#submitButton");
const resultGrid = document.querySelector("#resultGrid");
const warningStack = document.querySelector("#warningStack");
const liveBmi = document.querySelector("#liveBmi");
const liveBmiLabel = document.querySelector("#liveBmiLabel");
const bmiGroup = document.querySelector("#bmiGroup");
const calculatedAt = document.querySelector("#calculatedAt");
const systemStatus = document.querySelector("#systemStatus");
const copyButton = document.querySelector("#copyButton");
const toast = document.querySelector("#toast");
const modelInfoButton = document.querySelector("#modelInfoButton");
const modelInfoDialog = document.querySelector("#modelInfoDialog");
const waistHelpButton = document.querySelector("#waistHelpButton");
const waistHelpDialog = document.querySelector("#waistHelpDialog");
const modelTableBody = document.querySelector("#modelTableBody");
const integrityLine = document.querySelector("#integrityLine");
const retryButton = document.querySelector("#retryButton");
const resultStatus = document.querySelector("#resultStatus");
const inputSnapshot = document.querySelector("#inputSnapshot");

let latestResponse = null;
let latestInputSignature = null;
let requestSerial = 0;
let formRevision = 0;
let activeRequest = null;
let toastTimer = null;
let modelInfoLoaded = false;
const REQUEST_TIMEOUT_MS = 20000;

const targetClasses = {
  body_fat_percent: "fat-percent",
  total_fat_mass_kg: "fat-mass",
  total_lean_mass_kg: "lean-mass",
};

const familyLabels = {
  extra_trees: "Extra Trees",
  hist_gradient_boosting: "Hist GBDT",
};

function bmiState(value) {
  if (value < 18.5) return "BMI <18.5";
  if (value < 25) return "BMI 18.5–24.9";
  if (value < 30) return "BMI 25–29.9";
  if (value < 35) return "BMI 30–34.9";
  if (value < 40) return "BMI 35–39.9";
  return "BMI ≥40";
}

function updateLiveBmi() {
  const height = Number(document.querySelector("#height").value);
  const weight = Number(document.querySelector("#weight").value);
  if (!Number.isFinite(height) || !Number.isFinite(weight) || height <= 0 || weight <= 0) {
    liveBmi.textContent = "—";
    liveBmiLabel.textContent = "等待有效输入";
    return;
  }
  const bmi = weight / Math.pow(height / 100, 2);
  liveBmi.textContent = bmi.toFixed(1);
  liveBmiLabel.textContent = bmiState(bmi);
}

function syncRangeFill(range) {
  const min = Number(range.min);
  const max = Number(range.max);
  const value = Number(range.value);
  const progress = ((value - min) / (max - min)) * 100;
  range.style.background = `linear-gradient(90deg, var(--cyan) ${progress}%, rgba(111, 135, 142, .22) ${progress}%)`;
}

document.querySelectorAll(".range-input").forEach((range) => {
  const numberInput = document.querySelector(`#${range.dataset.pair}`);
  syncRangeFill(range);
  range.addEventListener("input", () => {
    numberInput.value = range.value;
    syncRangeFill(range);
    updateLiveBmi();
  });
  numberInput.addEventListener("input", () => {
    const numeric = Number(numberInput.value);
    if (Number.isFinite(numeric)) {
      range.value = String(Math.min(Number(range.max), Math.max(Number(range.min), numeric)));
      syncRangeFill(range);
    }
    updateLiveBmi();
  });
});

function payloadFromForm() {
  const data = new FormData(form);
  return {
    sex: data.get("sex"),
    age_years: Number(data.get("age_years")),
    height_cm: Number(data.get("height_cm")),
    weight_kg: Number(data.get("weight_kg")),
    waist_circumference_cm: Number(data.get("waist_circumference_cm")),
  };
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("is-loading", isLoading);
  resultGrid.setAttribute("aria-busy", String(isLoading));
}

function setResultStatus(message, state) {
  resultStatus.textContent = message;
  resultStatus.className = `result-status ${state ? `is-${state}` : ""}`;
}

function invalidateResults() {
  formRevision += 1;
  requestSerial += 1;
  activeRequest?.abort();
  activeRequest = null;
  latestResponse = null;
  latestInputSignature = null;
  copyButton.disabled = true;
  retryButton.hidden = true;
  setLoading(false);
  resultGrid.classList.add("is-stale");
  setResultStatus("输入已修改 · 待重新计算，下方如有结果仅对应上次输入", "stale");
  warningStack.innerHTML = "";
  updateLiveBmi();
}

// Form-level listeners cover number fields, sliders and sex radios.
form.addEventListener("input", invalidateResults);
form.addEventListener("change", invalidateResults);

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function snapshotText(inputs) {
  return `${inputs.sex === "male" ? "男性" : "女性"} · ${inputs.age_years} 岁 · 身高 ${formatValue(inputs.height_cm)} cm · 体重 ${formatValue(inputs.weight_kg)} kg · 腰围 ${formatValue(inputs.waist_circumference_cm)} cm`;
}

function versionText(value) {
  if (!value) return "未知";
  return String(value).startsWith("v") ? value : `v${value}`;
}

function updateVersions(data) {
  if (data.software_version) {
    document.querySelectorAll("[data-software-version]").forEach((element) => {
      element.textContent = versionText(data.software_version);
    });
  }
  const modelVersion = data.model_version ?? data.version;
  document.querySelectorAll("[data-model-version]").forEach((element) => {
    element.textContent = modelVersion ? versionText(modelVersion) : "未加载";
  });
}

function formatValue(value, digits = 1) {
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function calibrationLabel(interval) {
  const bmiPartition = interval.calibration_group.split("|").at(-1);
  if (interval.calibration_method === "sex_bmi_group") {
    return `按性别与 BMI ${bmiPartition} 组的跨周期误差校准`;
  }
  if (interval.calibration_method === "sex") return "按性别组的跨周期误差校准";
  if (interval.calibration_method === "global") return "按总体跨周期误差校准";
  return `按 BMI ${bmiPartition} 组的跨周期误差校准`;
}

function renderResults(data, submittedInputs) {
  latestResponse = {
    ...data,
    inputs: data.inputs ?? submittedInputs,
    generated_at: data.generated_at ?? new Date().toISOString(),
  };
  latestInputSignature = JSON.stringify(submittedInputs);
  copyButton.disabled = false;
  retryButton.hidden = true;
  resultGrid.classList.remove("is-stale");
  setResultStatus("已完成 · 结果对应下方计算输入", "ready");
  inputSnapshot.textContent = `计算输入：${snapshotText(latestResponse.inputs)}`;
  inputSnapshot.hidden = false;
  updateVersions(data);
  systemStatus.className = "system-status is-ready";
  systemStatus.querySelector("span:last-child").textContent = `本地运行 · ${data.results.length} 个模型已就绪`;
  bmiGroup.textContent = `${formatValue(data.derived.bmi, 1)} · ${data.derived.bmi_label}`;
  calculatedAt.textContent = new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(latestResponse.generated_at));

  resultGrid.innerHTML = data.results
    .map((result) => {
      const digits = result.digits ?? 1;
      const lower = formatValue(result.interval.lower, digits);
      const upper = formatValue(result.interval.upper, digits);
      const half = formatValue(result.interval.half_width, digits);
      const halfUnit = result.unit === "%" ? "个百分点" : result.unit;
      return `
        <article class="result-card ${targetClasses[result.target] || ""}">
          <div class="card-topline">
            <span class="result-label">${escapeHtml(result.label)}</span>
            <span class="model-chip">${escapeHtml(familyLabels[result.model_family] || result.model_family)}</span>
          </div>
          <div class="estimate-value">
            ${formatValue(result.estimate, digits)}<small>${escapeHtml(result.unit)}</small>
          </div>
          <p class="result-description">${escapeHtml(result.description)}</p>
          <div class="interval-section">
            <div class="interval-caption">
              <span>预测不确定范围</span>
              <strong>± ${half} ${escapeHtml(halfUnit)}</strong>
            </div>
            <div class="interval-track" aria-hidden="true"></div>
            <div class="interval-values">
              <span>${lower}</span>
              <span>${upper}</span>
            </div>
            <p class="interval-note">${escapeHtml(calibrationLabel(result.interval))}</p>
          </div>
        </article>`;
    })
    .join("");

  warningStack.innerHTML = data.warnings.length
    ? data.warnings.map((warning) => `<div class="warning-item">${escapeHtml(warning)}</div>`).join("")
    : "";
}

function renderError(message) {
  latestResponse = null;
  latestInputSignature = null;
  copyButton.disabled = true;
  retryButton.hidden = false;
  resultGrid.classList.remove("is-stale");
  inputSnapshot.hidden = true;
  bmiGroup.textContent = "—";
  calculatedAt.textContent = "—";
  setResultStatus("本次估算未完成 · 检查输入或连接后可重试", "error");
  resultGrid.innerHTML = `
    <div class="result-placeholder">
      <p>暂时无法生成估算</p>
    </div>`;
  warningStack.innerHTML = `<div class="error-item">${escapeHtml(message)}</div>`;
}

function validationMessage(detail) {
  if (typeof detail === "string" && /[\u4e00-\u9fff]/.test(detail)) return detail;
  if (!Array.isArray(detail)) return "输入未通过校验，请检查各项数值及其组合后重试。";
  const labels = { sex: "生理性别", age_years: "年龄", height_cm: "身高", weight_kg: "体重", waist_circumference_cm: "腰围" };
  return detail.map((item) => {
    if (/[\u4e00-\u9fff]/.test(item.msg)) return item.msg.replace(/^Value error,\s*/, "");
    const label = labels[item.loc?.at(-1)];
    return label ? `${label}不在可接受范围内，请按页面提示填写` : "输入未通过校验，请检查数值及其组合";
  }).join("；");
}

async function fetchJson(url, options = {}, controller = new AbortController()) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = window.setTimeout(() => {
      reject(new Error("等待模型服务超时，请确认应用仍在运行后点击重试。"));
      controller.abort();
    }, REQUEST_TIMEOUT_MS);
  });
  try {
    return await Promise.race([
      (async () => {
        const response = await fetch(url, { ...options, signal: controller.signal });
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          const message = response.status === 422 ? validationMessage(body.detail)
            : response.status === 503 ? "模型暂未就绪，请确认模型文件已安装并启动应用后重试。"
              : "模型服务暂时无法处理请求，请稍后重试。";
          throw new Error(message);
        }
        return await response.json();
      })(),
      timeout,
    ]);
  } catch (error) {
    if (error instanceof TypeError) throw new Error("无法连接本地模型服务，请确认应用仍在运行后重试。");
    if (error instanceof SyntaxError) throw new Error("模型服务返回了无效数据，请重试或重新启动应用。");
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function requestPrediction() {
  if (!form.reportValidity()) return;
  const serial = ++requestSerial;
  const revision = formRevision;
  const inputs = payloadFromForm();
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  latestResponse = null;
  latestInputSignature = null;
  copyButton.disabled = true;
  retryButton.hidden = true;
  setLoading(true);
  resultGrid.classList.add("is-stale");
  setResultStatus("正在计算 · 请稍候", "loading");
  warningStack.innerHTML = "";
  const isCurrentRequest = () => serial === requestSerial && revision === formRevision
    && JSON.stringify(inputs) === JSON.stringify(payloadFromForm());
  try {
    const data = await fetchJson("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputs),
    }, controller);
    // Aborting fetch is best effort; also reject responses for superseded inputs.
    if (isCurrentRequest()) renderResults(data, inputs);
  } catch (error) {
    if (isCurrentRequest()) renderError(error.message || "模型服务没有响应，请稍后重试。");
  } finally {
    if (serial === requestSerial) {
      activeRequest = null;
      setLoading(false);
    }
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  requestPrediction();
});

retryButton.addEventListener("click", requestPrediction);

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 2200);
}

copyButton.addEventListener("click", async () => {
  if (!latestResponse || latestInputSignature !== JSON.stringify(payloadFromForm())) return;
  const lines = [
    `身体成分软测量 · 软件 ${versionText(latestResponse.software_version)} · 模型 ${versionText(latestResponse.model_version ?? latestResponse.version)}`,
    `计算输入：${snapshotText(latestResponse.inputs)}`,
    `计算时间：${latestResponse.generated_at}`,
    `BMI：${latestResponse.derived.bmi.toFixed(1)}（${latestResponse.derived.bmi_label}）`,
    ...latestResponse.results.map(
      (result) =>
        `${result.label}：${formatValue(result.estimate, 1)} ${result.unit}（预测不确定范围 ${formatValue(result.interval.lower, 1)}–${formatValue(result.interval.upper, 1)} ${result.unit}）`,
    ),
    "注：群体数据估算，不是医学诊断或 DXA 实测。",
  ];
  try {
    await navigator.clipboard.writeText(lines.join("\n"));
    showToast("结果摘要已复制");
  } catch {
    showToast("浏览器未允许复制，请手动记录结果");
  }
});

function formatPercent(value) {
  if (!Number.isFinite(value)) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatBenchmark(value, digits, suffix = "") {
  return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "—";
}

async function loadModelInfo() {
  if (modelInfoLoaded) return;
  const info = await fetchJson("/api/v1/model-info");
  updateVersions(info);
  modelTableBody.innerHTML = info.targets
    .map((target) => {
      const unit = target.unit === "%" ? " 个百分点" : ` ${escapeHtml(target.unit)}`;
      const benchmark = target.benchmark || {};
      return `<tr>
        <td>${escapeHtml(target.label)}</td>
        <td>${escapeHtml(familyLabels[target.model_family] || target.model_family)}</td>
        <td>${formatBenchmark(benchmark.mae, 2, unit)}</td>
        <td>${formatBenchmark(benchmark.r2, 3)}</td>
        <td>${formatPercent(benchmark.interval_coverage)}</td>
      </tr>`;
    })
    .join("");
  integrityLine.textContent = info.artifact_integrity_verified
    ? "模型文件已通过 SHA-256 完整性校验"
    : "当前自定义模型目录没有完整性清单";
  integrityLine.classList.toggle("is-verified", info.artifact_integrity_verified);
  if (info.synthetic_data) integrityLine.textContent += "；纯合成数据演示，无人体测量意义";
  modelInfoLoaded = true;
}

modelInfoButton.addEventListener("click", async () => {
  modelInfoDialog.showModal();
  try {
    await loadModelInfo();
  } catch (error) {
    integrityLine.textContent = `${error.message} 关闭并重新打开模型说明可重试。`;
  }
});

waistHelpButton.addEventListener("click", () => waistHelpDialog.showModal());

document.querySelectorAll("[data-close-dialog]").forEach((button) => {
  button.addEventListener("click", () => button.closest("dialog").close());
});

document.querySelectorAll("dialog").forEach((dialog) => {
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
});

async function checkHealth() {
  const serial = requestSerial;
  try {
    const health = await fetchJson("/api/health");
    updateVersions(health);
    if (health.status !== "ok") throw new Error("模型文件未能加载。请确认模型文件已安装并重新启动应用后重试。");
    systemStatus.className = "system-status is-ready";
    systemStatus.querySelector("span:last-child").textContent = `本地运行 · ${health.models_loaded} 个模型已就绪`;
    // Startup must not replace an edit or manually submitted request.
    if (serial === requestSerial) await requestPrediction();
  } catch (error) {
    if (serial !== requestSerial) return;
    systemStatus.className = "system-status is-error";
    systemStatus.querySelector("span:last-child").textContent = "模型未连接";
    renderError(error.message);
  }
}

updateLiveBmi();
checkHealth();

