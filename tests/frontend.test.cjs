/* Run with: node --test tests/frontend.test.cjs
 * Executes the shipped browser script with a minimal event-capable DOM and
 * controllable network/clock. Requests deliberately ignore abort to expose races.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const staticDir = path.resolve(__dirname, "../src/body_soft_sensor/static");
const script = fs.readFileSync(path.join(staticDir, "app.js"), "utf8");
const html = fs.readFileSync(path.join(staticDir, "index.html"), "utf8");
const initialInputs = { sex: "male", age_years: 30, height_cm: 175, weight_kg: 75, waist_circumference_cm: 85 };
const health = { status: "ok", models_loaded: 3, software_version: "0.3.0", model_version: "v0.2" };
const flush = () => new Promise((resolve) => setImmediate(resolve));

class Element {
  constructor(attributes = {}) {
    Object.assign(this, { textContent: "", innerHTML: "", disabled: false, hidden: false, style: {}, dataset: {}, listeners: {}, attributes: {}, value: "" }, attributes);
    this.classes = new Set();
    this.classList = {
      add: (value) => this.classes.add(value),
      remove: (value) => this.classes.delete(value),
      contains: (value) => this.classes.has(value),
      toggle: (value, enabled) => enabled ? this.classes.add(value) : this.classes.delete(value),
    };
  }
  addEventListener(type, callback) { (this.listeners[type] ??= []).push(callback); }
  dispatch(type, target = this) {
    const event = { target, preventDefault() {} };
    for (const callback of this.listeners[type] ?? []) callback(event);
    this.parent?.dispatch(type, target);
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  reportValidity() { return this.valid !== false; }
  querySelector(selector) {
    assert.equal(selector, "span:last-child");
    return this.statusLabel ??= new Element();
  }
  showModal() { this.open = true; }
  close() { this.open = false; }
}

function response(inputs, estimate = 25.0) {
  return {
    ...health,
    version: health.model_version,
    inputs,
    generated_at: "2026-10-11T03:00:00+00:00",
    derived: { bmi: inputs.weight_kg / (inputs.height_cm / 100) ** 2, bmi_label: "BMI 18.5–24.9" },
    results: [{ target: "body_fat_percent", label: "体脂率", description: "全身体脂占比", unit: "%", digits: 1, model_family: "extra_trees", estimate,
      interval: { lower: estimate - 5, upper: estimate + 5, half_width: 5, calibration_group: "male|lt25", calibration_method: "sex_bmi_group" } }],
    warnings: [],
  };
}

function harness() {
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map((match) => [match[1], new Element()]));
  const form = nodes.get("measurementForm");
  const names = { age: "age_years", height: "height_cm", weight: "weight_kg", waist: "waist_circumference_cm" };
  const sex = new Element({ value: "male", parent: form });
  const ranges = Object.entries(names).map(([id, name]) => {
    const input = nodes.get(id);
    input.value = String(initialInputs[name]);
    input.parent = form;
    const tag = html.match(new RegExp(`<input id="${id}"[^>]+>`))[0];
    return new Element({ dataset: { pair: id }, min: tag.match(/min="([^"]+)"/)[1], max: tag.match(/max="([^"]+)"/)[1], value: input.value, parent: form });
  });
  const softwareNodes = [new Element(), new Element(), new Element()];
  const modelNodes = [new Element(), new Element()];
  const requests = [];
  const timers = new Map();
  const clipboard = [];
  let nextTimer = 0;
  const context = vm.createContext({
    document: {
      querySelector(selector) {
        assert.ok(selector.startsWith("#"));
        const element = nodes.get(selector.slice(1));
        assert.ok(element, `Missing HTML element ${selector}`);
        return element;
      },
      querySelectorAll(selector) {
        if (selector === ".range-input") return ranges;
        if (selector === "[data-software-version]") return softwareNodes;
        if (selector === "[data-model-version]") return modelNodes;
        if (selector === "dialog") return [nodes.get("modelInfoDialog"), nodes.get("waistHelpDialog")];
        if (selector === "[data-close-dialog]") return [];
        throw new Error(`Unexpected selector: ${selector}`);
      },
    },
    FormData: class {
      get(name) {
        if (name === "sex") return sex.value;
        return nodes.get(Object.keys(names).find((id) => names[id] === name)).value;
      }
    },
    window: {
      setTimeout(callback, delay) { const id = ++nextTimer; timers.set(id, { callback, delay }); return id; },
      clearTimeout(id) { timers.delete(id); },
    },
    navigator: { clipboard: { async writeText(text) { clipboard.push(text); } } },
    fetch(url, options) {
      return new Promise((resolve, reject) => requests.push({ url, options, reject,
        respond(body, status = 200) { resolve({ ok: status >= 200 && status < 300, status, async json() { return body; } }); },
      }));
    },
    AbortController, TypeError, SyntaxError, Intl, Date,
  });
  vm.runInContext(script, context);
  return { nodes, form, sex, ranges, softwareNodes, modelNodes, requests, timers, clipboard,
    submit() { form.dispatch("submit"); },
    edit(id, value, event = "input") { const input = id === "sex" ? sex : nodes.get(id); input.value = String(value); input.dispatch(event); },
    expireNetworkTimers() {
      for (const [id, timer] of timers) if (timer.delay === 20000) { timers.delete(id); timer.callback(); }
    },
  };
}

async function ready() {
  const app = harness();
  app.requests[0].respond(health);
  await flush();
  app.requests[1].respond(response(initialInputs));
  await flush();
  assert.equal(app.nodes.get("copyButton").disabled, false);
  return app;
}

test("missing benchmark values render as unknown instead of historical scores", async () => {
  const app = await ready();
  app.nodes.get("modelInfoButton").dispatch("click");
  await flush();
  const request = app.requests.find((item) => item.url === "/api/v1/model-info");
  request.respond({
    ...health,
    artifact_integrity_verified: true,
    synthetic_data: true,
    targets: [{ label: "fixture", unit: "kg", model_family: "ridge", benchmark: {} }],
  });
  await flush();
  assert.equal((app.nodes.get("modelTableBody").innerHTML.match(/<td>—<\/td>/g) || []).length, 3);
  assert.match(app.nodes.get("integrityLine").textContent, /纯合成/);
  assert.doesNotMatch(app.nodes.get("modelTableBody").innerHTML, /NaN|undefined/);
});

test("all five inputs immediately invalidate results and disable copying", async () => {
  for (const [field, value, event] of [["age", 31, "input"], ["height", 180, "input"], ["weight", 90, "input"], ["waist", 95, "input"], ["sex", "female", "change"]]) {
    const app = await ready();
    app.edit(field, value, event);
    assert.equal(app.nodes.get("copyButton").disabled, true, field);
    assert.match(app.nodes.get("resultStatus").textContent, /待重新计算/);
    assert.equal(app.nodes.get("resultGrid").classList.contains("is-stale"), true);
    app.nodes.get("copyButton").dispatch("click");
    await flush();
    assert.equal(app.clipboard.length, 0);
    assert.match(app.nodes.get("inputSnapshot").textContent, /体重 75.0 kg/);
  }
});

test("slider edits synchronize their number input and invalidate the result", async () => {
  const app = await ready();
  const slider = app.ranges.find((range) => range.dataset.pair === "weight");
  slider.value = "90";
  slider.dispatch("input");
  assert.equal(app.nodes.get("weight").value, "90");
  assert.equal(app.nodes.get("liveBmi").textContent, "29.4");
  assert.equal(app.nodes.get("copyButton").disabled, true);
});

test("editing during a request cancels it and its late response cannot revive old results", async () => {
  const app = await ready();
  app.submit();
  const pending = app.requests.at(-1);
  app.edit("weight", 90);
  assert.equal(pending.options.signal.aborted, true);
  assert.equal(app.nodes.get("submitButton").disabled, false);
  pending.respond(response(initialInputs, 99));
  await flush();
  assert.equal(app.nodes.get("copyButton").disabled, true);
  assert.match(app.nodes.get("resultStatus").textContent, /待重新计算/);
  assert.doesNotMatch(app.nodes.get("resultGrid").innerHTML, /99.0/);
});

test("an old request finishing after a newer result cannot replace it", async () => {
  const app = await ready();
  app.submit();
  const older = app.requests.at(-1);
  app.edit("weight", 90);
  app.submit();
  const newer = app.requests.at(-1);
  newer.respond(response({ ...initialInputs, weight_kg: 90 }, 31));
  await flush();
  older.respond(response(initialInputs, 99));
  await flush();
  assert.match(app.nodes.get("resultGrid").innerHTML, /31.0/);
  assert.doesNotMatch(app.nodes.get("resultGrid").innerHTML, /99.0/);
  assert.match(app.nodes.get("inputSnapshot").textContent, /体重 90.0 kg/);
  assert.equal(app.nodes.get("copyButton").disabled, false);
});

test("an old failure cannot clear a newer in-progress request", async () => {
  const app = await ready();
  app.submit();
  const older = app.requests.at(-1);
  app.submit();
  const newer = app.requests.at(-1);
  older.respond({ detail: "old failure" }, 500);
  await flush();
  assert.equal(app.nodes.get("submitButton").disabled, true);
  assert.match(app.nodes.get("resultStatus").textContent, /正在计算/);
  newer.respond(response(initialInputs));
  await flush();
  assert.equal(app.nodes.get("submitButton").disabled, false);
});

test("timeout restores controls and retry uses the current inputs", async () => {
  const app = await ready();
  app.edit("weight", 90);
  app.submit();
  const stalled = app.requests.at(-1);
  app.expireNetworkTimers();
  await flush();
  assert.match(app.nodes.get("warningStack").innerHTML, /超时/);
  assert.equal(stalled.options.signal.aborted, true);
  assert.equal(app.nodes.get("retryButton").hidden, false);
  assert.equal(app.nodes.get("submitButton").disabled, false);
  app.nodes.get("retryButton").dispatch("click");
  const retry = app.requests.at(-1);
  assert.equal(JSON.parse(retry.options.body).weight_kg, 90);
  retry.respond(response({ ...initialInputs, weight_kg: 90 }, 31));
  await flush();
  stalled.respond(response(initialInputs, 99));
  await flush();
  assert.match(app.nodes.get("resultGrid").innerHTML, /31.0/);
  assert.equal(app.nodes.get("retryButton").hidden, true);
});

test("copy includes the exact input snapshot, timestamp and separate software/model versions", async () => {
  const app = await ready();
  app.nodes.get("copyButton").dispatch("click");
  await flush();
  assert.match(app.clipboard[0], /软件 v0.3.0 · 模型 v0.2/);
  assert.match(app.clipboard[0], /男性 · 30 岁 · 身高 175.0 cm · 体重 75.0 kg · 腰围 85.0 cm/);
  assert.match(app.clipboard[0], /2026-10-11T03:00:00\+00:00/);
  assert.equal(app.softwareNodes[0].textContent, "v0.3.0");
  assert.equal(app.modelNodes[0].textContent, "v0.2");
  assert.match(app.nodes.get("resultGrid").innerHTML, /± 5.0 个百分点/);
  assert.doesNotMatch(html, /瘦体重|v0\.2/);
});

test("initial health response does not auto-submit after the user edits", async () => {
  const app = harness();
  app.edit("sex", "female");
  app.requests[0].respond(health);
  await flush();
  assert.equal(app.requests.length, 1);
  assert.match(app.nodes.get("resultStatus").textContent, /待重新计算/);
});

test("network errors are readable, validation details are escaped, and failed results cannot be copied", async () => {
  const app = await ready();
  app.submit();
  app.requests.at(-1).reject(new TypeError("Failed to fetch"));
  await flush();
  assert.match(app.nodes.get("warningStack").innerHTML, /无法连接本地模型服务/);
  app.submit();
  app.requests.at(-1).respond({ detail: [{ msg: "Value error, 身高<img src=x>超出范围", loc: ["body"] }] }, 422);
  await flush();
  assert.match(app.nodes.get("warningStack").innerHTML, /身高&lt;img src=x&gt;超出范围/);
  assert.doesNotMatch(app.nodes.get("warningStack").innerHTML, /<img/);
  assert.equal(app.nodes.get("copyButton").disabled, true);
  assert.equal(app.nodes.get("inputSnapshot").hidden, true);
});
