const state = {
  config: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);

function values() {
  return {
    project: $("projectName").value.trim(),
    source_dir: $("sourceDir").value.trim(),
    trigger: $("trigger").value.trim(),
    mode: $("mode").value,
  };
}

function appendLog(text) {
  const log = $("consoleLog");
  const prefix = log.textContent === "Ready." ? "" : `${log.textContent}\n\n`;
  log.textContent = `${prefix}${text}`.trim();
  log.scrollTop = log.scrollHeight;
}

function setBusy(next) {
  state.busy = next;
  document.querySelectorAll("button[data-action]").forEach((button) => {
    button.disabled = next;
  });
}

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Request failed");
  return body;
}

function renderConfig(config) {
  state.config = config;
  const list = $("projectList");
  list.innerHTML = "";
  config.projects.forEach((project) => {
    const option = document.createElement("option");
    option.value = project;
    list.appendChild(option);
  });
  const projectInput = $("projectName");
  if (!projectInput.value && config.projects.length) {
    projectInput.value = config.projects[0];
  }
  renderStatus(config.checks);
}

function renderStatus(checks) {
  const strip = $("statusStrip");
  strip.innerHTML = "";
  const groups = {
    error: checks.filter((check) => check.level === "error").length,
    warn: checks.filter((check) => check.level === "warn").length,
    ok: checks.filter((check) => check.level === "ok").length,
  };
  [
    ["ok", `${groups.ok} OK`],
    ["warn", `${groups.warn} warnings`],
    ["error", `${groups.error} errors`],
  ].forEach(([level, label]) => {
    const chip = document.createElement("span");
    chip.className = `chip ${level}`;
    chip.textContent = label;
    strip.appendChild(chip);
  });
}

function renderReport(report) {
  const items = [
    ["Images", `${report.images.image_count} / ${report.images.total_mb} MB`],
    ["Captions", `${report.missing_caption_count} missing, ${report.empty_caption_count} empty`],
    ["Cache files", String(report.cache_file_count)],
    ["Outputs", String(report.lora_output_count)],
  ];
  const grid = $("reportGrid");
  grid.innerHTML = "";
  items.forEach(([label, value]) => {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<dt>${label}</dt><dd>${value}</dd>`;
    grid.appendChild(wrap);
  });
}

async function refreshConfig() {
  const config = await getJSON("/api/config");
  renderConfig(config);
}

async function refreshReport() {
  const project = values().project;
  if (!project) return;
  try {
    const report = await getJSON(`/api/report?project=${encodeURIComponent(project)}`);
    renderReport(report);
  } catch (error) {
    appendLog(`Report unavailable: ${error.message}`);
  }
}

function payloadFor(action) {
  const form = values();
  const payload = { action, ...form };
  if (action === "cache-latents-dry") {
    payload.action = "cache-latents";
    payload.dry_run = true;
  }
  if (action === "cache-text-dry") {
    payload.action = "cache-text";
    payload.dry_run = true;
  }
  if (action === "train-dry") {
    payload.action = "train";
    payload.dry_run = true;
  }
  return payload;
}

async function runAction(action) {
  setBusy(true);
  const payload = payloadFor(action);
  appendLog(`$ ${payload.action} ${payload.project || ""}`.trim());
  try {
    const result = await postJSON("/api/run", payload);
    const output = [result.stdout, result.stderr].filter(Boolean).join("\n");
    appendLog(`${output || "(no output)"}\nexit ${result.returncode}`);
    await refreshConfig();
    await refreshReport();
  } catch (error) {
    appendLog(`error: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

function bind() {
  document.querySelectorAll("button[data-action]").forEach((button) => {
    button.addEventListener("click", () => runAction(button.dataset.action));
  });
  $("clearLog").addEventListener("click", () => {
    $("consoleLog").textContent = "Ready.";
  });
  $("projectName").addEventListener("change", refreshReport);
}

async function init() {
  bind();
  try {
    await refreshConfig();
    await refreshReport();
  } catch (error) {
    appendLog(`Startup check failed: ${error.message}`);
  }
}

init();
