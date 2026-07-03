const state = {
  config: null,
  busy: false,
  projectCleared: false,
  captionDefaultsLoaded: false,
  latestReport: null,
};

const $ = (id) => document.getElementById(id);
const MODE_HELP = {
  symlink: "Symlink references the originals without duplicating large images.",
  copy: "Copy duplicates images into the project; safest if the source folder may move.",
  hardlink: "Hardlink avoids duplicate storage but only works on the same filesystem.",
};
const ACTION_TITLES = {
  "validate-env": "Validate Environment",
  "download-models": "Download Missing Models",
  "init-project": "Init Project",
  "import-images": "Import Images",
  "generate-captions": "Generate VL Captions",
  "dataset-report": "Dataset Report",
  "cache-latents-dry": "Dry Run Latents",
  "cache-text-dry": "Dry Run Text Cache",
  "train-dry": "Train Dry Run",
  "cache-latents": "Run Latent Cache",
  "cache-text": "Run Text Cache",
  "copy-to-comfy": "Copy To ComfyUI",
};

function values() {
  return {
    project: $("projectName").value.trim(),
    source_dir: $("sourceDir").value.trim(),
    trigger: $("trigger").value.trim(),
    mode: $("mode").value,
    caption_model: $("captionModel").value.trim(),
    caption_local_only: $("captionLocalOnly").checked,
    force_caption: $("overwriteCaptions").checked,
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

function setRunSummary(text, level = "neutral") {
  const summary = $("runSummary");
  summary.textContent = text;
  summary.className = `run-summary ${level}`;
}

function completionKey(action) {
  const project = values().project || "no-project";
  return `krea2.workflow.${project}.${action}`;
}

function restoreCompletedActions() {
  document.querySelectorAll("button[data-action]").forEach((button) => {
    const done = localStorage.getItem(completionKey(button.dataset.action)) === "1";
    button.classList.toggle("complete", done);
    button.setAttribute("aria-label", done ? `${button.innerText} completed` : button.innerText);
  });
}

function markActionComplete(action, complete) {
  const key = completionKey(action);
  if (complete) {
    localStorage.setItem(key, "1");
  } else {
    localStorage.removeItem(key);
  }
  const button = document.querySelector(`button[data-action="${CSS.escape(action)}"]`);
  if (button) button.classList.toggle("complete", complete);
}

function resetCompletedActions(project) {
  const prefix = project ? `krea2.workflow.${project}.` : "krea2.workflow.";
  for (let index = localStorage.length - 1; index >= 0; index -= 1) {
    const key = localStorage.key(index);
    if (key && key.startsWith(prefix)) {
      localStorage.removeItem(key);
    }
  }
  document.querySelectorAll("button[data-action]").forEach((button) => {
    button.classList.remove("complete");
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
  if (!projectInput.value && config.projects.length && !state.projectCleared) {
    projectInput.value = config.projects[0];
  }
  if (!state.captionDefaultsLoaded && config.captioning) {
    $("captionModel").value = config.captioning.model || $("captionModel").value;
    $("captionLocalOnly").checked = config.captioning.local_files_only !== false;
    state.captionDefaultsLoaded = true;
  }
  renderStatus(config.checks);
  restoreCompletedActions();
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
  state.latestReport = report;
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

function resetReport() {
  const items = ["Images", "Captions", "Cache files", "Outputs"];
  const grid = $("reportGrid");
  grid.innerHTML = "";
  items.forEach((label) => {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<dt>${label}</dt><dd>-</dd>`;
    grid.appendChild(wrap);
  });
}

async function refreshConfig() {
  const config = await getJSON("/api/config");
  renderConfig(config);
}

async function refreshReport() {
  const project = values().project;
  if (!project) return null;
  try {
    const report = await getJSON(`/api/report?project=${encodeURIComponent(project)}`);
    renderReport(report);
    return report;
  } catch (error) {
    appendLog(`Report unavailable: ${error.message}`);
    return null;
  }
}

function imageUrl(report, item) {
  const params = new URLSearchParams({
    project: report.project,
    image: item.relative_path,
  });
  return `/api/image?${params.toString()}`;
}

function captionStateLabel(status) {
  if (status === "ready") return "Ready";
  if (status === "empty") return "Empty";
  return "Missing";
}

function renderDatasetReview(report) {
  const list = $("datasetReviewList");
  const items = report?.review_items || [];
  const ready = items.filter((item) => item.caption_status === "ready").length;
  const missing = items.filter((item) => item.caption_status === "missing").length;
  const empty = items.filter((item) => item.caption_status === "empty").length;
  $("datasetModalTitle").textContent = `${report.project} image set`;
  $("datasetModalMeta").textContent =
    `${items.length} images. ${ready} captions ready, ${missing} missing, ${empty} empty.`;
  list.innerHTML = "";
  if (!items.length) {
    const emptyMessage = document.createElement("div");
    emptyMessage.className = "empty-review";
    emptyMessage.textContent = "No project images were found for review.";
    list.appendChild(emptyMessage);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("article");
    row.className = "review-item";

    const thumb = document.createElement("div");
    thumb.className = "review-thumb";
    const image = document.createElement("img");
    image.src = imageUrl(report, item);
    image.alt = item.file_name;
    image.loading = "lazy";
    thumb.appendChild(image);

    const copy = document.createElement("div");
    copy.className = "review-copy";
    const meta = document.createElement("div");
    meta.className = "review-meta";
    const name = document.createElement("div");
    name.className = "review-name";
    name.textContent = item.file_name;
    const status = document.createElement("span");
    status.className = `caption-state ${item.caption_status}`;
    status.textContent = captionStateLabel(item.caption_status);
    meta.append(name, status);

    const caption = document.createElement("p");
    caption.className = "review-caption";
    caption.textContent =
      item.caption || (item.caption_status === "empty" ? "Caption file exists but is empty." : "No caption file found.");
    copy.append(meta, caption);
    row.append(thumb, copy);
    list.appendChild(row);
  });
}

function openDatasetReview(report) {
  if (!report) {
    setRunSummary("Run Dataset Report before reviewing the image set.", "warn");
    return;
  }
  renderDatasetReview(report);
  $("datasetModal").hidden = false;
  $("closeDatasetModal").focus();
}

function closeDatasetReview() {
  $("datasetModal").hidden = true;
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

function captionSummary(text) {
  const generated = text.match(/Generated (\d+) captions\. Skipped (\d+) existing non-empty captions\./);
  if (generated) {
    return `VL captions are ready. Generated ${generated[1]} captions and skipped ${generated[2]} existing captions.`;
  }
  const noneNeeded = text.match(/No captions need generation\. Skipped (\d+) existing non-empty captions\./);
  if (noneNeeded) {
    return `VL captions are already present. Skipped ${noneNeeded[1]} existing captions.`;
  }
  return "VL caption generation finished successfully.";
}

function summarizeResult(action, result) {
  const ok = result.returncode === 0;
  const output = [result.stdout, result.stderr].filter(Boolean).join("\n");
  if (!ok) {
    const failures = {
      "validate-env": "Environment check found issues. Review the run log for missing paths or models.",
      "download-models": "Model download found a problem. Review the run log for missing files, Hugging Face access, or captioning venv setup.",
      "init-project": "Project setup did not complete. Review the run log before continuing.",
      "import-images": "Image import did not complete. Check the source image location and import mode.",
      "generate-captions": "VL caption generation did not complete. Check the captioning venv, model, and log.",
      "dataset-report": "Dataset needs attention before caching or training.",
      "cache-latents-dry": "Latent cache dry run could not build the command.",
      "cache-text-dry": "Text cache dry run could not build the command.",
      "train-dry": "Training dry run could not build the command.",
      "cache-latents": "Latent cache failed. Review the musubi output in the run log.",
      "cache-text": "Text cache failed. Review the musubi output in the run log.",
      "copy-to-comfy": "Copy to ComfyUI failed. Check that an output LoRA exists and the destination is writable.",
    };
    return { text: failures[action] || `${ACTION_TITLES[action] || "Action"} failed. Review the run log.`, level: "error" };
  }

  const successes = {
    "validate-env": "Environment is OK.",
    "download-models": "Required models are present or were downloaded successfully.",
    "init-project": "Project folders and config are ready.",
    "import-images": "Source images have been imported into the project.",
    "dataset-report": "Dataset report completed. Use the counts below to decide the next step.",
    "cache-latents-dry": "Latent cache command is ready for review. Nothing was run.",
    "cache-text-dry": "Text cache command is ready for review. Nothing was run.",
    "train-dry": "Training command is ready for review. No training was run.",
    "cache-latents": "Latent cache completed successfully.",
    "cache-text": "Text cache completed successfully.",
    "copy-to-comfy": "The latest LoRA was copied to ComfyUI.",
  };
  if (action === "generate-captions") {
    return { text: captionSummary(output), level: "ok" };
  }
  return { text: successes[action] || `${ACTION_TITLES[action] || "Action"} completed successfully.`, level: "ok" };
}

async function runAction(action) {
  setBusy(true);
  const payload = payloadFor(action);
  setRunSummary(`Running ${ACTION_TITLES[action] || payload.action}...`, "warn");
  appendLog(`$ ${payload.action} ${payload.project || ""}`.trim());
  try {
    const result = await postJSON("/api/run", payload);
    const output = [result.stdout, result.stderr].filter(Boolean).join("\n");
    appendLog(`${output || "(no output)"}\nexit ${result.returncode}`);
    markActionComplete(action, result.returncode === 0);
    const summary = summarizeResult(action, result);
    setRunSummary(summary.text, summary.level);
    await refreshConfig();
    const report = await refreshReport();
    if (action === "dataset-report") {
      openDatasetReview(report);
    }
  } catch (error) {
    markActionComplete(action, false);
    setRunSummary(`${ACTION_TITLES[action] || "Action"} could not start: ${error.message}`, "error");
    appendLog(`error: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function pickSourceFolder() {
  const status = $("sourcePickerStatus");
  if ("showDirectoryPicker" in window) {
    try {
      const handle = await window.showDirectoryPicker();
      status.textContent = `Selected folder "${handle.name}". Browser security hides the absolute path, so keep the WSL path above accurate.`;
      return;
    } catch (error) {
      if (error.name !== "AbortError") {
        status.textContent = `Folder picker failed: ${error.message}`;
      }
      return;
    }
  }
  $("sourcePicker").click();
}

function handleSourcePickerChange(event) {
  const files = Array.from(event.target.files || []);
  const first = files[0]?.webkitRelativePath || files[0]?.name || "";
  const folder = first.includes("/") ? first.split("/")[0] : "selected folder";
  $("sourcePickerStatus").textContent =
    files.length > 0
      ? `Selected ${files.length} files from "${folder}". Browser security hides the absolute path, so keep the WSL path above accurate.`
      : "Use the WSL path the server can read, for example /mnt/c/Temp/JAG.";
}

function updateModeHelp() {
  $("modeHelp").textContent = MODE_HELP[$("mode").value] || MODE_HELP.symlink;
}

function clearProject() {
  const project = values().project;
  resetCompletedActions(project);
  state.projectCleared = true;
  $("projectName").value = "";
  $("sourceDir").value = "";
  $("trigger").value = "";
  $("mode").value = "symlink";
  $("overwriteCaptions").checked = false;
  $("sourcePicker").value = "";
  $("sourcePickerStatus").textContent = "Use the WSL path the server can read, for example /mnt/c/Temp/JAG.";
  updateModeHelp();
  resetReport();
  state.latestReport = null;
  $("consoleLog").textContent = "Ready.";
  setRunSummary("Project fields cleared. Workflow ticks were reset. No files were deleted.", "ok");
  closeDatasetReview();
  restoreCompletedActions();
}

async function reviewCurrentDataset() {
  const report = state.latestReport || (await refreshReport());
  openDatasetReview(report);
}

function bind() {
  document.querySelectorAll("button[data-action]").forEach((button) => {
    button.addEventListener("click", () => runAction(button.dataset.action));
  });
  $("clearLog").addEventListener("click", () => {
    $("consoleLog").textContent = "Ready.";
    setRunSummary("Run log cleared. Workflow state is unchanged.", "neutral");
  });
  $("clearProject").addEventListener("click", clearProject);
  $("reviewDataset").addEventListener("click", reviewCurrentDataset);
  $("closeDatasetModal").addEventListener("click", closeDatasetReview);
  $("datasetModal").addEventListener("click", (event) => {
    if (event.target === $("datasetModal")) closeDatasetReview();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("datasetModal").hidden) closeDatasetReview();
  });
  $("projectName").addEventListener("input", () => {
    state.projectCleared = false;
    state.latestReport = null;
  });
  $("projectName").addEventListener("change", refreshReport);
  $("projectName").addEventListener("change", restoreCompletedActions);
  $("pickSourceDir").addEventListener("click", pickSourceFolder);
  $("sourcePicker").addEventListener("change", handleSourcePickerChange);
  $("mode").addEventListener("change", updateModeHelp);
}

async function init() {
  bind();
  try {
    await refreshConfig();
    await refreshReport();
    updateModeHelp();
  } catch (error) {
    appendLog(`Startup check failed: ${error.message}`);
  }
}

init();
