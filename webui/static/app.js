const state = {
  groups: [],
  operations: [],
  status: null,
  jobs: [],
  activeGroup: "all",
  activeOperationId: null,
  activeJobId: null,
  renderedJobId: null,
  search: "",
};

const $ = (selector) => document.querySelector(selector);

function storageKey(opId, fieldName) {
  return `openfugu:${opId}:${fieldName}`;
}

function slotStorageKey(opId, fieldName, slotIndex, slotField) {
  return storageKey(opId, `${fieldName}:${slotIndex}:${slotField}`);
}

function clampNumber(value, min, max) {
  const num = Number.parseInt(value, 10);
  if (!Number.isFinite(num)) return min;
  return Math.min(max, Math.max(min, num));
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `${response.status} ${response.statusText}`);
  }
  return data;
}

function fmtTime(seconds) {
  if (!seconds) return "";
  return new Date(seconds * 1000).toLocaleTimeString();
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${mins}m ${rest}s`;
}

function fmtBytes(bytes) {
  if (bytes === null || bytes === undefined) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function statusLabel(status) {
  const labels = {
    queued: "排队",
    running: "运行中",
    cancelling: "取消中",
    cancelled: "已取消",
    succeeded: "成功",
    failed: "失败",
  };
  return labels[status] || status;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function resultValueText(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function appendKvList(root, title, entries) {
  if (!entries.length) return;
  const section = el("div", "result-section");
  section.append(el("span", "result-section-title", title));
  const grid = el("div", "result-kv");
  entries.forEach(([key, value]) => {
    const item = el("div", "result-kv-item");
    item.append(el("span", "", key), el("strong", "", resultValueText(value)));
    grid.append(item);
  });
  section.append(grid);
  root.append(section);
}

function renderJobResult(job) {
  const root = $("#jobResult");
  root.replaceChildren();
  if (!job) return;

  const result = job.result || {};
  const status = result.status || job.status;
  const summary = el("div", "result-summary");
  summary.append(el("span", `job-status ${status}`, statusLabel(status)));
  summary.append(el("span", "result-chip", `exit ${result.exit_code ?? job.exit_code ?? "—"}`));
  summary.append(el("span", "result-chip", fmtDuration(result.duration_s)));
  root.append(summary);

  const metrics = result.metrics && typeof result.metrics === "object" ? Object.entries(result.metrics) : [];
  appendKvList(root, "指标", metrics);

  const training = result.training && typeof result.training === "object" ? result.training : {};
  appendKvList(
    root,
    "训练",
    ["n_train", "iters", "max_turns", "sigma0", "seed", "export_api_keys"]
      .filter((key) => key in training)
      .map((key) => [key, training[key]]),
  );

  const artifacts = Array.isArray(result.artifacts) ? result.artifacts.filter((item) => item.path) : [];
  if (artifacts.length) {
    const section = el("div", "result-section");
    section.append(el("span", "result-section-title", "产物"));
    const list = el("div", "artifact-list");
    artifacts.forEach((artifact) => {
      const item = el("div", `artifact-item ${artifact.exists ? "exists" : "missing"}`);
      item.append(
        el("span", "", artifact.label || "file"),
        el("strong", "", artifact.exists ? "存在" : "缺失"),
        el("code", "", `${artifact.path}${artifact.size ? ` · ${fmtBytes(artifact.size)}` : ""}`),
      );
      list.append(item);
    });
    section.append(list);
    root.append(section);
  }

  if (Array.isArray(result.error_tail) && result.error_tail.length) {
    const section = el("div", "result-section");
    section.append(el("span", "result-section-title", "错误摘要"));
    const pre = el("pre", "result-error", result.error_tail.join("\n"));
    section.append(pre);
    root.append(section);
  }
}

function renderResultPane(job) {
  const root = $("#resultPane");
  if (!root) return;
  root.replaceChildren();
  if (!job || !job.result) return;

  const r = job.result || {};
  const s = r.summary || {};
  const m = r.metrics || {};
  const rows = Array.isArray(r.table) ? r.table : [];
  const highlights = Array.isArray(r.highlights) ? r.highlights : [];
  const status = r.status || job.status;

  // verdict badge
  const verdict = s.verdict || r.verdict;
  if (verdict) {
    const cls = verdict === "PASS" ? "pass" : verdict === "FAIL" ? "fail" : "running";
    root.append(el("div", `rp-verdict ${cls}`, verdict));
  } else if (status === "running") {
    root.append(el("div", "rp-verdict running", "运行中..."));
  } else if (status === "succeeded") {
    root.append(el("div", "rp-verdict pass", "完成"));
  } else if (status === "failed") {
    root.append(el("div", "rp-verdict fail", "出错"));
  }

  // lift headline (eval)
  if ("lift_pct" in s && "coordinator" in s && "best_single" in s) {
    const lift = s.lift_pct;
    const cls = lift > 0 ? "pos" : "neg";
    const row = el("div", "rp-stat-row");
    row.append(el("span", "", "coordinator vs best single"));
    const strong = el("strong", `rp-lift ${cls}`, `${lift > 0 ? "+" : ""}${lift.toFixed(0)}%`);
    row.append(strong);
    root.append(row);
    const sub = el("div", "rp-stat-row");
    sub.append(el("span", "", `${s.coordinator.toFixed(3)} vs ${s.best_single.toFixed(3)}`));
    if (s.best_single_label) sub.append(el("strong", "", s.best_single_label));
    root.append(sub);
  }

  // training headline
  if ("solved" in s && "base" in s) {
    const delta = s.solved - s.base;
    const row = el("div", "rp-stat-row");
    row.append(el("span", "", "solved vs base"));
    const cls = delta > 0 ? "delta-good" : delta < 0 ? "delta-bad" : "";
    row.append(el("strong", cls, `${s.solved.toFixed(3)} / ${s.base.toFixed(3)}`));
    root.append(row);
    if (s.peak_solved !== undefined && s.peak_solved !== s.solved) {
      const peak = el("div", "rp-stat-row");
      peak.append(el("span", "", "peak solved"));
      peak.append(el("strong", "delta-good", s.peak_solved.toFixed(3)));
      root.append(peak);
    }
  }

  // oracle reach
  if ("oracle_pct" in s) {
    const row = el("div", "rp-stat-row");
    row.append(el("span", "", "oracle ceiling"));
    row.append(el("strong", "", `${s.oracle_pct.toFixed(0)}%`));
    root.append(row);
  }

  // saved head info
  if (s.saved_path) {
    const row = el("div", "rp-stat-row");
    row.append(el("span", "", "saved head"));
    const info = s.head_floats ? `${s.head_floats} floats` : "";
    row.append(el("strong", "", info));
    root.append(row);
  }

  // training progress
  const iterRows = (job.logs || []).join("").match(/\[iter\s+\d+\][^\n]*/g) || [];
  if (iterRows.length) {
    root.append(el("div", "rp-section-title", "训练进度"));
    const prog = el("div", "rp-progress");
    const last5 = iterRows.slice(-5);
    last5.forEach((line) => {
      const m2 = line.match(/\[iter\s+(\d+)\]\s+best_solved=([\d.]+)/);
      if (!m2) return;
      const row = el("div", "rp-progress-row");
      row.append(el("span", "", `iter ${m2[1]}`), el("span", "rp-peak", `best=${m2[2]}`));
      prog.append(row);
    });
    root.append(prog);
  }

  // comparison table
  if (rows.length) {
    root.append(el("div", "rp-section-title", "策略对比"));
    const table = el("div", "rp-table");
    const maxRate = Math.max(...rows.map((row) => row.rate || 0), 0.001);
    rows.forEach((row) => {
      const cls = row.best_single ? "best" : (row.label || "").includes("coordinator") ? "coordinator" : (row.label || "").toLowerCase().includes("oracle") ? "oracle" : "";
      const item = el("div", `rp-table-row ${cls}`);
      const pct = ((row.rate / maxRate) * 100).toFixed(0);
      const label = el("span", "rp-label", row.label);
      const rate = el("span", "rp-rate", row.rate.toFixed(3));
      const bar = el("div", "rp-bar");
      bar.append(el("div", "rp-bar-fill"));
      bar.firstChild.style.width = `${pct}%`;
      if (row.best_single) label.textContent = `★ ${row.label}`;
      item.append(label, rate, bar);
      table.append(item);
    });
    root.append(table);
  }

  // highlights
  if (highlights.length) {
    root.append(el("div", "rp-section-title", "关键输出"));
    const list = el("div", "rp-highlights");
    highlights.slice(-6).forEach((line) => {
      list.append(el("div", "rp-highlight", line));
    });
    root.append(list);
  }
}

function renderGroups() {
  const nav = $("#groupNav");
  nav.replaceChildren();
  state.groups.forEach((group) => {
    const count =
      group.id === "all"
        ? state.operations.length
        : state.operations.filter((op) => op.group === group.id).length;
    const button = el("button", `nav-button ${state.activeGroup === group.id ? "active" : ""}`);
    button.type = "button";
    button.append(el("span", "", group.label));
    button.append(el("span", "nav-count", String(count)));
    button.addEventListener("click", () => {
      state.activeGroup = group.id;
      renderGroups();
      renderOperations();
      renderOperationDetail();
    });
    nav.append(button);
  });
}

function renderStatus() {
  const list = $("#statusList");
  list.replaceChildren();
  if (!state.status) {
    list.append(el("div", "empty", "读取中"));
    return;
  }
  const depOk = Object.values(state.status.dependencies).filter(Boolean).length;
  const depTotal = Object.keys(state.status.dependencies).length;
  const rows = [
    ["Python", state.status.python.version, true],
    ["依赖", `${depOk}/${depTotal}`, depOk === depTotal],
    ["artifacts", state.status.artifacts.dir_exists ? "存在" : "缺失", state.status.artifacts.dir_exists],
    ["向量", state.status.artifacts.vector_exists ? "存在" : "缺失", state.status.artifacts.vector_exists],
    ["fixture", state.status.artifacts.fixture_exists ? "存在" : "缺失", state.status.artifacts.fixture_exists],
    ["任务", `${state.status.jobs.running} 运行 / ${state.status.jobs.total} 总计`, true],
  ];
  rows.forEach(([name, value, ok]) => {
    const row = el("div", "status-row");
    row.append(el("span", "", name));
    row.append(el("span", `status-dot ${ok ? "ok" : "bad"}`, value));
    list.append(row);
  });
}

function fieldValue(opId, field) {
  const saved = localStorage.getItem(storageKey(opId, field.name));
  if (field.type === "checkbox") {
    if (saved !== null) return saved === "true";
    return Boolean(field.default);
  }
  return saved !== null ? saved : field.default || "";
}

function slotFieldValue(opId, field, slotIndex, slotField) {
  const saved = localStorage.getItem(slotStorageKey(opId, field.name, slotIndex, slotField));
  const defaults = Array.isArray(field.default) ? field.default : [];
  const row = defaults[slotIndex] || {};
  return saved !== null ? saved : row[slotField] || "";
}

function slotCountValue(opId, field) {
  const maxSlots = field.max_slots || field.slots || 7;
  const minSlots = field.min_slots || 1;
  const saved = localStorage.getItem(storageKey(opId, `${field.name}:count`));
  return clampNumber(saved || field.slots || maxSlots, minSlots, maxSlots);
}

function slotJsonRows(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.slots)) return data.slots;
  if (Array.isArray(data?.slot_config)) return data.slot_config;
  throw new Error("JSON 中没有可导入的 slots 数组");
}

function renderSlotConfigField(op, field) {
  const wrapper = el("section", "slot-config");
  wrapper.dataset.name = field.name;
  wrapper.dataset.operation = op.id;
  const maxSlots = field.max_slots || field.slots || 7;
  const minSlots = field.min_slots || 1;
  let activeSlots = slotCountValue(op.id, field);
  wrapper.dataset.activeSlots = String(activeSlots);

  const head = el("div", "slot-config-head");
  head.append(el("span", "", field.label));
  const countWrap = el("label", "slot-count");
  countWrap.append(el("span", "", "数量"));
  const countInput = document.createElement("input");
  countInput.type = "number";
  countInput.min = String(minSlots);
  countInput.max = String(maxSlots);
  countInput.value = String(activeSlots);
  countWrap.append(countInput);
  const countText = el("small", "", `${activeSlots}/${maxSlots} slots`);
  head.append(countWrap, countText);
  wrapper.append(head);

  const tools = el("div", "slot-tools");
  const includeKeyLabel = el("label", "slot-toggle");
  const includeKeyInput = document.createElement("input");
  includeKeyInput.type = "checkbox";
  includeKeyInput.checked = localStorage.getItem(storageKey(op.id, `${field.name}:json_api_key`)) === "true";
  includeKeyLabel.append(includeKeyInput, el("span", "", "导出 API Key"));
  includeKeyInput.addEventListener("change", () => {
    localStorage.setItem(storageKey(op.id, `${field.name}:json_api_key`), String(includeKeyInput.checked));
  });

  const importInput = document.createElement("input");
  importInput.type = "file";
  importInput.accept = ".json,application/json";
  importInput.hidden = true;
  const importButton = el("button", "secondary slot-tool", "导入 JSON");
  importButton.type = "button";
  const exportButton = el("button", "secondary slot-tool", "导出 JSON");
  exportButton.type = "button";
  tools.append(includeKeyLabel, importButton, exportButton, importInput);
  wrapper.append(tools);

  const grid = el("div", "slot-grid");
  const columns = [
    ["model", "模型", "openai/gpt-4o-mini", "text"],
    ["api_base", "API Base URL", "https://api.deepseek.com", "url"],
    ["api_key", "API Key", "sk-...", "password"],
  ];

  function updateVisibleRows() {
    wrapper.dataset.activeSlots = String(activeSlots);
    countText.textContent = `${activeSlots}/${maxSlots} slots`;
    grid.querySelectorAll(".slot-row").forEach((row) => {
      row.hidden = Number(row.dataset.slotIndex) >= activeSlots;
    });
  }

  function setSlotInput(index, slotField, value) {
    const input = grid.querySelector(`[data-slot-index="${index}"][data-slot-field="${slotField}"]`);
    if (!input) return;
    input.value = value || "";
    localStorage.setItem(slotStorageKey(op.id, field.name, index, slotField), input.value);
  }

  function currentSlots(includeApiKey) {
    const slots = [];
    grid.querySelectorAll(".slot-row:not([hidden])").forEach((row) => {
      const slot = {};
      row.querySelectorAll("[data-slot-field]").forEach((input) => {
        const key = input.dataset.slotField;
        const value = input.value.trim();
        if (!value || (key === "api_key" && !includeApiKey)) return;
        slot[key] = value;
      });
      if (slot.model || slot.api_base || slot.api_key) slots.push(slot);
    });
    return slots;
  }

  function applySlots(rows) {
    const normalized = rows
      .filter((row) => row && typeof row === "object")
      .map((row) => ({
        model: String(row.model || row.model_name || "").trim(),
        api_base: String(row.api_base || row.base_url || row.url || "").trim(),
        api_key: String(row.api_key || row.key || "").trim(),
      }))
      .filter((row) => row.model || row.api_base || row.api_key)
      .slice(0, maxSlots);
    if (!normalized.length) {
      throw new Error("JSON 中没有有效槽位");
    }

    activeSlots = clampNumber(normalized.length, minSlots, maxSlots);
    countInput.value = String(activeSlots);
    localStorage.setItem(storageKey(op.id, `${field.name}:count`), String(activeSlots));
    for (let index = 0; index < maxSlots; index += 1) {
      const row = normalized[index] || {};
      ["model", "api_base", "api_key"].forEach((slotField) => {
        setSlotInput(index, slotField, row[slotField] || "");
      });
    }
    updateVisibleRows();
  }

  countInput.addEventListener("input", () => {
    activeSlots = clampNumber(countInput.value, minSlots, maxSlots);
    countInput.value = String(activeSlots);
    localStorage.setItem(storageKey(op.id, `${field.name}:count`), String(activeSlots));
    updateVisibleRows();
  });

  for (let index = 0; index < maxSlots; index += 1) {
    const row = el("div", "slot-row");
    row.dataset.slotIndex = String(index);
    row.append(el("div", "slot-index", `槽位 ${index}`));
    columns.forEach(([slotField, labelText, placeholder, type]) => {
      const label = el("label", "slot-input");
      label.append(el("span", "", labelText));
      const input = document.createElement("input");
      input.type = type;
      input.placeholder = placeholder;
      input.dataset.slotIndex = String(index);
      input.dataset.slotField = slotField;
      input.value = slotFieldValue(op.id, field, index, slotField);
      input.addEventListener("input", () => {
        localStorage.setItem(slotStorageKey(op.id, field.name, index, slotField), input.value);
      });
      label.append(input);
      row.append(label);
    });
    grid.append(row);
  }

  importButton.addEventListener("click", () => {
    importInput.value = "";
    importInput.click();
  });
  importInput.addEventListener("change", async () => {
    const file = importInput.files && importInput.files[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      applySlots(slotJsonRows(data));
    } catch (error) {
      alert(error.message);
    }
  });
  exportButton.addEventListener("click", () => {
    const blob = new Blob([`${JSON.stringify(currentSlots(includeKeyInput.checked), null, 2)}\n`], {
      type: "application/json",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${op.id}-${field.name}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  });

  wrapper.append(grid);
  updateVisibleRows();
  return wrapper;
}

function renderField(op, field) {
  if (field.type === "slot_config") {
    return renderSlotConfigField(op, field);
  }

  const label = el("label", `field ${field.type === "checkbox" ? "checkbox" : ""}`);
  const caption = el("span", "", `${field.label}${field.required ? " *" : ""}`);
  let input;

  if (field.type === "textarea") {
    input = document.createElement("textarea");
    input.rows = 3;
  } else {
    input = document.createElement("input");
    input.type = field.type === "checkbox" ? "checkbox" : field.type || "text";
  }

  input.name = field.name;
  input.dataset.operation = op.id;
  if (field.placeholder) input.placeholder = field.placeholder;
  if (field.type === "checkbox") {
    input.checked = fieldValue(op.id, field);
    label.append(input, caption);
  } else {
    input.value = fieldValue(op.id, field);
    label.append(caption, input);
  }

  // show_if: hide this field until the controlling checkbox is checked
  if (field.show_if) {
    label.classList.add("conditional-field");
    label.dataset.showIf = field.show_if;
    const controller = document.querySelector(`[data-operation="${op.id}"][name="${field.show_if}"]`);
    const syncVisibility = () => {
      const ctrl = document.querySelector(`[data-operation="${op.id}"][name="${field.show_if}"]`);
      if (!ctrl) return;
      label.hidden = ctrl.type === "checkbox" ? !ctrl.checked : !ctrl.value;
    };
    syncVisibility();
    // re-sync when the controller checkbox toggles (wired after render in renderOperationDetail)
  }

  input.addEventListener("change", () => {
    const value = field.type === "checkbox" ? String(input.checked) : input.value;
    localStorage.setItem(storageKey(op.id, field.name), value);
  });
  input.addEventListener("input", () => {
    if (field.type !== "checkbox") {
      localStorage.setItem(storageKey(op.id, field.name), input.value);
    }
  });
  return label;
}

function collectValues(card) {
  const values = {};
  card.querySelectorAll("[name]").forEach((input) => {
    values[input.name] = input.type === "checkbox" ? input.checked : input.value;
  });
  card.querySelectorAll(".slot-config[data-name]").forEach((root) => {
    const slots = [];
    root.querySelectorAll(".slot-row:not([hidden])").forEach((row, visibleIndex) => {
      const slot = {};
      row.querySelectorAll("[data-slot-field]").forEach((input) => {
        slot[input.dataset.slotField] = input.value;
      });
      slots[visibleIndex] = slot;
    });
    root.querySelectorAll(".slot-row[hidden] [data-slot-index][data-slot-field]").forEach((input) => {
      const index = Number(input.dataset.slotIndex);
      const field = input.dataset.slotField;
      localStorage.setItem(slotStorageKey(input.closest(".slot-config").dataset.operation, root.dataset.name, index, field), input.value);
    });
    values[root.dataset.name] = slots;
  });
  return values;
}

function filteredOperations() {
  const query = state.search.trim().toLowerCase();
  return state.operations.filter((op) => {
    const groupOk = state.activeGroup === "all" || op.group === state.activeGroup;
    const text = `${op.title} ${op.description} ${op.badge}`.toLowerCase();
    return groupOk && (!query || text.includes(query));
  });
}

function ensureActiveOperation(operations) {
  if (!operations.length) {
    state.activeOperationId = null;
    return null;
  }
  const active = operations.find((op) => op.id === state.activeOperationId);
  if (active) return active;
  state.activeOperationId = operations[0].id;
  return operations[0];
}

function renderOperations() {
  const grid = $("#operationGrid");
  grid.replaceChildren();
  const operations = filteredOperations();
  const active = ensureActiveOperation(operations);

  if (!operations.length) {
    grid.append(el("div", "empty", "没有匹配操作"));
    return;
  }

  operations.forEach((op) => {
    const item = el("button", `operation-mini ${active?.id === op.id ? "active" : ""}`);
    item.type = "button";
    item.dataset.group = op.group;
    item.dataset.operationId = op.id;
    const head = el("div", "mini-head");
    const titleWrap = el("div");
    titleWrap.append(el("strong", "", op.title));
    titleWrap.append(el("small", "", op.description || ""));
    head.append(titleWrap, el("span", "badge", op.badge || op.group));
    const meta = el("div", "mini-meta");
    meta.append(
      el("span", "", op.group),
      el("span", "", `${(op.fields || []).length} 项参数`),
      el("span", "", op.long_running ? "长任务" : "即时任务"),
    );
    item.append(head, meta);
    item.addEventListener("click", () => {
      state.activeOperationId = op.id;
      renderOperations();
      renderOperationDetail();
    });
    grid.append(item);
  });
}

function renderOperationDetail() {
  const detail = $("#operationDetail");
  if (!detail) return;
  detail.replaceChildren();

  const operations = filteredOperations();
  const op = ensureActiveOperation(operations);
  if (!op) {
    detail.append(el("div", "empty", "请选择左侧分类或调整筛选条件"));
    return;
  }

  const card = el("article", "operation-detail-card");
  card.dataset.group = op.group;
  card.dataset.operationId = op.id;

  const head = el("div", "detail-head");
  const titleWrap = el("div");
  titleWrap.append(el("h3", "", op.title));
  titleWrap.append(el("p", "", op.description || ""));
  head.append(titleWrap, el("span", "badge", op.badge || op.group));

  const body = el("div", "operation-detail-body");
  const fields = el("div", "field-grid");
  (op.fields || []).forEach((field) => fields.append(renderField(op, field)));
  body.append(fields);

  // wire show_if controllers: when a checkbox toggles, show/hide dependent fields
  const conditionalFields = fields.querySelectorAll(".conditional-field");
  if (conditionalFields.length) {
    const controllerNames = [...new Set([...conditionalFields].map((f) => f.dataset.showIf))];
    controllerNames.forEach((ctrlName) => {
      const ctrlInput = fields.querySelector(`[name="${ctrlName}"]`);
      if (!ctrlInput) return;
      const syncAll = () => {
        const checked = ctrlInput.type === "checkbox" ? ctrlInput.checked : Boolean(ctrlInput.value);
        conditionalFields
          .filter((f) => f.dataset.showIf === ctrlName)
          .forEach((f) => { f.hidden = !checked; });
      };
      ctrlInput.addEventListener("change", syncAll);
    });
  }

  const actions = el("div", "detail-actions");
  const button = el("button", "primary full", op.long_running ? "启动" : "运行");
  button.type = "button";
  button.addEventListener("click", async () => {
    button.disabled = true;
    const oldText = button.textContent;
    button.textContent = "提交中";
    try {
      const job = await request("/api/jobs", {
        method: "POST",
        body: JSON.stringify({ operation_id: op.id, values: collectValues(card) }),
      });
      state.activeJobId = job.id;
      await refreshJobs();
      await refreshActiveJob();
    } catch (error) {
      alert(error.message);
    } finally {
      button.disabled = false;
      button.textContent = oldText;
    }
  });
  actions.append(button);

  card.append(head, body, actions);
  detail.append(card);
}

function renderHistory() {
  const history = $("#jobHistory");
  history.replaceChildren();
  if (!state.jobs.length) {
    history.append(el("div", "empty", "暂无任务"));
    return;
  }
  state.jobs.slice(0, 12).forEach((job) => {
    const item = el("button", "job-item");
    item.type = "button";
    const left = el("span");
    left.append(el("strong", "", job.title));
    left.append(el("small", "", `${fmtTime(job.started_at || job.created_at)} · ${job.id}`));
    item.append(left, el("span", `job-status ${job.status}`, statusLabel(job.status)));
    item.addEventListener("click", async () => {
      state.activeJobId = job.id;
      await refreshActiveJob();
    });
    history.append(item);
  });
}

function renderJob(job) {
  const meta = $("#jobMeta");
  const log = $("#logOutput");
  const cancel = $("#cancelJob");

  if (!job) {
    meta.textContent = "暂无任务";
    renderJobResult(null);
    renderResultPane(null);
    log.textContent = "";
    state.renderedJobId = null;
    cancel.disabled = true;
    return;
  }

  const code = job.exit_code === null || job.exit_code === undefined ? "" : ` · exit=${job.exit_code}`;
  const jobChanged = state.renderedJobId !== job.id;
  const wasNearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
  const previousScrollTop = log.scrollTop;
  const nextLog = (job.logs || []).join("");

  meta.textContent = `${job.title} · ${statusLabel(job.status)}${code} · ${job.command.join(" ")}`;
  renderJobResult(job);
  renderResultPane(job);
  if (jobChanged || log.textContent !== nextLog) {
    log.textContent = nextLog;
    if (jobChanged || wasNearBottom) {
      log.scrollTop = log.scrollHeight;
    } else {
      log.scrollTop = previousScrollTop;
    }
  }
  state.renderedJobId = job.id;
  cancel.disabled = !["running", "queued"].includes(job.status);
}

async function refreshStatus() {
  state.status = await request("/api/status");
  renderStatus();
}

async function refreshJobs() {
  const data = await request("/api/jobs");
  state.jobs = data.jobs || [];
  renderHistory();
}

async function refreshActiveJob() {
  if (!state.activeJobId) {
    renderJob(null);
    return;
  }
  try {
    const job = await request(`/api/jobs/${state.activeJobId}`);
    renderJob(job);
  } catch {
    state.activeJobId = null;
    renderJob(null);
  }
}

async function sendChat() {
  const output = $("#chatOutput");
  if (!output) return;
  output.textContent = "请求中...";
  try {
    const data = await request("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        port: $("#chatPort")?.value,
        message: $("#chatMessage")?.value,
      }),
    });
    const content = data?.choices?.[0]?.message?.content || JSON.stringify(data, null, 2);
    output.textContent = content;
  } catch (error) {
    output.textContent = error.message;
  }
}

async function init() {
  const opData = await request("/api/operations");
  state.groups = opData.groups || [];
  state.operations = opData.operations || [];
  renderGroups();
  renderOperations();
  renderOperationDetail();
  await Promise.all([refreshStatus(), refreshJobs()]);
  if (state.jobs[0]) {
    state.activeJobId = state.jobs[0].id;
    await refreshActiveJob();
  }

  $("#searchInput").addEventListener("input", (event) => {
    state.search = event.target.value;
    renderOperations();
    renderOperationDetail();
  });
  $("#refreshStatus").addEventListener("click", refreshStatus);
  $("#refreshJobs").addEventListener("click", refreshJobs);
  $("#clearSelection").addEventListener("click", () => {
    state.activeJobId = null;
    renderJob(null);
  });
  $("#cancelJob").addEventListener("click", async () => {
    if (!state.activeJobId) return;
    await request(`/api/jobs/${state.activeJobId}/cancel`, { method: "POST", body: "{}" });
    await refreshJobs();
    await refreshActiveJob();
  });
  $("#sendChat")?.addEventListener("click", sendChat);

  setInterval(async () => {
    await refreshJobs().catch(() => {});
    await refreshActiveJob().catch(() => {});
    await refreshStatus().catch(() => {});
  }, 1800);
}

init().catch((error) => {
  document.body.innerHTML = `<pre style="padding:24px;color:#a33b3b">${error.message}</pre>`;
});
