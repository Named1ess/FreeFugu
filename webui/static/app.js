const state = {
  groups: [],
  operations: [],
  status: null,
  jobs: [],
  activeGroup: "all",
  activeJobId: null,
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

  const grid = el("div", "slot-grid");
  const columns = [
    ["model", "模型", "openai/gpt-4o-mini", "text"],
    ["api_base", "API URL", "https://api.example.com/v1", "url"],
    ["api_key", "API Key", "sk-...", "password"],
  ];

  function updateVisibleRows() {
    wrapper.dataset.activeSlots = String(activeSlots);
    countText.textContent = `${activeSlots}/${maxSlots} slots`;
    grid.querySelectorAll(".slot-row").forEach((row) => {
      row.hidden = Number(row.dataset.slotIndex) >= activeSlots;
    });
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

function renderOperations() {
  const grid = $("#operationGrid");
  grid.replaceChildren();
  const query = state.search.trim().toLowerCase();
  const operations = state.operations.filter((op) => {
    const groupOk = state.activeGroup === "all" || op.group === state.activeGroup;
    const text = `${op.title} ${op.description} ${op.badge}`.toLowerCase();
    return groupOk && (!query || text.includes(query));
  });

  if (!operations.length) {
    grid.append(el("div", "empty", "没有匹配操作"));
    return;
  }

  operations.forEach((op) => {
    const card = el("article", "operation-card");
    if ((op.fields || []).some((field) => field.type === "slot_config")) {
      card.classList.add("wide-card");
    }
    card.dataset.group = op.group;
    card.dataset.operationId = op.id;
    const head = el("div", "card-head");
    const titleWrap = el("div");
    titleWrap.append(el("h3", "", op.title));
    head.append(titleWrap, el("span", "badge", op.badge || op.group));
    card.append(head, el("p", "", op.description || ""));

    const fields = el("div", "field-grid");
    (op.fields || []).forEach((field) => fields.append(renderField(op, field)));
    card.append(fields);

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
    card.append(button);
    grid.append(card);
  });
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
    log.textContent = "";
    cancel.disabled = true;
    return;
  }

  const code = job.exit_code === null || job.exit_code === undefined ? "" : ` · exit=${job.exit_code}`;
  meta.textContent = `${job.title} · ${statusLabel(job.status)}${code} · ${job.command.join(" ")}`;
  log.textContent = (job.logs || []).join("");
  log.scrollTop = log.scrollHeight;
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
  output.textContent = "请求中...";
  try {
    const data = await request("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        port: $("#chatPort").value,
        message: $("#chatMessage").value,
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
  await Promise.all([refreshStatus(), refreshJobs()]);
  if (state.jobs[0]) {
    state.activeJobId = state.jobs[0].id;
    await refreshActiveJob();
  }

  $("#searchInput").addEventListener("input", (event) => {
    state.search = event.target.value;
    renderOperations();
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
  $("#sendChat").addEventListener("click", sendChat);

  setInterval(async () => {
    await refreshJobs().catch(() => {});
    await refreshActiveJob().catch(() => {});
    await refreshStatus().catch(() => {});
  }, 1800);
}

init().catch((error) => {
  document.body.innerHTML = `<pre style="padding:24px;color:#a33b3b">${error.message}</pre>`;
});
