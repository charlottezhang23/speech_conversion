/* 本地访谈转录工作台 — 前端（无外部依赖，可完全离线） */

const state = {
  system: null,
  settings: null,
  projects: [],
  project: null,
  transcript: null,
  summary: null,
  job: null,
  tab: "transcribe",
  dirty: false,
  saving: false,
  currentTime: 0,
  summaryEditing: false,
  pollTimer: null,
  saveTimer: null,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function hhmmss(sec) {
  if (sec === null || sec === undefined || isNaN(sec)) return "--:--:--";
  const t = Math.max(0, Math.round(Number(sec)));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function fmtSize(n) {
  if (!n) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0, v = Number(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

async function api(path, options = {}) {
  const opts = { ...options };
  if (opts.json !== undefined) {
    opts.body = JSON.stringify(opts.json);
    opts.headers = { ...(opts.headers || {}), "Content-Type": "application/json" };
    delete opts.json;
  }
  const res = await fetch(path, opts);
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("application/json") ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) {
    const msg = (data && data.detail) || (typeof data === "string" ? data : "") || `请求失败 (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

function toast(message, isError = false) {
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " error" : "");
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), isError ? 6000 : 3000);
}

/* ------------------------------------------------------------------ markdown */
function inlineMd(text) {
  let s = esc(text);
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*(?!\s)(.+?)\*(?=[\s).,，。]|$)/g, "$1<em>$2</em>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\[(\d{2}:\d{2}:\d{2})\]/g, '<button class="ts-link" data-seek="$1">[$1]</button>');
  return s;
}

function mdToHtml(md) {
  const lines = String(md || "").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let listType = null;
  let inQuote = false;
  const closeList = () => { if (listType) { out.push(`</${listType}>`); listType = null; } };
  const closeQuote = () => { if (inQuote) { out.push("</blockquote>"); inQuote = false; } };

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) { closeList(); closeQuote(); continue; }

    if (line.startsWith("|") && line.endsWith("|")) {
      closeList(); closeQuote();
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|") && lines[i].trim().endsWith("|")) {
        const cells = lines[i].trim().slice(1, -1).split("|").map((c) => c.trim());
        if (!cells.every((c) => /^:?-{2,}:?$/.test(c))) rows.push(cells);
        i++;
      }
      i--;
      if (rows.length) {
        out.push("<table><thead><tr>" + rows[0].map((c) => `<th>${inlineMd(c)}</th>`).join("") + "</tr></thead><tbody>");
        rows.slice(1).forEach((r) => out.push("<tr>" + r.map((c) => `<td>${inlineMd(c)}</td>`).join("") + "</tr>"));
        out.push("</tbody></table>");
      }
      continue;
    }

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { closeList(); closeQuote(); out.push(`<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`); continue; }

    if (/^(-{3,}|\*{3,})$/.test(line)) { closeList(); closeQuote(); out.push("<hr/>"); continue; }

    if (line.startsWith("> ")) {
      closeList();
      if (!inQuote) { out.push("<blockquote>"); inQuote = true; }
      out.push(`<p>${inlineMd(line.slice(2))}</p>`);
      continue;
    }
    closeQuote();

    const ol = line.match(/^(\d+)[.)]\s+(.*)$/);
    const ul = line.match(/^[-*]\s+(.*)$/);
    if (ol) {
      if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; }
      out.push(`<li>${inlineMd(ol[2])}</li>`);
    } else if (ul) {
      if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; }
      out.push(`<li>${inlineMd(ul[1])}</li>`);
    } else {
      closeList();
      out.push(`<p>${inlineMd(line)}</p>`);
    }
  }
  closeList(); closeQuote();
  return out.join("\n");
}

/* ------------------------------------------------------------------- status */
function setPill(id, ok, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle("ok", !!ok);
  el.classList.toggle("bad", ok === false);
  el.innerHTML = `<span class="dot"></span>${esc(text)}`;
}

async function loadSystem() {
  try {
    state.system = await api("/api/system");
    const env = state.system.env || {};
    setPill("pill-ffmpeg", !!env.ffmpeg, env.ffmpeg ? "FFmpeg 就绪" : "缺少 FFmpeg");
    setPill("pill-ollama", state.system.ollama?.online, state.system.ollama?.online ? "Ollama 在线" : "Ollama 离线");
    setPill("pill-token", env.hf_token_set, env.hf_token_set ? "HF Token 已配置" : "HF Token 未配置");
  } catch (e) {
    toast("无法读取系统状态：" + e.message, true);
  }
}

async function loadProjects() {
  try {
    state.projects = await api("/api/projects");
  } catch (e) {
    state.projects = [];
  }
  renderSidebar();
}

function stageBadge(label, status) {
  const map = { done: "ok", running: "accent", failed: "danger", pending: "" };
  return `<span class="badge ${map[status] || ""}">${esc(label)}</span>`;
}

function renderSidebar() {
  const list = $("#project-list");
  if (!state.projects.length) {
    list.innerHTML = `<div class="muted" style="padding:14px;font-size:12.5px">还没有项目。点击右上「新建访谈」上传录音。</div>`;
    return;
  }
  list.innerHTML = state.projects.map((p) => {
    const active = state.project && state.project.id === p.id ? "active" : "";
    return `<div class="project-card ${active}" data-project="${p.id}">
      <div class="pc-title">${esc(p.name)}</div>
      <div class="pc-meta">
        <span>${hhmmss(p.duration_sec)}</span>
        <span>${fmtSize(p.size_bytes)}</span>
        <span>${esc(p.format || "")}</span>
      </div>
      <div class="pc-badges">
        ${stageBadge("转录", p.stages.transcribe)}
        ${p.has_draft ? stageBadge("初稿", "done") : ""}
        ${p.stages.diarize === "failed" ? `<span class="badge danger">分离失败</span>` : ""}
        ${p.has_edited ? `<span class="badge accent">已校对</span>` : ""}
        ${p.has_summary ? `<span class="badge ok">纪要</span>` : ""}
        ${p.busy ? `<span class="badge warn">运行中</span>` : ""}
      </div>
    </div>`;
  }).join("");
}

/* -------------------------------------------------------------------- open */
async function openProject(id, keepTab = false) {
  if (state.dirty && state.project && state.project.id !== id) {
    await saveTranscript({ silent: true });
  }
  clearInterval(state.pollTimer);
  state.job = null;
  try {
    state.project = await api(`/api/projects/${id}`);
  } catch (e) {
    toast("打开项目失败：" + e.message, true);
    return;
  }
  state.transcript = null;
  state.transcriptVersion = "effective";
  state.summary = null;
  state.summaryEditing = false;
  if (!keepTab) state.tab = "transcribe";

  try { state.transcript = await api(`/api/projects/${id}/transcript?version=${state.transcriptVersion || "effective"}`); } catch (e) { /* not transcribed yet */ }
  try { state.summary = await api(`/api/projects/${id}/summary`); } catch (e) { state.summary = null; }

  if (state.transcript && state.tab === "transcribe") state.tab = "transcript";
  if (state.summary && state.tab === "transcribe") state.tab = "summary";

  renderSidebar();
  renderMain();
  const active = await api("/api/jobs/active").catch(() => null);
  if (active && active.project_id === id) { state.job = active; startPolling(active.id); }
}

function renderMain() {
  const main = $("#main");
  if (!state.project) {
    main.innerHTML = `<div class="empty"><h3>还没有打开任何项目</h3><p>点击「新建访谈」上传一段录音，或从左侧选择一个历史项目。</p></div>`;
    return;
  }
  const p = state.project;
  const tabs = [
    ["transcribe", "音频转录"],
    ["transcript", "逐字稿"],
    ["summary", "AI 纪要"],
    ["export", "导出"],
  ];
  const nav = tabs.map(([k, label]) => {
    const disabled = (k === "transcript" && !state.transcript) || (k === "summary" && !state.transcript) ? "" : "";
    return `<button class="tab ${state.tab === k ? "active" : ""}" data-tab="${k}" ${disabled}>${label}</button>`;
  }).join("");

  let body = "";
  if (state.tab === "transcribe") body = viewTranscribe();
  else if (state.tab === "transcript") body = viewTranscript();
  else if (state.tab === "summary") body = viewSummary();
  else body = viewExport();

  main.innerHTML = `
    <div class="panel" style="display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap">
      <div style="flex:1 1 320px;min-width:0">
        <h3 style="margin-bottom:6px">${esc(p.name)}</h3>
        <div class="kv">
          <div class="k">原始文件</div><div class="v">${esc(p.audio?.original_name || "—")}</div>
          <div class="k">时长 / 大小</div><div class="v">${hhmmss(p.duration_sec)} · ${fmtSize(p.size_bytes)}</div>
          <div class="k">语言 / 模型</div><div class="v">${esc(p.settings?.language || "auto")} · ${esc(p.settings?.model || "—")}</div>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn btn-sm" data-action="rename-project">重命名</button>
        <button class="btn btn-sm btn-danger" data-action="delete-project">删除项目</button>
      </div>
    </div>
    <div class="tabs">${nav}</div>
    <div id="tab-body">${body}</div>`;

  afterRender();
}

/* -------------------------------------------------------------- transcribe */
function viewTranscribe() {
  const p = state.project;
  const s = { ...(state.settings || {}), ...(p.settings || {}) };
  const job = state.job;
  const running = job && (job.status === "running" || job.status === "queued");
  const hasDraft = !!state.transcript;
  const diarFailed = p.stages && p.stages.diarize === "failed";
  const diarErr = ((p.stage_details && p.stage_details.diarize && p.stage_details.diarize.error) || "").split("\n")[0];

  return `
  ${diarFailed ? `<div class="panel" style="border-left:4px solid var(--danger)">
    <h3 style="color:var(--danger);margin-bottom:6px">说话人分离失败</h3>
    <div class="muted" style="font-size:12.5px;white-space:pre-wrap;margin-bottom:8px">${esc(diarErr || "无错误信息")}</div>
    <div class="muted" style="font-size:12.5px">常见原因：未在 Hugging Face 模型页接受条款（403 gated）或 Token 无效。请在「设置」→「验证 Token 与模型授权」确认后，再点下方「仅重跑说话人分离」。</div>
  </div>` : ""}
  <div class="panel">
    <h3>识别设置 <span class="hint">修改后点击开始，不会重复转码已完成的分段</span></h3>
    <div class="grid-2">
      <div class="field">
        <label>识别语言</label>
        <select id="set-language">
          <option value="auto" ${(s.language || "auto") === "auto" ? "selected" : ""}>自动检测</option>
          <option value="zh" ${s.language === "zh" ? "selected" : ""}>中文（普通话）</option>
          <option value="ja" ${s.language === "ja" ? "selected" : ""}>日语</option>
          <option value="en" ${s.language === "en" ? "selected" : ""}>英语</option>
        </select>
        <span class="desc">中日混说时建议手动指定主语言</span>
      </div>
      <div class="field">
        <label>Whisper 模型</label>
        <select id="set-model">
          ${["medium", "large-v3"].map((m) => `<option value="${m}" ${(s.model || "medium") === m ? "selected" : ""}>${m}${m === "medium" ? "（更快，约 1.5GB）" : "（更准，约 3GB）"}</option>`).join("")}
        </select>
      </div>
      <div class="field">
        <label>说话人数量</label>
        <select id="set-speakers">
          <option value="auto" ${!s.num_speakers ? "selected" : ""}>自动检测（2–5 人）</option>
          ${[1, 2, 3, 4, 5].map((n) => `<option value="${n}" ${Number(s.num_speakers) === n ? "selected" : ""}>固定 ${n} 人</option>`).join("")}
        </select>
        <span class="desc">两人访谈建议选「固定 2 人」，效果更稳定</span>
      </div>
      <div class="field">
        <label>CPU 线程数</label>
        <input type="number" id="set-threads" min="1" max="${(state.system?.env?.cpu_count || 16)}" value="${s.threads || 8}" />
        <span class="desc">本机逻辑核心 ${state.system?.env?.cpu_count || "?"} 个</span>
      </div>
      <div class="field">
        <label>计算精度</label>
        <select id="set-compute">
          ${["int8", "float32", "int8_float32"].map((m) => `<option value="${m}" ${(s.compute_type || "int8") === m ? "selected" : ""}>${m}</option>`).join("")}
        </select>
      </div>
      <div class="field">
        <label>批大小 / Beam</label>
        <div class="row" style="gap:6px">
          <input type="number" id="set-batch" min="1" max="32" value="${s.batch_size || 8}" title="batch size" />
          <input type="number" id="set-beam" min="1" max="10" value="${s.beam_size || 5}" title="beam size" />
        </div>
      </div>
      <div class="field">
        <label>说话人分离</label>
        <label class="muted" style="display:flex;gap:8px;align-items:center;font-size:12.5px">
          <input type="checkbox" id="set-diarize" ${s.diarize_enabled !== false ? "checked" : ""} />
          使用 pyannote Community-1（需要 HF Token）
        </label>
        <label class="muted" style="display:flex;gap:8px;align-items:center;font-size:12.5px">
          <input type="checkbox" id="set-align" ${s.align ? "checked" : ""} />
          强制对齐（更准的词级时间戳，更慢）
        </label>
      </div>
    </div>
    <div class="toolbar mt">
      <button class="btn btn-primary" data-action="start-transcribe" ${running ? "disabled" : ""}>${hasDraft ? "重新转录" : "开始转录"}</button>
      ${hasDraft ? `<button class="btn" data-action="rebuild-draft" ${running ? "disabled" : ""} title="用已缓存的识别结果重新合并逐字稿，并清理提示词复读/重复循环段落">重建逐字稿（清理异常段落）</button>` : ""}
      ${hasDraft ? `<button class="btn" data-action="align-draft" ${running ? "disabled" : ""} title="对已有识别结果做强制对齐（不重新识别），再按词级时间点切分说话人。约 0.5 倍录音时长，首次需下载 2.5GB 对齐模型">词级对齐（改善切分点）</button>` : ""}
      ${hasDraft ? `<button class="btn" data-action="refine-speakers" ${running ? "disabled" : ""} title="重新提取每个发言片段的声纹并全局聚类，纠正 pyannote 偶发的两人短暂对调。不重新识别、不重跑分离，约 1–2 分钟">二次声纹校验（纠正归属）</button>` : ""}
      ${hasDraft ? `<button class="btn" data-action="rerun-diarize" ${running ? "disabled" : ""}>仅重跑说话人分离</button>` : ""}
      ${running ? `<button class="btn btn-danger" data-action="cancel-job">取消任务</button>` : ""}
      <div class="spacer"></div>
      <span class="muted" style="font-size:12px">识别与分离都是本机 CPU 推理，${hhmmss(p.duration_sec)} 的录音通常需要数十倍于时长的处理时间</span>
    </div>
  </div>

  <div class="panel">
    <h3>进度与日志</h3>
    <div id="job-progress">${renderProgress(job)}</div>
    <div class="log mt" id="job-log">${esc((job?.logs || []).join("\n") || "尚无日志。点击「开始转录」后在此显示每一步的进度。")}</div>
  </div>

  <div class="panel">
    <h3>阶段耗时</h3>
    <div class="kv">
      ${Object.entries(p.timings || {}).map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${v.elapsed_sec ?? "—"} 秒 ${v.segments ? `· ${v.segments} 句` : ""}${v.utterances ? ` · ${v.utterances} 段` : ""}${v.speakers ? ` · ${v.speakers} 人` : ""}</div>`).join("") || `<div class="k">—</div><div class="v muted">尚未运行</div>`}
    </div>
  </div>`;
}

function renderProgress(job) {
  if (!job) return `<div class="progress-wrap"><div class="progress"><div style="width:0"></div></div><div class="progress-meta"><span>空闲</span><span></span></div></div>`;
  const statusText = { queued: "排队中", running: "运行中", done: "已完成", failed: "失败", cancelled: "已取消" }[job.status] || job.status;
  const cls = job.status === "failed" ? "danger" : job.status === "done" ? "ok" : "accent";
  const started = job.started_at ? new Date(job.started_at).getTime() : null;
  const elapsed = started ? Math.max(0, Math.round((Date.now() - started) / 1000)) : 0;
  const elapsedText = job.status === "running" || job.status === "queued" ? ` · 已运行 ${elapsed}s` : "";
  return `<div class="progress-wrap">
    <div class="progress"><div style="width:${job.progress || 0}%"></div></div>
    <div class="progress-meta">
      <span><span class="badge ${cls}">${statusText}</span> ${esc(job.stage || "")} — ${esc(job.message || "")}</span>
      <span>${Math.round(job.progress || 0)}%${elapsedText}</span>
    </div>
    ${job.error ? `<div class="badge danger">${esc(job.error)}</div>` : ""}
  </div>`;
}

/* --------------------------------------------------------------transcript */
function viewTranscript() {
  const t = state.transcript;
  if (!t) return `<div class="empty"><h3>还没有逐字稿</h3><p>先去「音频转录」运行识别。</p></div>`;
  const p = state.project;
  const names = t.speaker_names || {};
  const speakers = Object.keys(names);
  const counts = {};
  t.utterances.forEach((u) => { counts[u.speaker] = (counts[u.speaker] || 0) + 1; });

  const chips = speakers.map((spk, i) => `
    <div class="speaker-chip">
      <span class="tag" style="background:${speakerColor(spk)}"></span>
      <input type="text" data-speaker-name="${esc(spk)}" value="${esc(names[spk] || spk)}" />
      <span class="count">${counts[spk] || 0} 段</span>
    </div>`).join("");

  const diarErr = (p.stage_details && p.stage_details.diarize && p.stage_details.diarize.error) || "";
  const diarFailed = p.stages && p.stages.diarize === "failed";
  const busy = state.job && (state.job.status === "running" || state.job.status === "queued");
  const diarBanner = diarFailed ? `
    <div class="panel" style="border-left:4px solid var(--danger)">
      <h3 style="color:var(--danger);margin-bottom:8px">说话人分离未成功，这份逐字稿没有说话人区分</h3>
      <div class="muted" style="font-size:12.5px;white-space:pre-wrap;margin-bottom:10px;max-height:150px;overflow:auto">${esc(diarErr || "没有记录到具体错误")}</div>
      <div class="muted" style="font-size:12.5px;margin-bottom:12px;line-height:1.7">
        最常见的原因是 Hugging Face 的 gated 模型授权没有生效（403 gated）：<br/>
        1) 用<strong>同一个账号</strong>打开 <a href="https://huggingface.co/pyannote/speaker-diarization-community-1" target="_blank" rel="noreferrer">模型页面</a> 并接受条款；<br/>
        2) 在「设置」里点「验证 Token」，确认显示「已具备访问权限」；<br/>
        3) 再点下面的按钮重跑分离（不会重新识别音频）。
      </div>
      <button class="btn btn-primary btn-sm" data-action="rerun-diarize" ${busy ? "disabled" : ""}>${busy ? "任务运行中…" : "重跑说话人分离"}</button>
    </div>` : "";

  const staleBanner = t.draft_stale ? `
    <div class="panel" style="border-left:4px solid var(--warn,#d97706)">
      <h3 style="margin-bottom:8px">机器初稿已更新</h3>
      <div class="muted" style="font-size:12.5px;line-height:1.7;margin-bottom:10px">
        对齐 / 重建 / 声纹校验只重写<strong>机器初稿</strong>，不会改动你的人工校对版。
        当前显示的还是校对版（基于旧初稿），所以看不到这些变化。
        在新初稿上编辑会以新初稿为基准重新保存校对版。
      </div>
      <button class="btn btn-primary btn-sm" data-action="view-draft">查看新初稿</button>
    </div>` : "";

  const rows = t.utterances.map((u) => {
    const flags = (u.flags || []).filter((f) => f !== "nearest_speaker");
    const flagHtml = flags.map((f) => `<span class="flag">${esc(flagLabel(f))}</span>`).join("");
    return `<div class="utterance" data-uid="${esc(u.id)}" data-start="${u.start}" data-end="${u.end}">
      <div class="ts" data-seek="${u.start}">${hhmmss(u.start)}</div>
      <div class="body">
        <div class="head">
          <select class="speaker-select" data-speaker-for="${esc(u.id)}" style="border-left:4px solid ${speakerColor(u.speaker)}">
            ${speakers.map((spk) => `<option value="${esc(spk)}" ${spk === u.speaker ? "selected" : ""}>${esc(names[spk] || spk)}</option>`).join("")}
          </select>
          ${flagHtml}
          ${u.edited ? `<span class="badge accent">已修改</span>` : ""}
        </div>
        <textarea class="utt-text" data-text-for="${esc(u.id)}" rows="1">${esc(u.text)}</textarea>
      </div>
    </div>`;
  }).join("");

  return `
  <div class="player-bar">
    <audio id="audio-player" controls preload="metadata" src="/api/projects/${p.id}/audio"></audio>
    <div class="now" id="now-time">00:00:00</div>
  </div>

  ${diarBanner}
  ${staleBanner}

  <div class="toolbar">
    <span class="badge ${t.version === "edited" ? "accent" : ""}">${t.version === "edited" ? "人工校对版" : "机器初稿"}</span>
    ${t.has_edited ? `<span class="badge ok">已保存人工校对</span>` : ""}
    <select id="transcript-version" title="查看哪一版逐字稿" style="max-width:170px">
      <option value="effective" ${(state.transcriptVersion || "effective") === "effective" ? "selected" : ""}>自动（优先校对版）</option>
      <option value="draft" ${state.transcriptVersion === "draft" ? "selected" : ""}>机器初稿</option>
      ${t.has_edited ? `<option value="edited" ${state.transcriptVersion === "edited" ? "selected" : ""}>人工校对版</option>` : ""}
    </select>
    ${t.draft_stale ? `<span class="badge warn">初稿已更新，校对版基于旧初稿</span>` : ""}
    <span class="muted" style="font-size:12px">共 ${t.utterances.length} 段 · ${t.review?.uncertain_speaker || 0} 段说话人不确定 · ${t.review?.overlap || 0} 段疑似重叠 · ${t.review?.low_confidence || 0} 段置信度低</span>
    ${(t.review?.filtered_echo || 0) + (t.review?.filtered_loop || 0) > 0 ? `<span class="badge warn" title="提示词复读或重复循环，已在初稿中过滤；原始识别结果仍保存在 asr\\chunks">已过滤 ${(t.review.filtered_echo || 0) + (t.review.filtered_loop || 0)} 段异常识别</span>` : ""}
    <div class="spacer"></div>
    <span class="save-state" id="save-state">${state.dirty ? "有未保存修改" : ""}</span>
    <button class="btn btn-sm" data-action="save-transcript">保存</button>
    ${t.has_edited ? `<button class="btn btn-sm btn-danger" data-action="reset-transcript">放弃校对版</button>` : ""}
  </div>

  <div class="panel">
    <h3>说话人 <span class="hint">可直接改名（如「访谈者」「受访者」），保存后同步到导出与纪要</span></h3>
    <div class="speaker-chips">${chips}</div>
  </div>

  <div class="panel">
    <div class="utterances" id="utterances">${rows}</div>
  </div>`;
}

function flagLabel(f) {
  return {
    speaker_uncertain: "说话人不确定",
    overlap: "疑似重叠",
    low_confidence: "低置信度",
    no_diarization: "无分离结果",
    nearest_speaker: "按最近片段推断",
  }[f] || f;
}

const SPEAKER_COLORS = ["#2f6df6", "#0f9d70", "#c026d3", "#d97706", "#0891b2", "#dc2626", "#7c3aed", "#65a30d"];
function speakerColor(spk) {
  let h = 0;
  for (const ch of String(spk)) h = (h * 31 + ch.charCodeAt(0)) % 997;
  return SPEAKER_COLORS[h % SPEAKER_COLORS.length];
}

/* ----------------------------------------------------------------- summary */
function viewSummary() {
  const p = state.project;
  const s = state.settings || {};
  const summary = state.summary;
  const models = (state.ollamaModels || []).map((m) => m.name);
  const curModel = summary?.model || s.ollama_model || "qwen3:8b";
  const curTemplate = summary?.template || "interview";
  const job = state.job;
  const running = job && (job.status === "running" || job.status === "queued");
  const templates = state.system?.templates || {};

  const modelOptions = (models.length ? models : [curModel]).map((m) =>
    `<option value="${esc(m)}" ${m === curModel ? "selected" : ""}>${esc(m)}</option>`).join("");

  const verification = summary?.verification;
  const issues = (verification?.unsupported || []);
  const verifyHtml = verification ? `
    <div class="panel">
      <h3>依据核查 <span class="hint">${verification.verdict === "ok" ? "未发现无依据的结论" : `${issues.length} 条需要人工复核`}</span></h3>
      ${issues.length ? `<div class="issue-list">${issues.map((it) => `
        <div class="issue">
          <div class="claim">${inlineMd(it.claim)}</div>
          <div class="reason">${esc(it.reason)}${it.suggestion ? " · 建议：" + esc(it.suggestion) : ""}</div>
        </div>`).join("")}</div>` : `<div class="muted">模型未发现明显无依据的结论。仍建议人工抽查关键数字。</div>`}
      ${verification.notes ? `<div class="muted mt" style="font-size:12px">${esc(verification.notes)}</div>` : ""}
    </div>` : "";

  const summaryHtml = summary ? `
    ${state.summaryEditing ? `
      <div class="panel">
        <h3>编辑纪要（Markdown）</h3>
        <textarea id="summary-editor" rows="26" class="mono" style="font-size:12.5px">${esc(summary.markdown)}</textarea>
        <div class="toolbar mt">
          <button class="btn btn-primary btn-sm" data-action="save-summary">保存修改</button>
          <button class="btn btn-sm" data-action="cancel-summary-edit">取消</button>
          <span class="muted" style="font-size:12px">人工修改不会影响原始逐字稿，也不会阻止重新生成</span>
        </div>
      </div>` : `
      <div class="markdown" id="summary-md">${mdToHtml(summary.markdown)}</div>`}
    ${verifyHtml}` : `<div class="empty"><h3>还没有纪要</h3><p>选择模板与本地模型后点击「生成纪要」。逐字稿会按时间戳分段送入模型，每段结果都会保存，失败可续跑。</p></div>`;

  return `
  <div class="panel">
    <h3>纪要设置</h3>
    <div class="grid-2">
      <div class="field">
        <label>模板</label>
        <select id="sum-template">
          ${Object.entries(templates).map(([k, v]) => `<option value="${k}" ${k === curTemplate ? "selected" : ""}>${esc(v.label)}</option>`).join("")}
        </select>
        <span class="desc">${esc(templates[curTemplate]?.description || "")}</span>
      </div>
      <div class="field">
        <label>本地 Ollama 模型</label>
        <select id="sum-model">${modelOptions}</select>
        <span class="desc">${state.system?.ollama?.online ? "Ollama 在线" : "Ollama 未连接，请先启动 ollama serve"}</span>
      </div>
      <div class="field">
        <label>证据核查</label>
        <label class="muted" style="display:flex;gap:8px;align-items:center;font-size:12.5px">
          <input type="checkbox" id="sum-verify" ${s.verify_summary !== false ? "checked" : ""} /> 生成后对照逐字稿检查依据
        </label>
        <label class="muted" style="display:flex;gap:8px;align-items:center;font-size:12.5px">
          <input type="checkbox" id="sum-force" /> 忽略缓存，重新逐段抽取
        </label>
      </div>
    </div>
    <div class="toolbar mt">
      <button class="btn btn-primary" data-action="generate-summary" ${running ? "disabled" : ""}>${summary ? "重新生成纪要" : "生成纪要"}</button>
      ${summary ? `<button class="btn" data-action="toggle-summary-edit">${state.summaryEditing ? "退出编辑" : "编辑纪要"}</button>` : ""}
      ${summary ? `<a class="btn" href="/api/projects/${p.id}/summary.md" target="_blank">下载 Markdown</a>` : ""}
      ${running ? `<button class="btn btn-danger" data-action="cancel-job">取消任务</button>` : ""}
      <div class="spacer"></div>
      ${summary ? `<span class="muted" style="font-size:12px">模型 ${esc(summary.model)} · ${esc(summary.generated_at)} · 依据 ${summary.stats?.facts || 0} 条 · 来源：${summary.source === "edited" ? "人工校对逐字稿" : "机器初稿"}</span>` : ""}
    </div>
    ${running ? `<div class="mt" id="job-progress">${renderProgress(job)}</div><div class="log mt" id="job-log">${esc((job.logs || []).join("\n"))}</div>` : ""}
  </div>
  ${summaryHtml}`;
}

/* ------------------------------------------------------------------ export */
function viewExport() {
  const p = state.project;
  const hasEdited = !!state.transcript?.has_edited;
  const formats = [
    ["txt", "纯文本 TXT", "带时间戳与说话人的逐字稿"],
    ["md", "Markdown", "逐字稿 + 已生成的纪要"],
    ["srt", "字幕 SRT", "可导入剪映 / Premiere / YouTube"],
    ["docx", "Word DOCX", "排版好的逐字稿与纪要"],
    ["json", "原始 JSON", "完整结构化数据（含词级时间戳）"],
  ];
  return `
  <div class="panel">
    <h3>导出 <span class="hint">文件同时保存在项目目录 exports/ 下</span></h3>
    <div class="field" style="max-width:280px;margin-bottom:14px">
      <label>逐字稿版本</label>
      <select id="export-version">
        <option value="effective" ${hasEdited ? "selected" : ""}>自动（优先校对版）</option>
        <option value="draft">机器初稿</option>
        ${hasEdited ? `<option value="edited">人工校对版</option>` : ""}
      </select>
    </div>
    <div class="grid-2">
      ${formats.map(([fmt, label, desc]) => `
        <div class="panel" style="margin:0;padding:12px 14px">
          <div style="font-weight:600;margin-bottom:2px">${label}</div>
          <div class="muted" style="font-size:12px;margin-bottom:10px">${desc}</div>
          <button class="btn btn-sm" data-export="${fmt}">导出 ${fmt.toUpperCase()}</button>
        </div>`).join("")}
    </div>
  </div>
  <div class="panel">
    <h3>数据位置</h3>
    <div class="kv">
      <div class="k">项目目录</div><div class="v mono">${esc(p.id)}</div>
      <div class="k">原始音频</div><div class="v">保留在项目 audio/ 下，删除项目时一并删除</div>
      <div class="k">隐私</div><div class="v">仅监听 127.0.0.1；无遥测、无云 API</div>
    </div>
  </div>`;
}

/* --------------------------------------------------------------- rendering */
function afterRender() {
  $$("#utterances textarea.utt-text").forEach((ta) => autosize(ta));
  const player = $("#audio-player");
  if (player) {
    player.addEventListener("timeupdate", onTimeUpdate);
    player.addEventListener("loadedmetadata", () => { player.currentTime = state.currentTime || 0; });
  }
  const log = $("#job-log");
  if (log) log.scrollTop = log.scrollHeight;
}

function autosize(el) {
  el.style.height = "auto";
  el.style.height = el.scrollHeight + "px";
}

function onTimeUpdate(e) {
  const t = e.target.currentTime;
  state.currentTime = t;
  const now = $("#now-time");
  if (now) now.textContent = hhmmss(t);
  const rows = $$("#utterances .utterance");
  for (const row of rows) {
    const s = Number(row.dataset.start), en = Number(row.dataset.end);
    const on = t >= s && t < en;
    row.classList.toggle("playing", on);
  }
}

function seekTo(seconds, play = true) {
  if (state.tab !== "transcript") {
    state.tab = "transcript";
    renderMain();
  }
  const player = $("#audio-player");
  if (!player) return;
  player.currentTime = Math.max(0, Number(seconds) || 0);
  state.currentTime = player.currentTime;
  if (play) player.play().catch(() => {});
  const row = $(`#utterances .utterance[data-start="${Number(seconds)}"]`);
  if (row) row.scrollIntoView({ block: "center", behavior: "smooth" });
}

/* ------------------------------------------------------------------- save */
function markDirty() {
  state.dirty = true;
  const el = $("#save-state");
  if (el) el.textContent = "有未保存修改";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => saveTranscript({ silent: true }), 1500);
}

async function saveTranscript({ silent = false } = {}) {
  if (!state.project || !state.transcript) return;
  if (!state.dirty && silent) return;
  if (state.saving) return;
  state.saving = true;
  const el = $("#save-state");
  if (el) el.textContent = "保存中…";
  try {
    const payload = {
      utterances: state.transcript.utterances.map((u) => ({
        id: u.id, start: u.start, end: u.end, speaker: u.speaker, text: u.text, flags: u.flags,
      })),
      speaker_names: state.transcript.speaker_names,
    };
    const res = await api(`/api/projects/${state.project.id}/transcript`, { method: "PUT", json: payload });
    state.dirty = false;
    if (el) el.textContent = `已保存 ${new Date().toLocaleTimeString()}`;
    state.transcript.version = "edited";
    state.transcript.has_edited = true;
    if (!silent) toast("已保存");
    return res;
  } catch (e) {
    if (el) el.textContent = "保存失败";
    toast("保存失败：" + e.message, true);
  } finally {
    state.saving = false;
  }
}

/* ------------------------------------------------------------------ events */
document.addEventListener("click", async (e) => {
  const target = e.target.closest("[data-project],[data-tab],[data-action],[data-seek],[data-export],[data-speaker]");
  if (!target) return;

  if (target.dataset.project) {
    if (state.project?.id !== target.dataset.project) openProject(target.dataset.project);
    return;
  }
  if (target.dataset.tab) {
    if (state.dirty) await saveTranscript({ silent: true });
    state.tab = target.dataset.tab;
    renderMain();
    return;
  }
  if (target.dataset.seek) {
    seekTo(parseFloat(target.dataset.seek));
    return;
  }
  if (target.dataset.export) {
    const version = $("#export-version")?.value || "effective";
    const url = `/api/projects/${state.project.id}/export/${target.dataset.export}?version=${version}`;
    window.location.href = url;
    return;
  }
  const action = target.dataset.action;
  if (!action) return;

  try {
    if (action === "start-transcribe") await startTranscribe();
    else if (action === "rebuild-draft") await rebuildDraft();
    else if (action === "align-draft") await alignDraft();
    else if (action === "refine-speakers") await refineSpeakers();
    else if (action === "view-draft") await setTranscriptVersion("draft");
    else if (action === "rerun-diarize") await startDiarize();
    else if (action === "cancel-job") await cancelJob();
    else if (action === "save-transcript") await saveTranscript();
    else if (action === "reset-transcript") await resetTranscript();
    else if (action === "generate-summary") await generateSummary();
    else if (action === "toggle-summary-edit") { if (state.dirtySummary) await saveSummary(); state.summaryEditing = !state.summaryEditing; renderMain(); }
    else if (action === "save-summary") await saveSummary();
    else if (action === "cancel-summary-edit") { state.summaryEditing = false; renderMain(); }
    else if (action === "rename-project") await renameProject();
    else if (action === "delete-project") await deleteProject();
  } catch (err) {
    toast(err.message, true);
  }
});

document.addEventListener("input", (e) => {
  const ta = e.target.closest("textarea[data-text-for]");
  if (ta) {
    autosize(ta);
    const uid = ta.dataset.textFor;
    const u = state.transcript.utterances.find((x) => x.id === uid);
    if (u) { u.text = ta.value; u.edited = true; }
    markDirty();
    return;
  }
  const nameInput = e.target.closest("input[data-speaker-name]");
  if (nameInput) {
    const spk = nameInput.dataset.speakerName;
    state.transcript.speaker_names[spk] = nameInput.value;
    markDirty();
    $$(`select[data-speaker-for] option`).forEach((opt) => {
      if (opt.value === spk) opt.textContent = nameInput.value || spk;
    });
  }
});

document.addEventListener("change", (e) => {
  if (e.target.id === "transcript-version") {
    setTranscriptVersion(e.target.value);
    return;
  }
  const sel = e.target.closest("select[data-speaker-for]");
  if (sel) {
    const uid = sel.dataset.speakerFor;
    const u = state.transcript.utterances.find((x) => x.id === uid);
    if (u) { u.speaker = sel.value; }
    sel.style.borderLeftColor = speakerColor(sel.value);
    markDirty();
    return;
  }
  const nameInput = e.target.closest("input[data-speaker-name]");
  if (nameInput) saveTranscript({ silent: true });
});

/* ------------------------------------------------------------- job control */
async function collectSettings() {
  const s = {};
  const lang = $("#set-language")?.value;
  if (lang) s.language = lang;
  s.model = $("#set-model")?.value;
  s.compute_type = $("#set-compute")?.value;
  s.threads = Number($("#set-threads")?.value || 8);
  s.batch_size = Number($("#set-batch")?.value || 8);
  s.beam_size = Number($("#set-beam")?.value || 5);
  s.align = !!$("#set-align")?.checked;
  s.diarize_enabled = !!$("#set-diarize")?.checked;
  const spk = $("#set-speakers")?.value;
  s.num_speakers = spk === "auto" ? null : Number(spk);
  s.min_speakers = spk === "auto" ? 2 : null;
  s.max_speakers = spk === "auto" ? 5 : null;
  return s;
}

async function startTranscribe() {
  const settings = await collectSettings();
  if (settings.diarize_enabled && !state.system?.env?.hf_token_set) {
    toast("未配置 Hugging Face Token，将只做语音识别（可在设置中填入 Token 后重跑说话人分离）", true);
  }
  await saveTranscript({ silent: true });
  const job = await api(`/api/projects/${state.project.id}/transcribe`, { method: "POST", json: { settings } });
  state.job = job;
  state.tab = "transcribe";
  renderMain();
  startPolling(job.id);
}

async function rebuildDraft() {
  const job = await api(`/api/projects/${state.project.id}/rebuild`, { method: "POST" });
  state.job = job;
  state.tab = "transcribe";
  renderMain();
  startPolling(job.id);
}

async function alignDraft() {
  const mins = Math.max(1, Math.round((state.project.duration_sec || 0) / 60 * 0.3));
  if (!confirm(`词级对齐会重新跑一遍声学对齐（不重新识别文字），不会覆盖你已校对的内容。\n预计耗时约 ${mins} 分钟（首次还要下载约 2.5GB 对齐模型）。继续？`)) return;
  const job = await api(`/api/projects/${state.project.id}/align`, { method: "POST" });
  state.job = job;
  state.tab = "transcribe";
  renderMain();
  startPolling(job.id);
}

async function setTranscriptVersion(version) {
  if (state.dirty) await saveTranscript({ silent: true });
  state.transcriptVersion = version;
  try {
    state.transcript = await api(`/api/projects/${state.project.id}/transcript?version=${version}`);
  } catch (e) {
    toast("切换版本失败：" + e.message, true);
    return;
  }
  state.tab = "transcript";
  renderMain();
}

async function refineSpeakers() {
  const job = await api(`/api/projects/${state.project.id}/refine`, { method: "POST" });
  state.job = job;
  state.tab = "transcribe";
  renderMain();
  startPolling(job.id);
}

async function startDiarize() {
  const form = $("#set-speakers");
  let payload;
  if (form) {
    const settings = await collectSettings();
    payload = { num_speakers: settings.num_speakers, min_speakers: settings.min_speakers, max_speakers: settings.max_speakers };
  } else {
    // 从逐字稿页重跑时没有设置表单，沿用项目已保存的设置
    const ps = (state.project && state.project.settings) || {};
    payload = { num_speakers: ps.num_speakers ?? null, min_speakers: ps.min_speakers ?? null, max_speakers: ps.max_speakers ?? null };
  }
  const job = await api(`/api/projects/${state.project.id}/diarize`, { method: "POST", json: payload });
  state.job = job;
  state.tab = "transcribe";
  renderMain();
  startPolling(job.id);
}

async function cancelJob() {
  if (!state.job) return;
  await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" });
  toast("已请求取消，已完成的中间结果会保留");
}

function startPolling(jobId) {
  clearInterval(state.pollTimer);
  let failures = 0;
  state.pollTimer = setInterval(async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
      failures = 0;
    } catch (err) {
      failures += 1;
      // 服务重启后旧任务会消失，停止轮询并刷新界面
      if (failures >= 3) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        state.job = null;
        if (state.project) openProject(state.project.id, true);
        toast("任务已失效（服务可能已重启），已刷新项目状态", true);
      }
      return;
    }
    state.job = job;
    if (state.tab === "transcribe" || state.tab === "summary") {
      // 只更新进度块，避免整页重绘打断编辑与点击
      const prog = $("#job-progress");
      if (prog) {
        prog.innerHTML = renderProgress(job);
        const log = $("#job-log");
        if (log) {
          const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
          log.textContent = (job.logs || []).join("\n");
          if (atBottom) log.scrollTop = log.scrollHeight;
        }
      } else {
        const body = $("#tab-body");
        if (body) {
          body.innerHTML = state.tab === "transcribe" ? viewTranscribe() : viewSummary();
          afterRender();
        }
      }
    }
    if (job.status === "done" || job.status === "failed" || job.status === "cancelled") {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      await loadProjects();
      if (job.status === "done") {
        const id = state.project.id;
        const tab = state.tab;
        await openProject(id, true);
        if (job.kind === "summary") state.tab = "summary";
        else if (job.kind === "transcribe" && state.transcript) state.tab = "transcript";
        else state.tab = tab;
        renderMain();
        const doneMsg = {
          summary: "纪要已生成",
          transcribe: "转录完成",
          align: "词级对齐完成，机器初稿已更新",
          refine: "声纹校验完成，机器初稿已更新",
          rebuild: "逐字稿已重建",
          diarize: "说话人分离已更新",
        }[job.kind] || "完成";
        toast(doneMsg);
      } else if (job.status === "failed") {
        toast("任务失败：" + (job.error || ""), true);
      } else {
        toast("任务已取消，已完成的部分已保存");
      }
    }
  }, 1200);
}

/* ------------------------------------------------------------------ summary */
async function generateSummary() {
  const template = $("#sum-template")?.value || "interview";
  const model = $("#sum-model")?.value;
  const verify = !!$("#sum-verify")?.checked;
  const force = !!$("#sum-force")?.checked;
  state.summaryEditing = false;
  const job = await api(`/api/projects/${state.project.id}/summary`, {
    method: "POST",
    json: { template, model, verify, force },
  });
  state.job = job;
  state.tab = "summary";
  renderMain();
  startPolling(job.id);
}

async function saveSummary() {
  const editor = $("#summary-editor");
  if (!editor) return;
  const markdown = editor.value;
  await api(`/api/projects/${state.project.id}/summary`, { method: "PUT", json: { markdown } });
  state.summary.markdown = markdown;
  state.summaryEditing = false;
  renderMain();
  toast("纪要已保存");
}

/* ------------------------------------------------------------------ project */
function modal(html) {
  const root = $("#modal-root");
  root.innerHTML = `<div class="modal-mask"><div class="modal">${html}</div></div>`;
  root.querySelector(".modal-mask").addEventListener("click", (e) => {
    if (e.target.classList.contains("modal-mask")) closeModal();
  });
}
function closeModal() { $("#modal-root").innerHTML = ""; }

function openNewProject() {
  modal(`
    <h2>新建访谈</h2>
    <div class="sub">支持 M4A / MP3 / WAV / FLAC / MP4 等格式，录音只保存在本机。</div>
    <div class="field" style="margin-bottom:14px">
      <label>项目名称（可留空，默认使用文件名）</label>
      <input type="text" id="np-name" placeholder="例如：宠物用品访谈-01" />
    </div>
    <div class="dropzone" id="np-drop">
      <strong>拖放音频文件到此处，或点击选择</strong>
      <span class="muted">单个文件，建议 ≤ 2 小时</span>
      <input type="file" id="np-file" accept="audio/*,video/mp4,.m4a,.mp3,.wav,.flac,.aac,.ogg,.opus,.wma,.mp4,.mov,.webm" style="display:none" />
    </div>
    <div id="np-status" class="muted mt" style="font-size:12.5px"></div>
    <div class="actions">
      <button class="btn" onclick="closeModal()">取消</button>
      <button class="btn btn-primary" id="np-create" disabled>上传并创建</button>
    </div>`);
  const drop = $("#np-drop"), fileInput = $("#np-file"), btn = $("#np-create");
  let selected = null;
  const pick = (f) => {
    selected = f;
    $("#np-status").innerHTML = `已选择：<b>${esc(f.name)}</b> · ${fmtSize(f.size)}`;
    btn.disabled = false;
  };
  drop.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) pick(fileInput.files[0]); });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });

  btn.addEventListener("click", async () => {
    if (!selected) return;
    btn.disabled = true;
    btn.textContent = "上传中…";
    const form = new FormData();
    form.append("file", selected);
    form.append("name", $("#np-name").value || "");
    try {
      const res = await api("/api/projects", { method: "POST", body: form });
      closeModal();
      await loadProjects();
      await openProject(res.project.id);
      toast("项目已创建");
    } catch (e) {
      toast("创建失败：" + e.message, true);
      btn.disabled = false;
      btn.textContent = "上传并创建";
    }
  });
}
window.closeModal = closeModal;

async function renameProject() {
  const name = prompt("新的项目名称", state.project.name);
  if (!name) return;
  await api(`/api/projects/${state.project.id}`, { method: "PATCH", json: { name } });
  await openProject(state.project.id, true);
  await loadProjects();
}

async function deleteProject() {
  if (!confirm(`确定删除项目「${state.project.name}」？\n音频、逐字稿、纪要都会被永久删除。`)) return;
  await api(`/api/projects/${state.project.id}`, { method: "DELETE" });
  state.project = null;
  state.transcript = null;
  state.summary = null;
  $("#main").innerHTML = `<div class="empty"><h3>项目已删除</h3></div>`;
  await loadProjects();
}

async function resetTranscript() {
  if (!confirm("放弃人工校对版，回到机器初稿？此操作不可撤销（导出过的文件不受影响）。")) return;
  await api(`/api/projects/${state.project.id}/transcript/edited`, { method: "DELETE" });
  await openProject(state.project.id, true);
  toast("已回到机器初稿");
}

/* ----------------------------------------------------------------- settings */
async function openSettings() {
  const s = await api("/api/settings");
  modal(`
    <h2>设置</h2>
    <div class="sub">全部为本机配置，不会上传。</div>
    <div class="grid-2">
      <div class="field" style="grid-column:1/-1">
        <label>Hugging Face Token（用于首次下载 pyannote 说话人分离模型）</label>
        <input type="password" id="st-token" placeholder="${s.hf_token_set ? "已保存（留空则不修改）" : "hf_..."}" />
        <span class="desc">在 huggingface.co 上同意 pyannote/speaker-diarization-community-1 使用条款后再填入。Token 只保存到 secrets/hf_token.txt，不会写入日志。</span>
        <div style="display:flex;gap:8px;align-items:flex-start;margin-top:8px">
          <button class="btn btn-sm" id="st-verify" type="button">验证 Token 与模型授权</button>
          <span id="st-verify-result" class="muted" style="font-size:12px;white-space:pre-wrap;line-height:1.6"></span>
        </div>
      </div>
      <div class="field">
        <label>Ollama 地址</label>
        <input type="text" id="st-ollama" value="${esc(s.ollama_url)}" />
      </div>
      <div class="field">
        <label>默认纪要模型</label>
        <input type="text" id="st-model" value="${esc(s.ollama_model)}" />
      </div>
      <div class="field">
        <label>默认 Whisper 模型</label>
        <select id="st-asr">
          ${["medium", "large-v3"].map((m) => `<option value="${m}" ${s.default_asr_model === m ? "selected" : ""}>${m}</option>`).join("")}
        </select>
      </div>
      <div class="field">
        <label>默认线程数</label>
        <input type="number" id="st-threads" min="1" max="${(state.system?.env?.cpu_count || 16)}" value="${s.default_threads}" />
      </div>
      <div class="field">
        <label>长文分段大小（字符）</label>
        <input type="number" id="st-chunk" min="1000" max="12000" value="${s.summary_chunk_chars}" />
      </div>
      <div class="field">
        <label>Ollama 上下文（num_ctx）</label>
        <input type="number" id="st-ctx" min="4096" max="65536" step="1024" value="${s.ollama_num_ctx}" />
      </div>
    </div>
    <div class="mt muted" style="font-size:12px;line-height:1.7">
      运行环境：${esc(state.system?.env?.os || "")} · Python ${esc(state.system?.env?.python || "")} · 逻辑核心 ${state.system?.env?.cpu_count || "?"}<br/>
      WhisperX ${esc(state.system?.packages?.whisperx || "未安装")} · pyannote ${esc(state.system?.packages?.["pyannote.audio"] || "未安装")} · torch ${esc(state.system?.packages?.torch || "")}
    </div>
    <div class="actions">
      <button class="btn" onclick="closeModal()">取消</button>
      <button class="btn btn-primary" id="st-save">保存</button>
    </div>`);
  $("#st-save").addEventListener("click", async () => {
    try {
      const token = $("#st-token").value.trim();
      if (token) await api("/api/settings/hf-token", { method: "POST", json: { token } });
      await api("/api/settings", {
        method: "POST",
        json: {
          ollama_url: $("#st-ollama").value.trim(),
          ollama_model: $("#st-model").value.trim(),
          default_asr_model: $("#st-asr").value,
          default_threads: Number($("#st-threads").value),
          summary_chunk_chars: Number($("#st-chunk").value),
          ollama_num_ctx: Number($("#st-ctx").value),
        },
      });
      state.settings = await api("/api/settings");
      await loadSystem();
      if (token) {
        const v = await api("/api/hf/verify").catch(() => null);
        if (v) toast(v.ok ? "设置已保存 · Token 校验通过" : "设置已保存，但 Token 校验未通过：" + (v.reason || ""), !v.ok);
      } else {
        toast("设置已保存");
      }
      closeModal();
      if (state.project) renderMain();
    } catch (e) {
      toast("保存失败：" + e.message, true);
    }
  });

  $("#st-verify").addEventListener("click", async () => {
    const out = $("#st-verify-result");
    const token = $("#st-token").value.trim();
    out.style.color = "var(--muted)";
    out.textContent = "检查中…（需要访问 huggingface.co）";
    try {
      if (token) await api("/api/settings/hf-token", { method: "POST", json: { token } });
      const res = await api("/api/hf/verify");
      out.style.color = res.ok ? "var(--ok)" : "var(--danger)";
      out.textContent = res.message || (res.ok ? "校验通过" : "校验失败");
      if (res.ok) $("#st-token").value = "";
      await loadSystem();
    } catch (e) {
      out.style.color = "var(--danger)";
      out.textContent = "校验失败：" + e.message;
    }
  });
}

/* -------------------------------------------------------------------- init */
async function init() {
  document.getElementById("btn-settings").addEventListener("click", openSettings);
  document.getElementById("btn-new-project").addEventListener("click", openNewProject);
  document.getElementById("pill-ollama").addEventListener("click", async () => {
    await loadSystem();
    try {
      const res = await api("/api/ollama/models");
      state.ollamaModels = res.models || [];
      toast(res.online ? `Ollama 在线，${res.models.length} 个模型可用` : "Ollama 未连接");
    } catch (e) {
      state.ollamaModels = [];
      toast("Ollama 未连接", true);
    }
    if (state.project) renderMain();
  });
  document.getElementById("pill-token").addEventListener("click", openSettings);

  await loadSystem();
  state.settings = await api("/api/settings").catch(() => null);
  await loadProjects();
  try {
    const res = await api("/api/ollama/models");
    state.ollamaModels = res.models || [];
  } catch { state.ollamaModels = []; }
  if (state.projects.length) openProject(state.projects[0].id);

  setInterval(loadProjects, 15000);
  window.addEventListener("beforeunload", (e) => {
    if (state.dirty) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
}

init();
