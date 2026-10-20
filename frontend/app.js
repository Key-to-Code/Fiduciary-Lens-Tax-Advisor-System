const API_BASE = (window.FIDUCIARY_API_BASE || "http://localhost:8000/api/v1").replace(/\/$/, "");

const splashScreen = document.querySelector("#splashScreen");
const splashStartedAt = performance.now();
const transcript = document.querySelector("#transcript");
const conversation = document.querySelector("#conversation");
const input = document.querySelector("#questionInput");
const composer = document.querySelector("#composer");
const sendButton = document.querySelector("#sendButton");
const quickSection = document.querySelector("#quickSection");
const welcome = document.querySelector("#welcome");
const sourcesPanel = document.querySelector("#sourcesPanel");
const appShell = document.querySelector(".app-shell");
const chatColumn = document.querySelector(".chat-column");
const sourcesContent = document.querySelector("#sourcesContent");
const sourceCount = document.querySelector("#sourceCount");
const sourcesToggle = document.querySelector("#sourcesToggle");
const mobileScrim = document.querySelector("#mobileScrim");
const drawerClose = document.querySelector("#drawerClose");
const announcer = document.querySelector("#announcer");
const documentsNav = document.querySelector("#documentsNav");
const documentWorkspace = document.querySelector("#documentsView");
const authPanel = document.querySelector("#documentAuthPanel");
const uploadPanel = document.querySelector("#documentUploadPanel");
const authForm = document.querySelector("#documentAuthForm");
const authModeToggle = document.querySelector("#authModeToggle");
const authSubmit = document.querySelector("#authSubmit");
const authMessage = document.querySelector("#authMessage");
const uploadForm = document.querySelector("#documentUploadForm");
const documentFile = document.querySelector("#documentFile");
const documentDropZone = document.querySelector("#documentDropZone");
const analyzeButton = document.querySelector("#analyzeButton");
const uploadMessage = document.querySelector("#uploadMessage");
const analysisResults = document.querySelector("#analysisResults");
const membersNav = document.querySelector("#membersNav");
const evaluationNav = document.querySelector("#evaluationNav");
const membersView = document.querySelector("#membersView");
const evaluationView = document.querySelector("#evaluationView");

let accessToken = null;
let creatingAccount = false;
let authPending = false;
let analysisPending = false;

let pending = false;
let turns = [];
let latestSources = [];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function checkKnowledgeBase() {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 4500);
  try {
    const response = await fetch(`${API_BASE}/knowledge`, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Knowledge status returned ${response.status}`);
    await response.json();
  } catch {
    // The splash still closes when the knowledge API is unavailable.
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function dismissSplash() {
  const minimumVisibleMs = 760;
  const remaining = Math.max(0, minimumVisibleMs - (performance.now() - splashStartedAt));
  window.setTimeout(() => {
    splashScreen.classList.add("is-leaving");
    window.setTimeout(() => {
      splashScreen.hidden = true;
      document.body.setAttribute("aria-busy", "false");
    }, 440);
  }, remaining);
}

function scrollToLatest() {
  requestAnimationFrame(() => {
    transcript.scrollTo({ top: transcript.scrollHeight, behavior: "smooth" });
  });
}

function setBusy(value) {
  pending = value;
  input.disabled = value;
  sendButton.disabled = value || !input.value.trim();
  document.querySelectorAll(".quick-action").forEach((button) => { button.disabled = value; });
  sendButton.setAttribute("aria-label", value ? "Sending question" : "Send question");
}

function createTurn(role, message, { grounded = true, sources = [], error = false, latency = null, provider = null, onRetry = null } = {}) {
  const turn = element("article", `turn ${role}${grounded ? "" : " ungrounded"}${error ? " error" : ""}`);
  turn.dataset.role = role;
  const bubble = element("div", "bubble");
  bubble.setAttribute("aria-label", role === "user" ? "Your message" : "Fiduciary Lens response");

  if (message instanceof Node) {
    bubble.append(message);
  } else {
    String(message).split(/\n\s*\n/).filter(Boolean).forEach((paragraph) => {
      bubble.append(element("p", "", paragraph.trim()));
    });
  }

  if (role === "assistant" && !error) {
    if (!grounded) {
      const note = element("div", "answer-note");
      note.append(element("span", "note-mark", "§"));
      note.append(element("span", "", "The indexed sources did not provide enough support for this answer. No sources are attached."));
      bubble.append(note);
    } else if (sources.length) {
      const chips = element("div", "answer-citations");
      sources.forEach((source, index) => {
        const chip = makeCitationChip(source, index);
        chip.addEventListener("click", () => focusSource(index));
        chips.append(chip);
      });
      bubble.append(chips);
    }
  }

  turn.append(bubble);
  if (role === "assistant") {
    const providerNames = { openai: "OpenAI", openrouter: "OpenRouter", ollama: "Ollama", local: "Local model", extractive: "Extractive" };
    const providerLabel = provider ? ` · ${providerNames[provider] || provider}` : "";
    const metaText = error ? "Connection issue" : grounded
      ? `Grounded · ${sources.length} ${sources.length === 1 ? "source" : "sources"}${providerLabel}${latency !== null ? ` · ${latency} ms` : ""}`
      : "Not sufficiently grounded";
    turn.append(element("div", "turn-meta", metaText));
    if (onRetry) {
      const retry = element("button", "retry-button", "Try again");
      retry.type = "button";
      retry.addEventListener("click", onRetry);
      turn.querySelector(".bubble").append(retry);
    }
  } else {
    turn.append(element("div", "turn-meta", "You"));
  }

  conversation.append(turn);
  scrollToLatest();
  return turn;
}

function sectionLabel(source) {
  const sourceText = `${source.short || ""} ${source.citation || ""}`;
  const section = sourceText.match(/\b(?:section|s\.)\s*([\dA-Z]+(?:\([\w]+\))?)/i);
  if (section) return { symbol: "§", label: section[1] };
  const rule = sourceText.match(/\b(?:rule|r\.)\s*([\dA-Z]+(?:\([\w]+\))?)/i);
  if (rule) return { symbol: "Rule", label: rule[1] };
  return { symbol: "", label: source.short || source.citation || `Source ${source.n ?? ""}` };
}

function makeCitationChip(source, sourceIndex, { sourceRail = false } = {}) {
  const chip = element(sourceRail ? "span" : "button", sourceRail ? "source-short" : "citation-chip");
  const label = sectionLabel(source);
  if (label.symbol) {
    if (!sourceRail) chip.append(element("span", "section-symbol", label.symbol));
    chip.append(document.createTextNode(label.label));
  } else {
    chip.textContent = label.label;
  }
  chip.title = source.citation || source.short || "Retrieved source";
  chip.dataset.sourceIndex = String(sourceIndex);
  if (!sourceRail) {
    chip.type = "button";
    chip.setAttribute("aria-label", `Locate source ${source.n ?? sourceIndex + 1}: ${source.citation || label.label}`);
  }
  return chip;
}

function renderEmptySources(message = "Sources will appear here with your answer.") {
  sourceCount.textContent = "";
  sourcesContent.replaceChildren();
  const empty = element("div", "sources-empty");
  empty.append(element("span", "empty-mark", "§"));
  empty.append(element("p", "", message));
  empty.append(element("span", "", "Only retrieved passages are shown here."));
  sourcesContent.append(empty);
}

function safeWebUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function renderSources(sources) {
  latestSources = Array.isArray(sources) ? sources : [];
  sourceCount.textContent = latestSources.length ? `(${latestSources.length})` : "";
  if (!latestSources.length) {
    renderEmptySources("No retrieved sources for this response.");
    return;
  }

  sourcesContent.replaceChildren();
  latestSources.forEach((source, index) => {
    const card = element("article", "source-card");
    card.id = `source-${index}`;
    card.dataset.sourceIndex = String(index);

    const top = element("div", "source-topline");
    top.append(element("span", "source-number", String(source.n ?? index + 1)));
    top.append(makeCitationChip(source, index, { sourceRail: true }));
    card.append(top);
    card.append(element("h3", "source-title", source.citation || source.document || "Retrieved passage"));
    if (source.document) card.append(element("p", "source-meta", source.document));
    if (source.as_of) card.append(element("p", "source-meta", source.as_of));
    if (source.excerpt) card.append(element("p", "source-excerpt", source.excerpt));
    if (source.attribution) {
      const attribution = source.attribution;
      const explain = element("details", "source-attribution");
      const summary = element("summary", "", "Why this source ranked");
      explain.append(summary);
      const formula = element("p", "attribution-formula", `${attribution.algorithm || "Hybrid retrieval attribution"} · total score ${Number(source.score ?? 0).toFixed(3)}`);
      explain.append(formula);
      const contributions = element("div", "attribution-contributions");
      [
        [`Dense embedding · ${Math.round(Number(attribution.dense_weight ?? 0) * 100)}%`, attribution.dense_contribution],
        [`BM25 terms · ${Math.round(Number(attribution.bm25_weight ?? 0) * 100)}%`, attribution.bm25_contribution],
      ].forEach(([label, value]) => {
        const row = element("div", "attribution-row");
        row.append(element("span", "", label));
        row.append(element("strong", "", Number(value ?? 0).toFixed(3)));
        contributions.append(row);
      });
      if (Number(attribution.statute_weight ?? 1) < 1) {
        contributions.append(element("p", "attribution-formula", `Statute weight applied: ${Number(attribution.statute_weight).toFixed(2)}`));
      }
      explain.append(contributions);
      card.append(explain);
    }

    const url = safeWebUrl(source.url);
    if (url) {
      const link = element("a", "source-url", "Open source ↗");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      card.append(link);
    }
    sourcesContent.append(card);
  });
}

function focusSource(index) {
  openSources();
  requestAnimationFrame(() => {
    const card = document.querySelector(`#source-${index}`);
    if (!card) return;
    sourcesContent.querySelectorAll(".source-card").forEach((node) => node.classList.remove("is-active"));
    card.classList.add("is-active");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    conversation.querySelectorAll(".citation-chip").forEach((chip) => {
      chip.classList.toggle("is-active", Number(chip.dataset.sourceIndex) === index);
    });
  });
}

function showThinking() {
  const content = element("span", "thinking", "Looking through the indexed provisions");
  const dots = element("span", "thinking-dots");
  dots.setAttribute("aria-hidden", "true");
  for (let index = 0; index < 3; index += 1) dots.append(element("i"));
  content.append(dots);
  const turn = createTurn("assistant", content);
  turn.classList.add("thinking-turn");
  turn.querySelector(".turn-meta")?.remove();
  turn.setAttribute("aria-label", "Finding relevant tax-law sources");
  announcer.textContent = "Searching the indexed tax law. Please wait.";
  return turn;
}

function errorMessage(data, status) {
  const detail = data?.error || data?.detail?.error || data?.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") return detail.message || detail.details || `The service returned an error (${status}).`;
  return `The tax-law service could not be reached (HTTP ${status}). Please try again.`;
}

function showFormMessage(target, message, kind = "") {
  target.textContent = message;
  target.classList.toggle("is-error", kind === "error");
  target.classList.toggle("is-success", kind === "success");
}

function setDocumentAccount(email) {
  authPanel.hidden = Boolean(accessToken);
  uploadPanel.hidden = !accessToken;
  if (accessToken) {
    document.querySelector("#uploadHeading").textContent = `Welcome, ${email}`;
    document.querySelector("#accountHeading").textContent = "Signed in";
  } else {
    document.querySelector("#accountHeading").textContent = "Sign in to analyze a document";
    document.querySelector("#uploadHeading").textContent = "Choose a document to analyze";
  }
}

async function postJson(path, body) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Could not reach the backend. Start the API server and try again.");
  }
  let data = {};
  try { data = await response.json(); } catch { /* handled below */ }
  if (!response.ok) throw new Error(errorMessage(data, response.status));
  return data;
}

async function handleAuth(event) {
  event.preventDefault();
  if (authPending) return;
  const email = document.querySelector("#documentEmail").value.trim();
  const password = document.querySelector("#documentPassword").value;
  authPending = true;
  authSubmit.disabled = true;
  authSubmit.textContent = creatingAccount ? "Creating account…" : "Signing in…";
  showFormMessage(authMessage, "Connecting securely…");
  try {
    if (creatingAccount) await postJson("/auth/register", { email, password });
    const token = await postJson("/auth/login", { email, password });
    if (typeof token.access_token !== "string") throw new Error("The sign-in response did not include an access token.");
    accessToken = token.access_token;
    setDocumentAccount(email);
    showFormMessage(uploadMessage, "Choose a file to begin.");
    document.querySelector("#documentPassword").value = "";
    announcer.textContent = "Signed in. You can now upload a document.";
  } catch (error) {
    showFormMessage(authMessage, error.message, "error");
  } finally {
    authPending = false;
    authSubmit.disabled = false;
    authSubmit.textContent = creatingAccount ? "Create account" : "Sign in";
  }
}

function formatEntityValue(value) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.length ? value.map(formatEntityValue).join(", ") : "None identified";
  if (value && typeof value === "object") {
    const items = Object.entries(value).filter(([key]) => key !== "error");
    return items.length ? items.map(([key, item]) => `${key}: ${formatEntityValue(item)}`).join(" · ") : "No additional details identified";
  }
  return String(value ?? "Not identified");
}

function renderEntities(entities) {
  const target = document.querySelector("#entityList");
  target.replaceChildren();
  const entries = entities && typeof entities === "object" ? Object.entries(entities) : [];
  const visible = entries.filter(([, value]) => !(value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0));
  if (!visible.length) {
    target.append(element("p", "result-empty", "The extraction pipeline did not identify financial details in this document."));
    return;
  }
  visible.forEach(([label, value]) => {
    const item = element("div", "entity-item");
    item.append(element("span", "entity-label", label.replaceAll("_", " ")));
    item.append(element("span", "entity-value", formatEntityValue(value)));
    target.append(item);
  });
}

function renderInsightSources(sources) {
  const list = element("details", "insight-sources");
  const summary = element("summary", "", `Retrieved sources (${sources.length})`);
  list.append(summary);
  sources.forEach((source) => {
    const item = element("p", "insight-source");
    const title = source.citation || source.document || source.short || "Retrieved passage";
    item.append(element("strong", "", title));
    if (source.excerpt) item.append(document.createTextNode(source.excerpt));
    if (source.attribution) {
      const attribution = source.attribution;
      const explain = element("details", "source-attribution");
      explain.append(element("summary", "", "Why this source ranked"));
      explain.append(element("p", "attribution-formula", `${attribution.algorithm || "Hybrid retrieval attribution"} · score ${Number(source.score ?? 0).toFixed(3)}`));
      const breakdown = element("div", "attribution-contributions");
      [
        [`Dense embedding · ${Math.round(Number(attribution.dense_weight ?? 0) * 100)}%`, attribution.dense_contribution],
        [`BM25 terms · ${Math.round(Number(attribution.bm25_weight ?? 0) * 100)}%`, attribution.bm25_contribution],
      ].forEach(([label, value]) => {
        const row = element("div", "attribution-row");
        row.append(element("span", "", label));
        row.append(element("strong", "", Number(value ?? 0).toFixed(3)));
        breakdown.append(row);
      });
      explain.append(breakdown);
      item.append(explain);
    }
    const url = safeWebUrl(source.url);
    if (url) {
      item.append(document.createTextNode(" "));
      const link = element("a", "", "Open source ↗");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      item.append(link);
    }
    list.append(item);
  });
  return list;
}

function renderInsights(data) {
  const target = document.querySelector("#insightList");
  target.replaceChildren();
  const insights = data.tax_insights && typeof data.tax_insights === "object" ? Object.entries(data.tax_insights) : [];
  if (!insights.length) {
    target.append(element("p", "result-empty", "No tax-law insights were returned for this document."));
    return;
  }
  insights.forEach(([key, answer]) => {
    const item = element("article", "insight-item");
    const title = element("h3", "insight-title");
    title.append(element("span", "", key === "general" ? "General tax-saving provisions" : `Section ${key}`));
    const grounded = Boolean(data.tax_insight_grounding?.[key]);
    title.append(element("span", `grounding-badge${grounded ? "" : " is-ungrounded"}`, grounded ? "Grounded" : "Insufficient grounding"));
    item.append(title);
    item.append(element("p", "insight-text", String(answer)));
    const sources = data.tax_insight_sources?.[key];
    if (grounded && Array.isArray(sources) && sources.length) item.append(renderInsightSources(sources));
    target.append(item);
  });
}

function renderDocumentAnalysis(data) {
  document.querySelector("#resultFilename").textContent = data.filename || "Analyzed document";
  document.querySelector("#analysisDuration").textContent = Number.isFinite(data.processing_time) ? `${data.processing_time.toFixed(1)} sec` : "";
  document.querySelector("#analysisSummary").textContent = data.summary || "The summarizer returned no summary.";
  renderEntities(data.entities);
  renderInsights(data);
  analysisResults.hidden = false;
  analysisResults.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyzeDocument(event) {
  event.preventDefault();
  if (analysisPending || !accessToken) return;
  const file = documentFile.files?.[0];
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) {
    showFormMessage(uploadMessage, "This file is larger than the current 20 MB upload limit.", "error");
    return;
  }
  analysisPending = true;
  analyzeButton.disabled = true;
  analyzeButton.classList.add("is-loading");
  analyzeButton.textContent = "Analyzing…";
  analysisResults.hidden = true;
  showFormMessage(uploadMessage, "Extracting text, summarizing, and retrieving related tax provisions. This may take a little while.");
  const payload = new FormData();
  payload.append("file", file);
  try {
    const response = await fetch(`${API_BASE}/documents/analyze`, {
      method: "POST",
      headers: { Authorization: `Bearer ${accessToken}`, Accept: "application/json" },
      body: payload,
    });
    let data = {};
    try { data = await response.json(); } catch { /* handled below */ }
    if (response.status === 401) {
      accessToken = null;
      setDocumentAccount("");
      throw new Error("Your session expired. Please sign in again.");
    }
    if (!response.ok) throw new Error(errorMessage(data, response.status));
    if (!data.summary || !data.document_id || !data.tax_insights) throw new Error("The analysis service returned incomplete results.");
    renderDocumentAnalysis(data);
    showFormMessage(uploadMessage, "Analysis complete. The document and summary were saved to your account.", "success");
    announcer.textContent = "Document analysis completed. Summary and tax-law insights are ready.";
  } catch (error) {
    showFormMessage(uploadMessage, error instanceof TypeError ? "The analysis service could not be reached. Check that the backend is running." : error.message, "error");
    announcer.textContent = error.message;
  } finally {
    analysisPending = false;
    analyzeButton.classList.remove("is-loading");
    analyzeButton.textContent = "Analyze document";
    analyzeButton.disabled = !documentFile.files?.length;
  }
}

function updateSelectedDocument(file) {
  document.querySelector("#selectedFileName").textContent = file?.name || "Select a tax document";
  document.querySelector("#selectedFileHint").textContent = file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · ${file.type || "Document"}` : "Choose a file or drop it here";
  analyzeButton.disabled = !file || analysisPending;
  analysisResults.hidden = true;
  if (file) showFormMessage(uploadMessage, "Ready to analyze. The original file is removed from temporary storage after extraction.");
}

function showDocuments() {
  closeSources({ returnFocus: false });
  appShell.classList.remove("project-page-open");
  appShell.classList.add("documents-open");
  setCurrentNavigation("documents");
  membersView.hidden = true;
  evaluationView.hidden = true;
  documentWorkspace.hidden = false;
  documentWorkspace.scrollTop = 0;
  (accessToken ? documentFile : document.querySelector("#documentEmail")).focus({ preventScroll: true });
}

function showChat() {
  appShell.classList.remove("documents-open");
  appShell.classList.remove("project-page-open");
  setCurrentNavigation("chat");
  documentWorkspace.hidden = true;
  membersView.hidden = true;
  evaluationView.hidden = true;
  input.focus({ preventScroll: true });
}

function setCurrentNavigation(current) {
  const navigation = { documents: documentsNav, members: membersNav, evaluation: evaluationNav };
  Object.entries(navigation).forEach(([name, button]) => {
    button.setAttribute("aria-current", name === current ? "page" : "false");
  });
}

function appendLink(parent, label, href, className = "project-link") {
  const link = element("a", className, label);
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  parent.append(link);
  return link;
}

function renderProjectContext(project) {
  const context = document.querySelector("#projectContext");
  context.replaceChildren();
  [["Group", project.group], ["Category", project.category], ["Course", `${project.courseCode} · ${project.courseName}`]].forEach(([label, value]) => {
    const item = element("div", "context-item");
    item.append(element("span", "context-label", label));
    item.append(element("strong", "context-value", value));
    context.append(item);
  });
}

function renderFaculty(faculty) {
  const card = document.querySelector("#facultyCard");
  card.replaceChildren();
  const content = element("div", "faculty-content");
  content.append(element("p", "faculty-role", faculty.role));
  content.append(element("h3", "faculty-name", faculty.name));
  content.append(element("p", "faculty-department", faculty.department));
  content.append(element("p", "faculty-school", faculty.school));
  const credentials = element("div", "faculty-credentials");
  credentials.append(element("span", "credential-label", "Qualification"));
  credentials.append(element("strong", "", faculty.qualification));
  content.append(credentials);
  const links = element("div", "faculty-links");
  const email = element("a", "project-link", faculty.email);
  email.href = `mailto:${faculty.email}`;
  links.append(email);
  content.append(links);
  card.append(content);
}

function renderMembers(members) {
  const grid = document.querySelector("#memberGrid");
  grid.replaceChildren();
  members.forEach((member, index) => {
    const card = element("article", "member-card");
    const top = element("div", "member-card-topline");
    top.append(element("span", "member-index", `0${index + 1}`));
    top.append(element("span", "member-caption", "Team member"));
    card.append(top);
    card.append(element("h3", "member-name", member.name));
    card.append(element("p", "member-roll", member.rollNo));
    const contacts = element("dl", "member-contacts");
    [["Email", member.email, `mailto:${member.email}`], ["Mobile", member.phone, `tel:${member.phone}`]].forEach(([label, value, href]) => {
      const row = element("div", "member-contact-row");
      row.append(element("dt", "", label));
      const dd = element("dd", "");
      const link = element("a", "member-contact-link", value);
      link.href = href;
      dd.append(link);
      row.append(dd);
      contacts.append(row);
    });
    const githubRow = element("div", "member-contact-row");
    githubRow.append(element("dt", "", "GitHub"));
    const githubValue = element("dd", "");
    appendLink(githubValue, member.github.replace("https://github.com/", "@"), member.github, "member-contact-link");
    githubRow.append(githubValue);
    contacts.append(githubRow);
    card.append(contacts);
    grid.append(card);
  });
}

function renderProjectLinks(project) {
  const footer = document.querySelector("#projectLinks");
  footer.replaceChildren();
  const repository = element("div", "project-link-group");
  repository.append(element("span", "context-label", "Project repository"));
  appendLink(repository, "View on GitHub ↗", project.repository);
  footer.append(repository);
  const demo = element("div", "project-link-group");
  demo.append(element("span", "context-label", "Product demo"));
  demo.append(element("span", "unavailable-link", project.productDemo || "Link not included in the supplied details"));
  footer.append(demo);
}

function renderEvaluation(evaluation) {
  const retrieval = evaluation.retrieval;
  const metrics = document.querySelector("#retrievalMetrics");
  metrics.replaceChildren();
  [
    [`Recall@${retrieval.k}`, `${Math.round(retrieval.recall * 100)}%`, `${retrieval.recallCount}/${retrieval.questions} target provisions in the top ${retrieval.k}`],
    ["Mean reciprocal rank", retrieval.mrr.toFixed(3), "Average rank of the first matching provision"],
    ["Off-topic refusal", `${Math.round(retrieval.refusalRate * 100)}%`, `${retrieval.refusedOffTopic}/${retrieval.offTopicQuestions} unrelated prompts refused`],
  ].forEach(([label, value, note]) => {
    const card = element("article", "metric-card");
    card.append(element("span", "metric-label", label));
    card.append(element("strong", "metric-value", value));
    card.append(element("span", "metric-note", note));
    metrics.append(card);
  });

  const details = document.querySelector("#retrievalDetails");
  details.replaceChildren();
  [
    ["Index", `${retrieval.indexChunks.toLocaleString()} passages`],
    ["Embedding model", retrieval.embeddingModel],
    ["Retrieval", retrieval.retrievalMethod],
    ["Evaluation set", `${retrieval.questions} in-scope prompts · ${retrieval.offTopicQuestions} off-topic controls`],
  ].forEach(([label, value]) => {
    const row = element("div", "evaluation-detail-row");
    row.append(element("span", "", label));
    row.append(element("strong", "", value));
    details.append(row);
  });
  details.append(element("p", "evaluation-historical-note", evaluation.historicalNote));

  const modelGrid = document.querySelector("#modelPerformanceGrid");
  modelGrid.replaceChildren();
  evaluation.models.forEach((model) => {
    const card = element("article", "model-performance-card");
    const header = element("div", "model-card-heading");
    header.append(element("span", "model-role", model.role));
    header.append(element("span", model.measured ? "measurement-tag is-measured" : "measurement-tag", model.measurementLabel || (model.measured ? "Measured" : "Not benchmarked")));
    card.append(header);
    card.append(element("h3", "", model.name));
    if (model.runtime) card.lastChild.dataset.runtimeModel = "true";
    card.append(element("p", "model-description", model.description));
    const measures = element("ul", "model-measures");
    model.measures.forEach((measure) => measures.append(element("li", "", measure)));
    card.append(measures);
    modelGrid.append(card);
  });
  const xai = document.querySelector("#xaiMethodCard");
  xai.replaceChildren();
  xai.append(element("strong", "", evaluation.xai.name));
  xai.append(element("p", "", evaluation.xai.description));
  xai.append(element("code", "", evaluation.xai.formula));

  const harness = evaluation.harness;
  const result = document.querySelector("#harnessResult");
  result.replaceChildren();
  result.append(element("span", "result-check", "✓"));
  result.append(element("strong", "", `${harness.passed} of ${harness.testCount} checks passed`));
  result.append(element("span", "result-date", `· ${evaluation.verifiedOn}`));
  const checks = document.querySelector("#evaluationChecks");
  checks.replaceChildren();
  harness.checks.forEach((check) => {
    const row = element("li", "evaluation-check");
    row.append(element("span", "check-mark", "✓"));
    row.append(element("span", "", check));
    checks.append(row);
  });
  document.querySelector("#evaluationIntro").textContent = `The current index was measured with a fixed tax-law retrieval set. Separate deterministic tests check the evaluation code; they do not measure the configured LLM. This page separates system-level results from per-model results that have not yet been benchmarked.`;
  document.querySelector("#evaluationDate").textContent = `Results verified ${evaluation.verifiedOn}`;
  document.querySelector("#evaluationCaveat").textContent = evaluation.caveat;
}

function initializeProjectPages() {
  const project = window.FIDUCIARY_PROJECT_DATA;
  if (!project) return;
  renderProjectContext(project.context);
  renderFaculty(project.faculty);
  renderMembers(project.members);
  renderProjectLinks(project.context);
  renderEvaluation(project.evaluation);
}

async function updateRuntimeModelLabel() {
  try {
    const response = await fetch(`${API_BASE}/knowledge`, { headers: { Accept: "application/json" } });
    if (!response.ok) return;
    const status = await response.json();
    const modelLabel = [status.generation_provider, status.generation_model].filter(Boolean).join(" · ");
    if (modelLabel) {
      document.querySelectorAll("[data-runtime-model='true']").forEach((node) => { node.textContent = modelLabel; });
    }
  } catch {
    // The information page remains useful if the API is offline.
  }
}

function showProjectPage(page) {
  closeSources({ returnFocus: false });
  appShell.classList.remove("documents-open");
  appShell.classList.add("project-page-open");
  documentWorkspace.hidden = true;
  membersView.hidden = page !== "members";
  evaluationView.hidden = page !== "evaluation";
  setCurrentNavigation(page);
  (page === "members" ? membersView : evaluationView).scrollTop = 0;
}

function activePage() {
  if (appShell.classList.contains("documents-open")) return "documents";
  if (!membersView.hidden) return "members";
  if (!evaluationView.hidden) return "evaluation";
  return "chat";
}

function navigateToPage(page) {
  const destination = activePage() === page ? "chat" : page;
  window.location.hash = destination === "chat" ? "#chat" : `#${destination}`;
}

function syncPageFromLocation() {
  const page = window.location.hash.slice(1);
  if (page === "documents") showDocuments();
  else if (page === "members" || page === "evaluation") showProjectPage(page);
  else showChat();
}

async function ask(question, { retry = false } = {}) {
  const cleaned = question.trim();
  if (!cleaned || pending) return;
  if (!retry) {
    document.body.classList.add("has-conversation");
    chatColumn.classList.remove("is-home");
    welcome.hidden = true;
    quickSection.hidden = true;
    createTurn("user", cleaned);
  }

  setBusy(true);
  const thinking = showThinking();
  const previousTurns = turns.slice(-6).map(({ question: q, answer }) => ({ question: q, answer }));
  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ question: cleaned, history: previousTurns }),
    });
    let data = {};
    try { data = await response.json(); } catch { /* handled as a service error below */ }
    if (!response.ok) throw new Error(errorMessage(data, response.status));
    if (typeof data.answer !== "string" || typeof data.grounded !== "boolean") {
      throw new Error("The service returned an incomplete answer. Please try again.");
    }

    thinking.remove();
    const sources = data.grounded && Array.isArray(data.sources) ? data.sources : [];
    const grounded = Boolean(data.grounded) && sources.length > 0;
    createTurn("assistant", data.answer, {
      grounded,
      sources: grounded ? sources : [],
      provider: typeof data.provider === "string" ? data.provider : null,
      latency: Number.isFinite(data.latency_ms) ? data.latency_ms : null,
    });
    renderSources(grounded ? sources : []);
    if (!retry) turns.push({ question: cleaned, answer: data.answer });
    announcer.textContent = grounded ? "A grounded answer and sources are ready." : "The retrieved provisions did not sufficiently support an answer.";
  } catch (error) {
    thinking.remove();
    const message = error instanceof TypeError
      ? "Could not connect to the tax-law service. Check that the backend is running, then try again."
      : error.message;
    createTurn("assistant", message, {
      grounded: false,
      error: true,
      onRetry: () => ask(cleaned, { retry: true }),
    });
    renderEmptySources("Sources are unavailable because the request did not complete.");
    announcer.textContent = message;
  } finally {
    setBusy(false);
    input.value = "";
    resizeInput();
    input.focus({ preventScroll: true });
    scrollToLatest();
  }
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 156)}px`;
  sendButton.disabled = pending || !input.value.trim();
}

function openSources() {
  if (!document.body.classList.contains("has-conversation")) return;
  sourcesPanel.hidden = false;
  sourcesPanel.classList.add("is-open");
  appShell.classList.add("sources-open");
  mobileScrim.hidden = false;
  sourcesToggle.setAttribute("aria-expanded", "true");
  sourcesToggle.setAttribute("aria-label", "Close sources panel");
  if (window.matchMedia("(max-width: 850px)").matches) {
    drawerClose.focus({ preventScroll: true });
  } else {
    sourcesPanel.focus({ preventScroll: true });
  }
}

function closeSources({ returnFocus = true } = {}) {
  sourcesPanel.classList.remove("is-open");
  sourcesPanel.hidden = true;
  appShell.classList.remove("sources-open");
  mobileScrim.hidden = true;
  sourcesToggle.setAttribute("aria-expanded", "false");
  sourcesToggle.setAttribute("aria-label", "Open sources panel");
  if (returnFocus && document.body.classList.contains("has-conversation")) {
    sourcesToggle.focus({ preventScroll: true });
  }
}

documentsNav.addEventListener("click", () => navigateToPage("documents"));
membersNav.addEventListener("click", () => navigateToPage("members"));
evaluationNav.addEventListener("click", () => navigateToPage("evaluation"));
window.addEventListener("hashchange", syncPageFromLocation);
authForm.addEventListener("submit", handleAuth);
authModeToggle.addEventListener("click", () => {
  creatingAccount = !creatingAccount;
  authSubmit.textContent = creatingAccount ? "Create account" : "Sign in";
  authModeToggle.textContent = creatingAccount ? "I already have an account" : "Create account";
  document.querySelector("#documentPassword").autocomplete = creatingAccount ? "new-password" : "current-password";
  showFormMessage(authMessage, creatingAccount ? "Use at least 8 characters. Your password is stored as a secure hash." : "");
});
document.querySelector("#signOutButton").addEventListener("click", () => {
  accessToken = null;
  documentFile.value = "";
  updateSelectedDocument(null);
  analysisResults.hidden = true;
  setDocumentAccount("");
  showFormMessage(authMessage, "You have signed out.", "success");
});
documentFile.addEventListener("change", () => updateSelectedDocument(documentFile.files?.[0] || null));
uploadForm.addEventListener("submit", analyzeDocument);
for (const name of ["dragenter", "dragover"]) {
  documentDropZone.addEventListener(name, (event) => {
    event.preventDefault();
    documentDropZone.classList.add("is-dragging");
  });
}
for (const name of ["dragleave", "drop"]) {
  documentDropZone.addEventListener(name, (event) => {
    event.preventDefault();
    documentDropZone.classList.remove("is-dragging");
  });
}
documentDropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer?.files || [];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  documentFile.files = transfer.files;
  updateSelectedDocument(file);
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(input.value);
});
input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!pending && input.value.trim()) ask(input.value);
  }
});
document.querySelectorAll(".quick-action").forEach((button) => {
  button.addEventListener("click", () => ask(button.dataset.question || ""));
});
sourcesToggle.addEventListener("click", () => {
  if (sourcesPanel.hidden) openSources();
  else closeSources();
});
drawerClose.addEventListener("click", closeSources);
mobileScrim.addEventListener("click", closeSources);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sourcesPanel.classList.contains("is-open")) closeSources();
});

initializeProjectPages();
updateRuntimeModelLabel();
syncPageFromLocation();
checkKnowledgeBase().finally(dismissSplash);
