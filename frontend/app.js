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

checkKnowledgeBase().finally(dismissSplash);
