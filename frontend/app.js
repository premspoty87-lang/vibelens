/* ============================================================
   VibeLens prototype frontend
   Vanilla JS, no build step. Talks to app.py when available and
   falls back to a local demo mode so the UI can be previewed.
   ============================================================ */

(() => {
  "use strict";

  const API = {
    status: "/api/status",
    overview: "/api/overview",
    predict: "/api/predict",
  };

  const SENTIMENTS = ["Positive", "Neutral", "Negative"];
  const HISTORY_KEY = "vibelens.history.v1";
  const HISTORY_LIMIT = 8;

  const SAMPLES = [
    {
      label: "Positive",
      text: "Absolutely love these earbuds. The bass is punchy, the battery easily lasts a full workday, and the fit stays comfortable for hours.",
    },
    {
      label: "Negative",
      text: "Stopped working after two weeks. The left bud lost sound completely and support kept redirecting me. Waste of money.",
    },
    {
      label: "Sarcastic",
      text: "Oh brilliant, the battery lasts a whole forty minutes. Truly revolutionary engineering, I love charging it twice a day.",
    },
    {
      label: "Neutral",
      text: "The product is okay for the price. Sound quality is average and the case scratches easily, but it does the job.",
    },
    {
      label: "Mixed",
      text: "Great sound and a comfortable fit, but the mic is useless in traffic and the connection drops constantly.",
    },
  ];

  const POSITIVE_WORDS = [
    "good", "great", "amazing", "love", "loved", "best", "nice", "excellent",
    "awesome", "perfect", "happy", "recommend", "worth", "quality", "comfortable",
    "fast", "beautiful", "superb", "brilliant", "fantastic", "solid", "impressive",
  ];

  const NEGATIVE_WORDS = [
    "bad", "worst", "waste", "poor", "hate", "terrible", "awful", "broken",
    "damaged", "disappointed", "refund", "return", "useless", "cheap", "slow",
    "problem", "issue", "defective", "useless", "pathetic", "annoying", "failed",
  ];

  const SARCASTIC_MARKERS = [
    "yeah right", "oh brilliant", "oh great", "thanks for nothing", "love how",
    "if you enjoy", "sure thing", "as if", "genius", "revolutionary", "shocking",
    "wonderful", "just what i needed", "how thoughtful", "what a surprise",
  ];

  const STOPWORDS = new Set([
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "it", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "my", "your",
    "its", "as", "at", "by", "from", "so", "not", "no", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "just", "very",
  ]);

  const DEMO_OVERVIEW = {
    ready: true,
    stats: {
      totalReviews: 9976,
      averageRating: 4.21,
      sentimentCounts: { Positive: 8091, Neutral: 884, Negative: 1001 },
      sentimentPercentages: { Positive: 81.1, Neutral: 8.9, Negative: 10.0 },
      topWords: {
        Positive: [
          ["good", 4310], ["nice", 2876], ["quality", 2431], ["product", 2210],
          ["sound", 1988], ["best", 1874], ["love", 1655], ["battery", 1421],
          ["amazing", 1310], ["value", 1188], ["great", 1094], ["comfortable", 986],
        ],
        Neutral: [
          ["okay", 642], ["product", 588], ["price", 512], ["average", 468],
          ["good", 401], ["decent", 366], ["quality", 322], ["works", 288],
          ["fine", 244], ["expected", 201], ["sound", 188], ["build", 164],
        ],
        Negative: [
          ["bad", 812], ["worst", 704], ["waste", 655], ["money", 588],
          ["quality", 544], ["product", 501], ["poor", 466], ["issue", 422],
          ["return", 388], ["damaged", 344], ["disappointed", 301], ["refund", 277],
        ],
      },
      sarcasmDataset: { total: 10000, sarcastic: 3618, notSarcastic: 6382 },
    },
    models: [
      { name: "Naive Bayes", kind: "Sentiment", accuracy: 0.8357, precision: 0.7849, recall: 0.8357, f1: 0.7716 },
      { name: "Logistic Regression", kind: "Sentiment", accuracy: 0.8662, precision: 0.832, recall: 0.8662, f1: 0.8363 },
      { name: "Sarcasm (Logistic Regression)", kind: "Sarcasm", accuracy: 0.8124, precision: 0.8088, recall: 0.8124, f1: 0.8061 },
    ],
    artifacts: [
      "sentiment_distribution.png",
      "positive_wordcloud.png",
      "negative_wordcloud.png",
      "confusion_matrix_nb.png",
      "confusion_matrix_lr.png",
    ],
  };

  const state = {
    demo: false,
    ready: false,
    overview: null,
    history: readHistory(),
    wordTab: "Positive",
    busy: false,
  };

  const els = {};

  const prefersReducedMotion = () =>
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function $(selector, root = document) {
    return root.querySelector(selector);
  }

  function $$(selector, root = document) {
    return Array.from(root.querySelectorAll(selector));
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]));
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function round1(value) {
    return Math.round(value * 10) / 10;
  }

  function formatNumber(value) {
    return Number(value).toLocaleString("en-US");
  }

  function formatPercent(fraction) {
    return `${(Number(fraction) * 100).toFixed(1)}%`;
  }

  function truncate(value, limit) {
    return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
  }

  function wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function timeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 45) return "just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  }

  /* ---------------- Boot ---------------- */

  function init() {
    cacheElements();
    bindEvents();
    renderSamples();
    renderHistory();
    setupReveal();
    updateCounter();

    if (els.analyzeBtn) {
      els.analyzeBtn.disabled = false;
    }

    pollStatus();
  }

  function cacheElements() {
    els.input = $("#reviewInput");
    els.charCount = $("#charCount");
    els.clearBtn = $("#clearBtn");
    els.analyzeBtn = $("#analyzeBtn");
    els.btnLabel = $(".btn__label", els.analyzeBtn);
    els.spinner = $(".spinner", els.analyzeBtn);
    els.errorBox = $("#errorBox");
    els.sampleChips = $("#sampleChips");
    els.emptyState = $("#emptyState");
    els.resultContent = $("#resultContent");
    els.resultPanel = $("#resultPanel");
    els.sentimentBadge = $("#sentimentBadge");
    els.sentimentConfidence = $("#sentimentConfidence");
    els.gauge = $("#gauge");
    els.reviewEcho = $("#reviewEcho");
    els.sarcasmTag = $("#sarcasmTag");
    els.sarcasmMeter = $("#sarcasmMeter");
    els.sarcasmMeterWrap = $("#sarcasmMeterWrap");
    els.sarcasmNote = $("#sarcasmNote");
    els.sentimentDist = $("#sentimentDist");
    els.cleanedText = $("#cleanedText");
    els.warnings = $("#warnings");
    els.statusPill = $("#statusPill");
    els.statusText = $(".status__text", els.statusPill);
    els.demoNotice = $("#demoNotice");
    els.historySection = $("#historySection");
    els.historyList = $("#historyList");
    els.clearHistory = $("#clearHistory");
    els.kpis = $("#kpis");
    els.distribution = $("#distribution");
    els.sarcasmDistribution = $("#sarcasmDistribution");
    els.wordTabs = $("#wordTabs");
    els.wordChips = $("#wordChips");
    els.modelBars = $("#modelBars");
    els.modelTableBody = $("#modelTable tbody");
    els.toast = $("#toast");
  }

  function bindEvents() {
    els.input.addEventListener("input", updateCounter);
    els.input.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        runAnalysis();
      }
    });

    els.clearBtn.addEventListener("click", () => {
      els.input.value = "";
      updateCounter();
      hideError();
      els.input.focus();
    });

    els.analyzeBtn.addEventListener("click", () => runAnalysis());

    els.sampleChips.addEventListener("click", (event) => {
      const button = event.target.closest("[data-sample]");
      if (!button) return;
      const sample = SAMPLES[Number(button.dataset.sample)];
      if (!sample) return;
      els.input.value = sample.text;
      updateCounter();
      runAnalysis(sample.text);
    });

    els.wordTabs.addEventListener("click", (event) => {
      const tab = event.target.closest("[data-word-tab]");
      if (!tab) return;
      state.wordTab = tab.dataset.wordTab;
      renderWordTabs();
      renderWordChips();
    });

    els.historyList.addEventListener("click", (event) => {
      const button = event.target.closest("[data-history]");
      if (!button) return;
      const entry = state.history[Number(button.dataset.history)];
      if (!entry) return;
      els.input.value = entry.review;
      updateCounter();
      runAnalysis(entry.review);
    });

    els.clearHistory.addEventListener("click", () => {
      state.history = [];
      saveHistory();
      renderHistory();
      showToast("History cleared.");
    });
  }

  /* ---------------- Status / overview ---------------- */

  async function pollStatus() {
    try {
      const response = await fetch(API.status, { cache: "no-store" });
      if (!response.ok) throw new Error("Status unavailable");
      const data = await response.json();
      applyStatus(data);

      if (data.ready) {
        await loadOverview();
        return;
      }

      if (data.state === "error") {
        enterDemoMode(data.error || "Model training failed.");
        return;
      }

      setTimeout(pollStatus, 1400);
    } catch (error) {
      enterDemoMode();
    }
  }

  function applyStatus(data) {
    if (data.state === "ready") {
      setStatus("ready", "Models ready");
    } else if (data.state === "training") {
      setStatus("loading", "Training models");
    } else if (data.state === "error") {
      setStatus("error", "Model error");
    } else {
      setStatus("loading", "Connecting");
    }
  }

  function setStatus(mode, label) {
    els.statusPill.dataset.state = mode;
    els.statusText.textContent = label;
  }

  async function loadOverview() {
    try {
      const response = await fetch(API.overview, { cache: "no-store" });
      if (!response.ok) throw new Error("Overview unavailable");
      state.overview = await response.json();
      state.ready = true;
      state.demo = false;
      els.demoNotice.hidden = true;
      setStatus("ready", "Models ready");
      renderOverview();
    } catch (error) {
      enterDemoMode();
    }
  }

  function enterDemoMode(reason) {
    if (state.demo) return;
    state.demo = true;
    state.ready = true;
    state.overview = DEMO_OVERVIEW;
    setStatus("demo", "Demo mode");
    els.demoNotice.hidden = false;
    renderOverview();
    if (reason) {
      showToast("Live models unavailable — showing demo data.");
    }
  }

  /* ---------------- Overview rendering ---------------- */

  function renderOverview() {
    if (!state.overview) return;
    renderKpis();
    renderDatasetDistribution();
    renderSarcasmDistribution();
    renderWordTabs();
    renderWordChips();
    renderModels();
    applyArtifacts();
  }

  function renderKpis() {
    const stats = state.overview.stats;
    const values = {
      totalReviews: formatNumber(stats.totalReviews),
      averageRating: Number(stats.averageRating).toFixed(2),
      positiveShare: `${Number(stats.sentimentPercentages.Positive).toFixed(1)}%`,
      sarcasticShare: `${((stats.sarcasmDataset.sarcastic / stats.sarcasmDataset.total) * 100).toFixed(1)}%`,
    };

    $$("[data-kpi]").forEach((node) => {
      const value = values[node.dataset.kpi];
      if (value !== undefined) node.textContent = value;
    });
  }

  function renderDatasetDistribution() {
    const stats = state.overview.stats;
    const items = SENTIMENTS.map((label) => ({
      key: label,
      label,
      percent: Number(stats.sentimentPercentages[label]),
      value: `${formatNumber(stats.sentimentCounts[label])} · ${Number(stats.sentimentPercentages[label]).toFixed(1)}%`,
    }));
    renderDistribution(els.distribution, items);
  }

  function renderSarcasmDistribution() {
    const data = state.overview.stats.sarcasmDataset;
    const total = data.total || 1;
    const items = [
      {
        key: "Negative",
        label: "Sarcastic",
        percent: (data.sarcastic / total) * 100,
        value: `${formatNumber(data.sarcastic)} · ${((data.sarcastic / total) * 100).toFixed(1)}%`,
      },
      {
        key: "Positive",
        label: "Not sarcastic",
        percent: (data.notSarcastic / total) * 100,
        value: `${formatNumber(data.notSarcastic)} · ${((data.notSarcastic / total) * 100).toFixed(1)}%`,
      },
    ];
    renderDistribution(els.sarcasmDistribution, items);
  }

  function renderDistribution(container, items) {
    container.innerHTML = items.map((item) => `
      <div class="dist__row">
        <span class="dist__label dist__label--${item.key.toLowerCase()}"><i></i>${escapeHtml(item.label)}</span>
        <span class="dist__track">
          <span class="dist__fill dist__fill--${item.key.toLowerCase()}" data-width="${clamp(item.percent, 0, 100)}"></span>
        </span>
        <span class="dist__value">${escapeHtml(item.value)}</span>
      </div>
    `).join("");

    requestAnimationFrame(() => {
      $$("[data-width]", container).forEach((node) => {
        node.style.width = `${node.dataset.width}%`;
      });
    });
  }

  function renderWordTabs() {
    els.wordTabs.innerHTML = SENTIMENTS.map((label) => `
      <button class="tab" type="button" role="tab" data-word-tab="${label}"
        aria-selected="${state.wordTab === label}">${label}</button>
    `).join("");
  }

  function renderWordChips() {
    const words = state.overview?.stats?.topWords?.[state.wordTab] || [];
    if (!words.length) {
      els.wordChips.innerHTML = '<p class="muted">No words available for this sentiment.</p>';
      return;
    }

    const counts = words.map((entry) => Number(entry[1]));
    const max = Math.max(...counts);
    const min = Math.min(...counts);
    const span = Math.max(1, max - min);
    const tone = state.wordTab.toLowerCase();

    els.wordChips.innerHTML = words.map((entry) => {
      const word = entry[0];
      const count = Number(entry[1]);
      const size = 0.78 + ((count - min) / span) * 0.72;
      return `<span class="wordchip wordchip--${tone}" style="font-size:${size.toFixed(2)}rem">
        <b>${escapeHtml(word)}</b><small>${formatNumber(count)}</small>
      </span>`;
    }).join("");
  }

  function renderModels() {
    const models = state.overview?.models || [];
    if (!models.length) return;

    els.modelBars.innerHTML = models.map((model) => `
      <div class="modelbar">
        <div class="modelbar__head">
          <span class="modelbar__name">${escapeHtml(model.name)}<span>${escapeHtml(model.kind)}</span></span>
          <span class="modelbar__value">${formatPercent(model.accuracy)} accuracy · ${formatPercent(model.f1)} F1</span>
        </div>
        <div class="track"><span class="fill fill--accuracy" data-width="${clamp(model.accuracy * 100, 0, 100)}"></span></div>
        <div class="track track--f1"><span class="fill fill--f1" data-width="${clamp(model.f1 * 100, 0, 100)}"></span></div>
      </div>
    `).join("");

    const sentimentModels = models.filter((model) => model.kind === "Sentiment");
    const best = sentimentModels.reduce(
      (winner, model) => (!winner || model.accuracy > winner.accuracy ? model : winner),
      null
    );

    els.modelTableBody.innerHTML = models.map((model) => `
      <tr class="${best && model.name === best.name ? "is-best" : ""}">
        <td>${escapeHtml(model.name)}</td>
        <td>${escapeHtml(model.kind)}</td>
        <td>${formatPercent(model.accuracy)}</td>
        <td>${formatPercent(model.precision)}</td>
        <td>${formatPercent(model.recall)}</td>
        <td>${formatPercent(model.f1)}</td>
      </tr>
    `).join("");

    requestAnimationFrame(() => {
      $$("[data-width]", els.modelBars).forEach((node) => {
        node.style.width = `${node.dataset.width}%`;
      });
    });
  }

  function applyArtifacts() {
    const base = location.protocol === "file:" ? "../output/" : "/output/";
    $$("[data-artifact]").forEach((image) => {
      const frame = image.closest("[data-frame]");
      if (!image.dataset.artifact) return;
      image.src = `${base}${image.dataset.artifact}`;
      image.addEventListener("load", () => frame.classList.remove("is-missing"), { once: true });
      image.addEventListener("error", () => frame.classList.add("is-missing"), { once: true });
    });
  }

  /* ---------------- Analysis ---------------- */

  async function runAnalysis(textOverride) {
    if (state.busy) return;

    const review = (textOverride ?? els.input.value).trim();
    if (review.length < 2) {
      showError("Please enter a review with at least 2 characters.");
      els.input.focus();
      return;
    }

    hideError();
    setBusy(true);

    try {
      let result;
      if (state.demo) {
        await wait(650);
        result = demoPredict(review);
      } else {
        const response = await fetch(API.predict, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ review }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.error || "Analysis failed. Please try again.");
        }
        result = data;
      }

      renderResult(result);
      pushHistory(result);

      if (window.innerWidth < 980) {
        els.resultPanel.scrollIntoView({
          behavior: prefersReducedMotion() ? "auto" : "smooth",
          block: "start",
        });
      }
    } catch (error) {
      showError(error.message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  function renderResult(result) {
    const sentiment = SENTIMENTS.includes(result.sentiment) ? result.sentiment : "Neutral";
    const confidence = Number(result.sentimentConfidence) || 0;
    const sarcastic = result.sarcasm === "Sarcastic";
    const likelihood = Number(
      result.sarcasmLikelihood ??
      result.sarcasmScores?.Sarcastic ??
      0
    );
    const sarcasmConfidence = Number(result.sarcasmConfidence) || 0;

    els.emptyState.hidden = true;
    els.resultContent.hidden = false;

    els.sentimentBadge.dataset.sentiment = sentiment;
    els.sentimentBadge.textContent = sentiment;
    els.sentimentConfidence.textContent =
      `${confidence.toFixed(1)}% confidence · ${formatNumber(result.review.length)} characters`;

    els.reviewEcho.textContent = result.review;
    renderGauge(confidence, sentiment);

    els.sarcasmTag.textContent = sarcastic ? "Sarcastic" : "Not sarcastic";
    els.sarcasmTag.className = `tag ${sarcastic ? "tag--sarcastic" : "tag--clean"}`;
    els.sarcasmMeter.className =
      `meter__fill ${sarcastic ? "meter__fill--sarcastic" : "meter__fill--clean"}`;
    requestAnimationFrame(() => {
      els.sarcasmMeter.style.width = `${clamp(likelihood, 0, 100)}%`;
    });
    els.sarcasmMeterWrap.setAttribute(
      "aria-label",
      `Sarcasm likelihood ${likelihood.toFixed(1)} percent`
    );
    els.sarcasmNote.textContent = sarcastic
      ? `Sarcasm likelihood ${likelihood.toFixed(1)}% · the model is ${sarcasmConfidence.toFixed(1)}% confident this review is sarcastic.`
      : `Sarcasm likelihood ${likelihood.toFixed(1)}% · the model is ${sarcasmConfidence.toFixed(1)}% confident this review is sincere.`;

    const scores = result.sentimentScores || {};
    const items = SENTIMENTS.map((label) => ({
      key: label,
      label,
      percent: Number(scores[label]) || 0,
      value: `${(Number(scores[label]) || 0).toFixed(1)}%`,
    }));
    renderDistribution(els.sentimentDist, items);

    els.cleanedText.textContent =
      result.cleaned || "Nothing left after removing stopwords and punctuation.";
    els.cleanedText.parentElement.open = false;

    els.warnings.innerHTML = (result.warnings || [])
      .map((warning) => `<li>${escapeHtml(warning)}</li>`)
      .join("");
    els.warnings.hidden = !(result.warnings || []).length;
  }

  function renderGauge(percent, sentiment) {
    const radius = 46;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference * (1 - clamp(percent, 0, 100) / 100);
    const tone = sentiment.toLowerCase();

    els.gauge.innerHTML = `
      <svg viewBox="0 0 120 120" role="img" aria-label="Sentiment confidence ${percent.toFixed(1)} percent">
        <circle class="gauge__track" cx="60" cy="60" r="${radius}"></circle>
        <circle class="gauge__value gauge__value--${tone}" cx="60" cy="60" r="${radius}"
          stroke-dasharray="${circumference.toFixed(2)}"
          stroke-dashoffset="${circumference.toFixed(2)}"></circle>
      </svg>
      <div class="gauge__center">
        <strong>${percent.toFixed(0)}%</strong>
        <span>confidence</span>
      </div>
    `;

    const circle = $(".gauge__value", els.gauge);
    requestAnimationFrame(() => {
      circle.style.strokeDashoffset = offset.toFixed(2);
    });
  }

  function demoPredict(review) {
    const text = review.toLowerCase();
    const tokens = text
      .replace(/[^a-z'\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);

    const positiveHits = tokens.filter((token) => POSITIVE_WORDS.includes(token)).length;
    const negativeHits = tokens.filter((token) => NEGATIVE_WORDS.includes(token)).length;
    const markers = SARCASTIC_MARKERS.filter((marker) => text.includes(marker)).length;

    const positive = 18 + positiveHits * 16 + markers * 4;
    const negative = 14 + negativeHits * 18 + markers * 14;
    const neutral = 16 + Math.max(0, 4 - positiveHits - negativeHits) * 9 +
      (tokens.length < 8 ? 12 : 0);

    const total = positive + negative + neutral;
    const scores = {
      Positive: round1((positive / total) * 100),
      Neutral: round1((neutral / total) * 100),
      Negative: round1((negative / total) * 100),
    };

    const sentiment = SENTIMENTS.reduce(
      (winner, label) => (scores[label] > scores[winner] ? label : winner),
      SENTIMENTS[0]
    );

    const sarcasmLikelihood = clamp(
      12 + markers * 24 + negativeHits * 6 +
        (positiveHits && negativeHits ? 12 : 0),
      3,
      96
    );
    const sarcastic = sarcasmLikelihood >= 50;
    const sarcasmScores = {
      Sarcastic: round1(sarcasmLikelihood),
      "Not Sarcastic": round1(100 - sarcasmLikelihood),
    };

    const warnings = [];
    if (sarcastic && sentiment === "Positive") {
      warnings.push(
        "The review reads positive on the surface but is flagged as sarcastic. Treat the positive score with caution."
      );
    } else if (sarcastic) {
      warnings.push(
        "Sarcastic tone detected. Surface-level sentiment may not reflect the reviewer's true opinion."
      );
    }
    if (scores[sentiment] < 55) {
      warnings.push(
        "Sentiment confidence is low, which usually means the review is mixed or ambiguous."
      );
    }

    const cleaned = tokens
      .filter((token) => !STOPWORDS.has(token))
      .join(" ");

    return {
      review,
      cleaned,
      sentiment,
      sentimentConfidence: scores[sentiment],
      sentimentScores: scores,
      sarcasm: sarcastic ? "Sarcastic" : "Not Sarcastic",
      sarcasmConfidence: sarcastic ? sarcasmLikelihood : 100 - sarcasmLikelihood,
      sarcasmScores,
      sarcasmLikelihood,
      warnings,
    };
  }

  /* ---------------- History ---------------- */

  function readHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history));
    } catch (error) {
      /* Storage may be unavailable; history stays in memory. */
    }
  }

  function pushHistory(result) {
    const entry = {
      review: result.review,
      sentiment: result.sentiment,
      sarcasm: result.sarcasm,
      confidence: Number(result.sentimentConfidence) || 0,
      at: Date.now(),
    };

    state.history = [
      entry,
      ...state.history.filter((item) => item.review !== entry.review),
    ].slice(0, HISTORY_LIMIT);

    saveHistory();
    renderHistory();
  }

  function renderHistory() {
    if (!state.history.length) {
      els.historySection.hidden = true;
      els.historyList.innerHTML = "";
      return;
    }

    els.historySection.hidden = false;
    els.historyList.innerHTML = state.history.map((entry, index) => `
      <button class="history__item" type="button" data-history="${index}">
        <span class="history__review">${escapeHtml(entry.review)}</span>
        <span class="history__meta">
          <span class="history__sentiment" data-sentiment="${escapeHtml(entry.sentiment)}">${escapeHtml(entry.sentiment)}</span>
          <span>${Number(entry.confidence).toFixed(1)}%</span>
          <span>${timeAgo(entry.at)}</span>
        </span>
      </button>
    `).join("");
  }

  /* ---------------- UI helpers ---------------- */

  function renderSamples() {
    els.sampleChips.innerHTML = SAMPLES.map((sample, index) => `
      <button class="chip" type="button" data-sample="${index}"
        title="${escapeHtml(sample.text)}">${escapeHtml(sample.label)} · ${escapeHtml(truncate(sample.text, 34))}</button>
    `).join("");
  }

  function updateCounter() {
    const length = els.input.value.length;
    els.charCount.textContent = `${formatNumber(length)} / 5000`;
  }

  function setBusy(busy) {
    state.busy = busy;
    els.analyzeBtn.disabled = busy;
    els.spinner.hidden = !busy;
    els.btnLabel.textContent = busy ? "Analyzing" : "Analyze review";
  }

  function showError(message) {
    els.errorBox.textContent = message;
    els.errorBox.hidden = false;
  }

  function hideError() {
    els.errorBox.hidden = true;
  }

  let toastTimer = null;

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.hidden = false;
    requestAnimationFrame(() => els.toast.classList.add("is-visible"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      els.toast.classList.remove("is-visible");
      setTimeout(() => {
        els.toast.hidden = true;
      }, 260);
    }, 2600);
  }

  function setupReveal() {
    const nodes = $$(".reveal");
    if (!("IntersectionObserver" in window) || prefersReducedMotion()) {
      nodes.forEach((node) => node.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });

    nodes.forEach((node) => observer.observe(node));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
