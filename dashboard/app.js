const state = { data: null, window: "all", group: "all", query: "", disagreementsOnly: false };
const $ = (selector) => document.querySelector(selector);

function parseOffsetTime(match) {
  const timeMatch = match.time.match(/(\d{2}):(\d{2}) UTC([+-]\d+)/);
  if (!timeMatch) return new Date(`${match.date}T12:00:00Z`);
  const [, hour, minute, offset] = timeMatch;
  const sign = Number(offset) >= 0 ? "+" : "-";
  const absolute = String(Math.abs(Number(offset))).padStart(2, "0");
  return new Date(`${match.date}T${hour}:${minute}:00${sign}${absolute}:00`);
}

function oddsAgeHours(match) {
  if (!match.oddsCapturedAt) return null;
  return Math.max(0, (Date.now() - new Date(match.oddsCapturedAt).getTime()) / 3600000);
}

function freshness(match) {
  const age = oddsAgeHours(match);
  if (!match.hasOdds || age === null) return "none";
  return age <= state.data.oddsFreshHours ? "fresh" : "stale";
}

function formatDate(dateString) {
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric" }).format(new Date(`${dateString}T12:00:00Z`));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
}

function matchCard(match) {
  const status = freshness(match);
  const age = oddsAgeHours(match);
  const probabilities = match.marketProbabilities
    ? `<span class="probabilities">${match.marketProbabilities.map(value => Math.round(value)).join(" / ")}%</span>`
    : "<span class=\"odds-meta\">no odds</span>";
  const freshnessBadge = status === "fresh"
    ? "<span class=\"market-badge fresh\">fresh</span>"
    : status === "stale"
      ? "<span class=\"market-badge stale\">stale</span>"
      : "<span class=\"market-badge\">model-only</span>";
  const marketMeta = match.hasOdds
    ? `${match.bookmakerCount} book${match.bookmakerCount === 1 ? "" : "s"} · captured ${age < 1 ? "<1h" : `${Math.round(age)}h`} ago`
    : "market unavailable";
  return `
    <article class="match-card ${match.marketDisagrees ? "disagrees" : ""}">
      <div class="match-topline">
        <span class="match-time">${formatDate(match.date)} · ${escapeHtml(match.time)}</span>
        <span class="group-badge">Group ${escapeHtml(match.group)}</span>
      </div>
      <div class="match-body">
        <div class="teams">
          <div class="team-names"><span>${escapeHtml(match.teamA)}</span><span class="versus">vs</span><span>${escapeHtml(match.teamB)}</span></div>
          <div class="submit-score"><span>submit</span><strong>${escapeHtml(match.recommended)}</strong></div>
        </div>
        <div class="comparison-row">
          <span class="comparison-label">contender</span><span class="contender-score">${escapeHtml(match.contender)}</span>
          <span class="comparison-label">market</span>
          <div class="odds-line">${probabilities}${freshnessBadge}${match.marketDisagrees ? "<span class=\"market-badge disagree\">disagrees</span>" : ""}<span class="odds-meta">${marketMeta}</span></div>
        </div>
      </div>
    </article>`;
}

function filteredMatches() {
  const now = new Date();
  const horizon = new Date(now.getTime() + 3 * 86400000);
  return state.data.matches.filter(match => {
    const searchText = `${match.teamA} ${match.teamB}`.toLowerCase();
    if (state.query && !searchText.includes(state.query)) return false;
    if (state.group !== "all" && match.group !== state.group) return false;
    if (state.disagreementsOnly && !match.marketDisagrees) return false;
    if (state.window === "next3") {
      const kickoff = parseOffsetTime(match);
      return kickoff >= now && kickoff <= horizon;
    }
    return true;
  });
}

function render() {
  const matches = filteredMatches();
  $("#matchList").innerHTML = matches.map(matchCard).join("");
  $("#emptyState").hidden = matches.length > 0;
  $("#matchCountLabel").textContent = `${matches.length} match${matches.length === 1 ? "" : "es"}`;
  document.querySelectorAll(".segment").forEach(button => button.classList.toggle("active", button.dataset.window === state.window));
  $("#disagreementFilter").classList.toggle("active", state.disagreementsOnly);
}

function updateSummary() {
  const matches = state.data.matches;
  $("#strategyName").textContent = state.data.strategy.name;
  $("#strategyPoints").textContent = state.data.strategy.wcBacktestPoints.toFixed(2);
  $("#allCount").textContent = matches.length;
  $("#freshOddsCount").textContent = matches.filter(match => freshness(match) === "fresh").length;
  $("#staleOddsCount").textContent = matches.filter(match => freshness(match) !== "fresh").length;
  $("#disagreementCount").textContent = matches.filter(match => match.marketDisagrees).length;
  $("#generatedAt").textContent = `Built ${new Date(state.data.generatedAt).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`;
  const groups = [...new Set(matches.map(match => match.group))].sort();
  $("#groupFilter").innerHTML += groups.map(group => `<option value="${escapeHtml(group)}">Group ${escapeHtml(group)}</option>`).join("");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => toast.classList.remove("show"), 2600);
}

async function loadData() {
  const response = await fetch(`data.json?t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Dashboard data could not be loaded.");
  state.data = await response.json();
  updateSummary();
  render();
}

document.querySelectorAll(".segment").forEach(button => button.addEventListener("click", () => {
  state.window = button.dataset.window;
  render();
}));
$("#disagreementFilter").addEventListener("click", () => { state.disagreementsOnly = !state.disagreementsOnly; render(); });
$("#groupFilter").addEventListener("change", event => { state.group = event.target.value; render(); });
$("#teamSearch").addEventListener("input", event => { state.query = event.target.value.trim().toLowerCase(); render(); });
$("#rebuildButton").addEventListener("click", async () => {
  const button = $("#rebuildButton");
  button.disabled = true;
  try {
    const response = await fetch("/api/rebuild", { method: "POST" });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.message || "Rebuild failed.");
    location.reload();
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
});

loadData().catch(error => showToast(error.message));
