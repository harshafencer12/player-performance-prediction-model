const playerSelect = document.getElementById("playerSelect");
const quickStats = document.getElementById("quickStats");
const predictBtn = document.getElementById("predictBtn");
const errorMsg = document.getElementById("errorMsg");
const scoreboard = document.getElementById("scoreboard");
const ledger = document.getElementById("ledger");
const ledgerBody = document.getElementById("ledgerBody");
const scoreDigits = document.getElementById("scoreDigits");

playerSelect.addEventListener("change", async () => {
  const player = playerSelect.value;
  errorMsg.classList.add("hidden");
  scoreboard.classList.add("hidden");
  ledger.classList.add("hidden");

  if (!player) {
    quickStats.classList.add("hidden");
    predictBtn.disabled = true;
    return;
  }

  const res = await fetch("/player_stats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_name: player })
  });
  const data = await res.json();

  if (data.error) {
    errorMsg.textContent = data.error;
    errorMsg.classList.remove("hidden");
    quickStats.classList.add("hidden");
    predictBtn.disabled = true;
    return;
  }

  document.getElementById("qCareerAvg").textContent = data.career_avg;
  document.getElementById("qRecentForm").textContent = data.recent_form;
  document.getElementById("qMatches").textContent = data.matches_played;
  quickStats.classList.remove("hidden");
  predictBtn.disabled = false;

  ledgerBody.innerHTML = data.history.map(row =>
    `<tr><td>${row.date}</td><td>${row.runs}</td></tr>`
  ).join("");
});

predictBtn.addEventListener("click", async () => {
  const player = playerSelect.value;
  if (!player) return;

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ player_name: player })
  });
  const data = await res.json();

  if (data.error) {
    errorMsg.textContent = data.error;
    errorMsg.classList.remove("hidden");
    return;
  }

  scoreboard.classList.remove("hidden");
  ledger.classList.remove("hidden");
  animateCount(data.predicted_runs);
  scoreboard.scrollIntoView({ behavior: "smooth", block: "center" });
});

function animateCount(target) {
  const duration = 700;
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    scoreDigits.textContent = Math.round(target * eased);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}