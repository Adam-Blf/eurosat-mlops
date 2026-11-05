import createUpload from "./vendor/reicon/icons/Upload.js";
import createAlert from "./vendor/reicon/icons/AlertTriangle.js";
import createCheck from "./vendor/reicon/icons/CheckCircle.js";

const LABELS = {
  AnnualCrop: "Culture annuelle",
  Forest: "Forêt",
  HerbaceousVegetation: "Végétation herbacée",
  Highway: "Autoroute",
  Industrial: "Zone industrielle",
  Pasture: "Pâturage",
  PermanentCrop: "Culture permanente",
  Residential: "Zone résidentielle",
  River: "Rivière",
  SeaLake: "Mer ou lac",
};
const ERRORS = {
  401: "Collez votre jeton d'accès pour interroger le modèle.",
  403: "Ce jeton n'est pas reconnu.",
  413: "Image trop lourde, la limite est de 2 Mo.",
  415: "Ce fichier n'est pas une image lisible.",
  429: "Trop de requêtes, patientez quelques secondes.",
};
const API = "api";

const $ = (id) => document.getElementById(id);
const els = {
  drop: $("drop"), file: $("file"), preview: $("preview"), text: $("drop-text"),
  go: $("go"), status: $("status"), token: $("token"), empty: $("empty"),
  answer: $("answer"), label: $("label"), confidence: $("confidence"),
  bars: $("bars"), meta: $("meta"),
};
let current = null;

$("drop-icon").append(createUpload({ size: 42 }));
els.token.value = sessionStorage.getItem("eurosat-token") || "";
els.token.addEventListener("input", () => sessionStorage.setItem("eurosat-token", els.token.value));
$("token-form").addEventListener("submit", (e) => e.preventDefault());

function say(message, kind = "") {
  els.status.className = `hint ${kind}`;
  els.status.replaceChildren();
  if (kind) els.status.append((kind === "error" ? createAlert : createCheck)({ size: 18 }));
  els.status.append(message);
}

function pick(file) {
  if (!file) return;
  current = file;
  els.preview.src = URL.createObjectURL(file);
  els.preview.hidden = false;
  els.text.textContent = file.name;
  els.go.disabled = false;
  say("");
}

els.file.addEventListener("change", () => pick(els.file.files[0]));
els.drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.file.click(); }
});
for (const type of ["dragenter", "dragover"]) {
  els.drop.addEventListener(type, (e) => { e.preventDefault(); els.drop.classList.add("over"); });
}
for (const type of ["dragleave", "drop"]) {
  els.drop.addEventListener(type, (e) => { e.preventDefault(); els.drop.classList.remove("over"); });
}
els.drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

function barRow(name, p, isTop) {
  const li = document.createElement("li");
  if (isTop) li.className = "top";
  const title = document.createElement("span");
  title.textContent = LABELS[name] ?? name;
  const track = document.createElement("span");
  track.className = "track";
  const fill = document.createElement("span");
  fill.className = "fill";
  track.append(fill);
  const pct = document.createElement("span");
  pct.className = "pct";
  pct.textContent = (p * 100).toFixed(1);
  li.append(title, track, pct);
  requestAnimationFrame(() => { fill.style.width = `${Math.max(p * 100, 0.5)}%`; });
  return li;
}

function show(result) {
  els.empty.hidden = true;
  els.answer.hidden = false;
  els.label.textContent = LABELS[result.label] ?? result.label;
  els.confidence.textContent = `${(result.confidence * 100).toFixed(1)} %`;
  const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  els.bars.replaceChildren(...ranked.map(([name, p], i) => barRow(name, p, i === 0)));
  els.meta.textContent = `Modèle ${result.model_version} - ${result.latency_ms} ms`;
}

els.go.addEventListener("click", async () => {
  if (!current) return;
  els.go.disabled = true;
  say("Analyse en cours");
  const body = new FormData();
  body.append("file", current);
  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { Authorization: `Bearer ${els.token.value.trim()}` },
      body,
    });
    if (!res.ok) {
      say(ERRORS[res.status] ?? "Le service a répondu par une erreur, réessayez.", "error");
      return;
    }
    show(await res.json());
    say("Analyse terminée", "ok");
  } catch {
    say("Le service est injoignable.", "error");
  } finally {
    els.go.disabled = false;
  }
});
