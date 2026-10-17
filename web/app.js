// Shared helpers for the VoxCPM web pages.

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    // FastAPI returns a string detail, or a list of validation errors
    const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
    const err = new Error(detail || res.statusText || "Request failed");
    err.status = res.status;
    throw err;
  }
  return data;
}

async function currentUser() {
  try { return await api("/api/me"); } catch { return null; }
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const fmt = (n) => Number(n).toLocaleString();
const fmtDate = (ts) => (ts ? new Date(ts * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "—");
const fmtDateTime = (ts) => (ts ? new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—");

function toast(message, isError = false) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.setAttribute("role", "status");
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.className = "show" + (isError ? " error" : "");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => (el.className = ""), 2600);
}

// Swap "Log in / Try for free" for "Dashboard" in the top nav when signed in.
async function initNav() {
  const actions = document.querySelector(".nav-actions");
  if (!actions) return null;
  const user = await currentUser();
  if (user) {
    actions.innerHTML = `<span class="badge">${fmt(user.credits)} credits</span><a class="btn btn-primary" href="/dashboard">Dashboard</a>`;
  }
  return user;
}

const LOGO_SVG = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect width="24" height="24" rx="6" fill="#fafafa"/><path d="M6 10v4M9 7v10M12 9v6M15 5v14M18 10v4" stroke="#0a0a0a" stroke-width="2" stroke-linecap="round"/></svg>`;
document.querySelectorAll("[data-logo]").forEach((el) => (el.innerHTML = LOGO_SVG));
