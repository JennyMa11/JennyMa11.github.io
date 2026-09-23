const root = document.documentElement;
const themeButton = document.getElementById("theme-toggle");
const menuButton = document.getElementById("menu-button");
const sidebar = document.getElementById("sidebar");
const scrim = document.getElementById("sidebar-scrim");
const progress = document.getElementById("progress");
const searchInput = document.getElementById("search-input");
const searchPanel = document.getElementById("search-results");
const searchBox = document.getElementById("searchbox");

function syncTheme() {
  const dark = root.dataset.theme === "dark";
  themeButton.textContent = dark ? "☀" : "◐";
  themeButton.setAttribute("aria-label", dark ? "切换浅色模式" : "切换深色模式");
}
syncTheme();
themeButton.addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("infra-theme", root.dataset.theme); } catch (_) { /* Private browsing. */ }
  syncTheme();
});

function setMenu(open) {
  sidebar.classList.toggle("open", open);
  scrim.classList.toggle("open", open);
  menuButton.setAttribute("aria-expanded", String(open));
  menuButton.setAttribute("aria-label", open ? "关闭目录" : "打开目录");
}
menuButton.addEventListener("click", () => setMenu(!sidebar.classList.contains("open")));
scrim.addEventListener("click", () => setMenu(false));
sidebar.querySelectorAll("a").forEach(link => link.addEventListener("click", () => setMenu(false)));

function updateProgress() {
  const page = document.documentElement;
  const max = page.scrollHeight - page.clientHeight;
  progress.style.width = `${max > 0 ? (page.scrollTop / max) * 100 : 0}%`;
}
window.addEventListener("scroll", updateProgress, { passive: true });
window.addEventListener("resize", updateProgress);
updateProgress();

document.querySelectorAll(".codehilite").forEach(box => {
  const code = box.querySelector("pre");
  if (!code) return;
  const button = document.createElement("button");
  button.className = "copy-button";
  button.type = "button";
  button.textContent = "复制";
  button.setAttribute("aria-label", "复制代码");
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.innerText);
      button.textContent = "已复制 ✓";
      window.setTimeout(() => { button.textContent = "复制"; }, 1600);
    } catch (_) {
      button.textContent = "复制失败";
    }
  });
  box.appendChild(button);
});

const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[char]));
let indexPromise;
function loadIndex() {
  if (!indexPromise) indexPromise = fetch("search-index.json").then(response => {
    if (!response.ok) throw new Error("Search index unavailable");
    return response.json();
  });
  return indexPromise;
}
function highlight(text, query) {
  const at = text.toLowerCase().indexOf(query.toLowerCase());
  if (at < 0) return escapeHtml(text.slice(0, 130));
  const start = Math.max(0, at - 35);
  const fragment = text.slice(start, start + 145);
  const offset = at - start;
  return escapeHtml(fragment.slice(0, offset)) + "<mark>" +
    escapeHtml(fragment.slice(offset, offset + query.length)) + "</mark>" +
    escapeHtml(fragment.slice(offset + query.length));
}
function showSearch(open) {
  searchPanel.classList.toggle("open", open);
  searchInput.setAttribute("aria-expanded", String(open));
}
let searchSequence = 0;
searchInput.addEventListener("input", async () => {
  const query = searchInput.value.trim();
  const sequence = ++searchSequence;
  if (!query) { showSearch(false); return; }
  searchPanel.innerHTML = '<div class="search-empty">正在搜索…</div>';
  showSearch(true);
  try {
    const index = await loadIndex();
    if (sequence !== searchSequence) return;
    const lower = query.toLowerCase();
    const words = lower.split(/\s+/).filter(Boolean);
    const results = index.map(item => {
      const title = item.title.toLowerCase();
      const heading = item.heading.toLowerCase();
      const text = item.text.toLowerCase();
      let score = 0;
      if (heading.includes(lower)) score += 15;
      if (title.includes(lower)) score += 10;
      const position = text.indexOf(lower);
      if (position >= 0) score += 5 + Math.max(0, 3 - position / 600);
      else if (words.length > 1 && words.every(word => text.includes(word))) score += 2;
      return { item, score };
    }).filter(result => result.score > 0).sort((a, b) => b.score - a.score).slice(0, 10);
    searchPanel.innerHTML = results.length ? results.map(({ item }) => {
      const location = `${item.url}${item.anchor ? `#${encodeURIComponent(item.anchor)}` : ""}`;
      return `<a class="search-item" role="option" href="${escapeHtml(location)}">
        <small>${escapeHtml(item.title)}</small><strong>${escapeHtml(item.heading)}</strong>
        <span>${highlight(item.text, query)}</span></a>`;
    }).join("") : `<div class="search-empty">没有找到“${escapeHtml(query)}”</div>`;
  } catch (_) {
    if (sequence === searchSequence) searchPanel.innerHTML = '<div class="search-empty">搜索暂不可用，请通过本地服务器预览。</div>';
  }
});
searchInput.addEventListener("focus", () => { if (searchInput.value.trim()) showSearch(true); });
document.addEventListener("click", event => { if (!searchBox.contains(event.target)) showSearch(false); });
document.addEventListener("keydown", event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault(); searchInput.focus(); searchInput.select();
  }
  if (event.key === "Escape") { showSearch(false); setMenu(false); searchInput.blur(); }
  if (event.key === "ArrowDown" && document.activeElement === searchInput && searchPanel.classList.contains("open")) {
    const first = searchPanel.querySelector("a");
    if (first) { event.preventDefault(); first.focus(); }
  }
});

const tocLinks = [...document.querySelectorAll(".page-toc a")];
if (tocLinks.length) {
  const linkById = new Map(tocLinks.map(link => [decodeURIComponent(link.hash.slice(1)), link]));
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      tocLinks.forEach(link => link.classList.remove("active"));
      linkById.get(entry.target.id)?.classList.add("active");
    }
  }, { rootMargin: "-100px 0px -65% 0px" });
  document.querySelectorAll(".prose h2[id]").forEach(heading => observer.observe(heading));
}
