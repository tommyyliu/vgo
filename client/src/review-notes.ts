// Notes on a review page: one shared document when the page runs as an artifact with the
// `db` capability (for Claude to read), this browser's storage otherwise.

interface Db { doc(path: string): DocRef }
interface DocRef {
  set(data: Record<string, unknown>): Promise<void>;
  onSnapshot(next: (snap: { exists: boolean; data(): Record<string, unknown> | undefined; metadata: { fromCache: boolean; hasPendingWrites: boolean } }) => void, error?: () => void): () => void;
}
declare global {
  interface Window { claude?: { use(name: string): Promise<unknown> } }
}

/**
 * Wires a form of radio groups and textareas (each textarea with `data-key`) to `docPath`.
 * The stored body is `{ picks: {group: value}, notes: {key: text}, at }`.
 */
export function connectNotes(form: HTMLFormElement, status: HTMLElement, docPath: string): void {
  const fields = Array.from(form.querySelectorAll<HTMLTextAreaElement>("textarea[data-key]"));
  const groups = [...new Set(Array.from(form.querySelectorAll<HTMLInputElement>('input[type="radio"]')).map((i) => i.name))];
  const read = () => ({
    picks: Object.fromEntries(groups.map((g) => [g, form.querySelector<HTMLInputElement>(`input[name="${g}"]:checked`)?.value ?? ""])),
    notes: Object.fromEntries(fields.map((t) => [t.dataset.key!, t.value])),
  });
  const fill = (body: Record<string, unknown> | undefined) => {
    if (!body) return;
    const notes = (body.notes ?? {}) as Record<string, string>, picks = (body.picks ?? {}) as Record<string, string>;
    for (const t of fields) if (document.activeElement !== t && typeof notes[t.dataset.key!] === "string") t.value = notes[t.dataset.key!]!;
    for (const g of groups) {
      const input = form.querySelector<HTMLInputElement>(`input[name="${g}"][value="${picks[g] ?? ""}"]`);
      if (input) input.checked = true;
    }
  };

  let save: ((body: object) => Promise<void>) | null = null;
  let timer = 0, writing = Promise.resolve(), last = "";
  form.addEventListener("input", () => {
    if (!save) return;
    clearTimeout(timer);
    status.textContent = "Saving…";
    timer = window.setTimeout(() => {
      const body = read(), text = JSON.stringify(body);
      if (text === last) { status.textContent = "Saved."; return; }
      writing = writing.then(() => save!(body)).then(
        () => { last = text; status.textContent = "Saved."; },
        () => { status.textContent = "Not saved: this view can't write notes. Copy them into the chat instead."; });
    }, 700);
  });

  const local = () => {
    try { fill(JSON.parse(localStorage.getItem(docPath) ?? "null") ?? undefined); } catch {}
    save = async (body) => { localStorage.setItem(docPath, JSON.stringify(body)); };
    status.textContent = "Notes are kept in this browser only.";
  };
  if (!window.claude?.use) { local(); return; }
  const waiting = window.setTimeout(() => { if (!save) status.textContent = "Notes can't be reached yet."; }, 10000);
  window.claude.use("db").then((found) => {
    const db = found as Db | null;
    if (!db) { clearTimeout(waiting); local(); return; }
    const doc = db.doc(docPath);
    save = (body) => doc.set({ ...body, at: Date.now() });
    status.textContent = "Notes save as you type, for Claude to read.";
    doc.onSnapshot((snap) => {
      if (!snap.metadata.fromCache) clearTimeout(waiting);
      if (snap.exists && !snap.metadata.hasPendingWrites) { fill(snap.data()); last = JSON.stringify(read()); }
    }, () => { clearTimeout(waiting); status.textContent = "Notes can't be reached. Copy them into the chat instead."; });
  }, () => { clearTimeout(waiting); local(); });
}
