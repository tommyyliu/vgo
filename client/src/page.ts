// What every built page starts from: the rules, loaded from the wasm the builder inlined,
// and the data it was built with (tools/page.ts).

import { loadRules, type Rules } from "./rules";

export function pageRules(): Rules {
  const text = document.getElementById("vgo-wasm")?.textContent;
  if (!text) throw new Error("this page was built without the rules (tools/page.ts inlines them)");
  const binary = atob(text.trim());
  const bytes = new Uint8Array(binary.length);
  for (let k = 0; k < binary.length; k++) bytes[k] = binary.charCodeAt(k);
  return loadRules(bytes);
}

export function pageData(): Record<string, string> {
  const text = document.getElementById("vgo-data")?.textContent;
  return text ? JSON.parse(text) : {};
}
