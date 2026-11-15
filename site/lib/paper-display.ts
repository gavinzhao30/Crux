/** A readable filename fallback until extraction supplies the actual paper title. */
export function paperDisplayTitle(title: string | null, filename: string): string {
  if (title?.trim()) return title.trim();
  return filename.replace(/\.pdf$/i, "").replace(/^\d+[_\s-]+/, "")
    .replace(/[_]+/g, " ").replace(/\s+/g, " ").trim() || "Untitled paper";
}
