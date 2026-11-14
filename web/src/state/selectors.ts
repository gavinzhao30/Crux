// Derived views over RunState. Pure functions; panels call these instead of
// re-deriving logic inline.

import type { FindingView, OptionView, QuestionView, RunState } from './fold';
import type { Artifact } from '../types';

export function visibleOptions(q: QuestionView): OptionView[] {
  const opts = q.optionOrder
    .map((id) => q.options[id])
    .filter((o): o is OptionView => o !== undefined && o.status !== 'merged');
  // "None of these" always sits last.
  return [...opts.filter((o) => !o.isNull), ...opts.filter((o) => o.isNull)];
}

/** The option with the highest posterior; null while the top two are tied. */
export function leaderId(q: QuestionView): string | null {
  const ranked = visibleOptions(q)
    .filter((o) => o.status !== 'rejected')
    .sort((a, b) => b.posterior - a.posterior);
  const [first, second] = ranked;
  if (!first || first.posterior <= 0) return null;
  if (second && first.posterior - second.posterior < 0.005) return null;
  return first.id;
}

export function optionStatement(s: RunState, questionId: string, hypothesisId: string): string {
  return s.questions[questionId]?.options[hypothesisId]?.statement ?? hypothesisId;
}

export function questionLabel(s: RunState, questionId: string): string {
  const idx = s.questionOrder.indexOf(questionId);
  return idx >= 0 ? `Q${idx + 1}` : questionId;
}

/* Critic ------------------------------------------------------------ */

export type CheckState = 'pass' | 'block' | 'warn' | 'checking';

export interface RuleLine {
  finding: FindingView;
  resolved: boolean;
}

export interface ArtifactCheck {
  artifact: Artifact & { writtenSeq: number; versions: number };
  status: CheckState;
  rules: RuleLine[];
}

/**
 * A finding counts as resolved when the artifact was written again after it,
 * that version passes its contract, and it no longer lists the finding.
 */
export function artifactChecks(s: RunState): ArtifactCheck[] {
  const names = [...s.artifactOrder];
  for (const f of s.findings) if (f.artifact && !names.includes(f.artifact)) names.push(f.artifact);
  return names.map((name) => {
    const a = s.artifacts[name] ?? {
      name,
      path: '',
      sha256: '',
      contract_ok: false,
      findings: [],
      writtenSeq: -1,
      versions: 0,
      firstSeq: -1,
    };
    const rules = s.findings
      .filter((f) => f.artifact === name)
      .map((finding) => ({
        finding,
        resolved:
          a.writtenSeq > finding.seq &&
          a.contract_ok &&
          !a.findings.includes(finding.id) &&
          !a.findings.includes(finding.rule),
      }));
    const open = rules.filter((r) => !r.resolved);
    let status: CheckState;
    if (open.some((r) => r.finding.severity === 'block')) status = 'block';
    else if (open.some((r) => r.finding.severity === 'warn')) status = 'warn';
    else if (a.contract_ok) status = 'pass';
    else status = 'checking';
    return { artifact: a, status, rules };
  });
}

export function criticSummary(s: RunState): Record<string, CheckState> {
  const out: Record<string, CheckState> = {};
  for (const c of artifactChecks(s)) out[c.artifact.name] = c.status;
  return out;
}

/* Time and money ---------------------------------------------------- */

export function elapsedSeconds(s: RunState): number {
  if (s.budget && s.budget.seconds > 0) return s.budget.seconds;
  if (s.firstTs !== null && s.lastTs !== null) return Math.max(0, s.lastTs - s.firstTs);
  return 0;
}

/* Numeric labs ------------------------------------------------------ */

/** Parse a design label such as "x=1.75" (or "x=128#9") into its x value. */
export function parseDesignX(label: string | null | undefined): number | null {
  if (!label) return null;
  const m = /x\s*=\s*(-?\d+(?:\.\d+)?)/i.exec(label);
  if (m && m[1] !== undefined) return Number(m[1]);
  const n = Number(label);
  return Number.isFinite(n) && label.trim() !== '' ? n : null;
}

export function isNumericLab(s: RunState): boolean {
  return Object.keys(s.curves).length > 0;
}

/** The question the experiment panel should follow. */
export function focusQuestionId(s: RunState): string | null {
  if (s.activeQuestionId && s.questions[s.activeQuestionId]) return s.activeQuestionId;
  return s.questionOrder[s.questionOrder.length - 1] ?? null;
}

/** The next design to mark on the curve chart (chosen or running, not yet observed). */
export function nextDesign(s: RunState, questionId: string): { x: number; label: string } | null {
  const r = s.rankings[questionId];
  if (r && (r.phase === 'chosen' || r.phase === 'running')) {
    const label = r.runningId ?? r.chosen;
    const x = parseDesignX(label);
    if (x !== null) return { x, label };
  }
  return null;
}

/* Evidence ---------------------------------------------------------- */

/** Best human-readable description of an evidence id. */
export function evidenceText(s: RunState, evidenceId: string | null): string | null {
  if (!evidenceId) return null;
  const ev = s.evidence[evidenceId];
  if (ev?.claim) return ev.claim;
  // Fall back to the experiment whose id appears in the evidence id.
  const ids = Object.keys(s.experiments).sort((a, b) => b.length - a.length);
  for (const id of ids) {
    if (!evidenceId.includes(id)) continue;
    const e = s.experiments[id];
    const obs = e?.observation;
    if (!e || !obs) continue;
    if (!obs.bin_id && typeof obs.value === 'number') return `measured ${obs.value} at ${id}`;
    const bin = obs.bin_id ? e.bins.find((b) => b.id === obs.bin_id) : undefined;
    const outcome = bin?.description ?? obs.bin_id ?? '';
    return outcome ? `${e.goal || id}: ${outcome}` : e.goal || null;
  }
  return null;
}

export function binDescription(s: RunState, experimentId: string | null, binId: string): string | null {
  if (!experimentId) return null;
  return s.experiments[experimentId]?.bins.find((b) => b.id === binId)?.description ?? null;
}
