// The run state and the pure reducer that builds it from run events.
// Every panel renders only from RunState. Unknown event types and unknown
// payload fields are ignored; malformed payloads never throw.

import type {
  Artifact,
  CriticFinding,
  CurveFamily,
  CurvePoint,
  Evidence,
  Observation,
  OutcomeBin,
  QuestionKind,
  QuestionStatus,
  RankingRow,
  Report,
  RunEvent,
} from '../types';

/* ------------------------------------------------------------------ */
/* State shape                                                         */
/* ------------------------------------------------------------------ */

export type OptionStatus = 'active' | 'rejected' | 'accepted' | 'merged';

export interface OptionView {
  id: string;
  questionId: string;
  statement: string;
  author: string;
  rationale: string;
  answerKey: string | null;
  isNull: boolean;
  hasModel: boolean;
  prior: number;
  posterior: number;
  /** Posterior before the latest belief update, for the delta chip. */
  prevPosterior: number | null;
  /** Seq of the latest belief update that touched this option. */
  updateSeq: number | null;
  history: number[];
  /** Update count on the question when history[0] was recorded. */
  historyStart: number;
  status: OptionStatus;
  rejectedReason: string | null;
  rejectedEvidence: string | null;
  mergedInto: string | null;
  mergedFrom: string[];
  proposedSeq: number;
  /** Proposed after a surprise on its question (options regenerated). */
  regenerated: boolean;
}

export interface SurpriseView {
  reason: string;
  experimentId: string;
  zScore: number | null;
  nullMass: number | null;
  seq: number;
  /** Number of belief updates on the question when the surprise was raised. */
  atUpdate: number;
  resolved: boolean;
}

export interface QuestionView {
  id: string;
  text: string;
  kind: QuestionKind;
  impact: number;
  raisedBy: string;
  status: QuestionStatus;
  openedSeq: number;
  optionOrder: string[];
  options: Record<string, OptionView>;
  settled: { hypothesisId: string; posterior: number; seq: number } | null;
  robust: { answerKey: string; seq: number } | null;
  unresolved: { reason: string; top: string[]; seq: number } | null;
  surprise: SurpriseView | null;
  updates: number;
  lastObserved: string | number | null;
  lastExperimentId: string | null;
}

export type ExperimentPhase = 'proposed' | 'chosen' | 'running' | 'done' | 'invalid';

export interface ExperimentView {
  id: string;
  questionIds: string[];
  kind: string;
  goal: string;
  language: string | null;
  query: string | null;
  bins: OutcomeBin[];
  estCostS: number | null;
  estCostUsd: number | null;
  eigBits: number | null;
  score: number | null;
  status: ExperimentPhase;
  observation: Observation | null;
  verdict: { valid: boolean; note: string } | null;
  proposedSeq: number;
}

export interface RankingView {
  questionId: string;
  rows: RankingRow[];
  chosen: string;
  seq: number;
  round: number;
  phase: 'chosen' | 'running' | 'done' | 'invalid' | 'aborted';
  runningId: string | null;
  observed: string | number | null;
  verdict: { valid: boolean; note: string } | null;
}

export interface CompletedCheck {
  seq: number;
  experimentId: string;
  questionIds: string[];
  kind: string;
  goal: string;
  observed: string | number | null;
  valid: boolean | null;
}

export type ConsoleKind = 'start' | 'stdout' | 'stderr' | 'finish' | 'verdict' | 'error';

export interface ConsoleLine {
  seq: number;
  kind: ConsoleKind;
  experimentId: string | null;
  text: string;
  valid?: boolean;
}

export type FeedItem =
  | {
      kind: 'recall';
      seq: number;
      reader: string;
      authors: string[];
      evidenceIds: string[];
      query: string;
      backend: string;
      latencyMs: number | null;
    }
  | { kind: 'write'; seq: number; author: string; evidenceId: string; backend: string }
  | {
      kind: 'search';
      seq: number;
      experimentId: string;
      query: string;
      urls: string[];
      blocked: string[];
      binId: string | null;
    }
  | { kind: 'lesson'; seq: number; author: string; claim: string; evidenceId: string }
  | { kind: 'profile'; seq: number; author: string; claims: string[] }
  | { kind: 'question'; seq: number; author: string; questionId: string; text: string };

export interface RejectedRow {
  seq: number;
  questionId: string;
  hypothesisId: string;
  reason: string;
  evidenceId: string | null;
}

export interface ArtifactView extends Artifact {
  firstSeq: number;
  writtenSeq: number;
  versions: number;
}

export interface FindingView extends CriticFinding {
  seq: number;
}

export interface CurvesView {
  questionId: string;
  x: number[];
  families: Record<string, CurveFamily>;
  points: CurvePoint[];
  seq: number;
}

export interface BudgetView {
  costUsd: number;
  llmCalls: number;
  tokensIn: number;
  tokensOut: number;
  seconds: number;
}

export type RunStatus = 'idle' | 'running' | 'finished' | 'error';

export interface RunState {
  runId: string | null;
  lastSeq: number;
  eventCount: number;
  lab: string;
  goal: string;
  config: Record<string, unknown>;
  startedAt: string | null;
  status: RunStatus;
  finishStatus: 'ok' | 'budget_exhausted' | 'stopped' | null;
  error: { message: string; where: string } | null;
  firstTs: number | null;
  lastTs: number | null;
  budget: BudgetView | null;
  questionOrder: string[];
  questions: Record<string, QuestionView>;
  activeQuestionId: string | null;
  experiments: Record<string, ExperimentView>;
  rankings: Record<string, RankingView>;
  completed: CompletedCheck[];
  console: ConsoleLine[];
  feed: FeedItem[];
  evidence: Record<string, Evidence>;
  rejected: RejectedRow[];
  artifactOrder: string[];
  artifacts: Record<string, ArtifactView>;
  findings: FindingView[];
  curves: Record<string, CurvesView>;
  surprises: number;
  report: Report | null;
  malformed: number;
}

export const CONSOLE_LIMIT = 400;
export const FEED_LIMIT = 80;

export function initialState(): RunState {
  return {
    runId: null,
    lastSeq: -1,
    eventCount: 0,
    lab: '',
    goal: '',
    config: {},
    startedAt: null,
    status: 'idle',
    finishStatus: null,
    error: null,
    firstTs: null,
    lastTs: null,
    budget: null,
    questionOrder: [],
    questions: {},
    activeQuestionId: null,
    experiments: {},
    rankings: {},
    completed: [],
    console: [],
    feed: [],
    evidence: {},
    rejected: [],
    artifactOrder: [],
    artifacts: {},
    findings: [],
    curves: {},
    surprises: 0,
    report: null,
    malformed: 0,
  };
}

/* ------------------------------------------------------------------ */
/* Defensive readers                                                   */
/* ------------------------------------------------------------------ */

type Obj = Record<string, unknown>;

function isObj(v: unknown): v is Obj {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}
function obj(v: unknown): Obj {
  return isObj(v) ? v : {};
}
function str(v: unknown, fallback = ''): string {
  return typeof v === 'string' ? v : typeof v === 'number' ? String(v) : fallback;
}
function strOrNull(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null;
}
function num(v: unknown, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}
function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}
function bool(v: unknown, fallback = false): boolean {
  return typeof v === 'boolean' ? v : fallback;
}
function arr(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}
function strArr(v: unknown): string[] {
  return arr(v).filter((x): x is string => typeof x === 'string');
}
function numArr(v: unknown): number[] {
  return arr(v).map((x) => num(x, Number.NaN));
}
function probMap(v: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [k, p] of Object.entries(obj(v))) {
    if (typeof p === 'number' && Number.isFinite(p)) out[k] = p;
  }
  return out;
}
function observedValue(v: unknown): string | number | null {
  if (typeof v === 'string' || (typeof v === 'number' && Number.isFinite(v))) return v;
  return null;
}

function readEvidence(v: unknown): Evidence | null {
  const e = obj(v);
  const id = str(e.id);
  if (!id) return null;
  return {
    id,
    kind: str(e.kind, 'observation') as Evidence['kind'],
    claim: str(e.claim),
    author_role: str(e.author_role, 'agent'),
    source_ref: strOrNull(e.source_ref),
    url: strOrNull(e.url),
    conditions: obj(e.conditions),
  };
}

function readBins(v: unknown): OutcomeBin[] {
  return arr(v)
    .map((b) => {
      const o = obj(b);
      return { id: str(o.id), description: str(o.description), rule: str(o.rule) };
    })
    .filter((b) => b.id !== '');
}

function readObservation(v: unknown): Observation | null {
  const o = obj(v);
  const id = str(o.experiment_id);
  if (!id) return null;
  const value = o.value;
  return {
    experiment_id: id,
    valid: bool(o.valid, true),
    value: typeof value === 'number' || isObj(value) ? (value as Observation['value']) : null,
    bin_id: strOrNull(o.bin_id),
    surprise: num(o.surprise),
    referee_note: str(o.referee_note),
    log_ref: str(o.log_ref),
    seconds: num(o.seconds),
    cost_usd: num(o.cost_usd),
  };
}

function readArtifact(v: unknown): Artifact | null {
  const a = obj(v);
  const name = str(a.name);
  if (!name) return null;
  return {
    name,
    path: str(a.path),
    sha256: str(a.sha256),
    contract_ok: bool(a.contract_ok),
    findings: strArr(a.findings),
  };
}

/* ------------------------------------------------------------------ */
/* Small immutable helpers                                             */
/* ------------------------------------------------------------------ */

function blankQuestion(id: string, seq: number): QuestionView {
  return {
    id,
    text: id,
    kind: 'categorical',
    impact: 1,
    raisedBy: '',
    status: 'open',
    openedSeq: seq,
    optionOrder: [],
    options: {},
    settled: null,
    robust: null,
    unresolved: null,
    surprise: null,
    updates: 0,
    lastObserved: null,
    lastExperimentId: null,
  };
}

function blankOption(id: string, questionId: string, seq: number): OptionView {
  return {
    id,
    questionId,
    statement: id,
    author: '',
    rationale: '',
    answerKey: null,
    isNull: id === 'h0',
    hasModel: false,
    prior: 0,
    posterior: 0,
    prevPosterior: null,
    updateSeq: null,
    history: [],
    historyStart: 0,
    status: 'active',
    rejectedReason: null,
    rejectedEvidence: null,
    mergedInto: null,
    mergedFrom: [],
    proposedSeq: seq,
    regenerated: false,
  };
}

function blankExperiment(id: string, seq: number): ExperimentView {
  return {
    id,
    questionIds: [],
    kind: 'compute',
    goal: '',
    language: null,
    query: null,
    bins: [],
    estCostS: null,
    estCostUsd: null,
    eigBits: null,
    score: null,
    status: 'proposed',
    observation: null,
    verdict: null,
    proposedSeq: seq,
  };
}

function withQuestion(
  s: RunState,
  qid: string,
  seq: number,
  fn: (q: QuestionView) => QuestionView,
): RunState {
  const existing = s.questions[qid];
  const q = fn(existing ?? blankQuestion(qid, seq));
  return {
    ...s,
    questions: { ...s.questions, [qid]: q },
    questionOrder: existing ? s.questionOrder : [...s.questionOrder, qid],
  };
}

function withOption(
  q: QuestionView,
  hid: string,
  seq: number,
  fn: (o: OptionView) => OptionView,
): QuestionView {
  const existing = q.options[hid];
  const o = fn(existing ?? blankOption(hid, q.id, seq));
  return {
    ...q,
    options: { ...q.options, [hid]: o },
    optionOrder: existing ? q.optionOrder : [...q.optionOrder, hid],
  };
}

function withExperiment(
  s: RunState,
  eid: string,
  seq: number,
  fn: (e: ExperimentView) => ExperimentView,
): RunState {
  const e = fn(s.experiments[eid] ?? blankExperiment(eid, seq));
  return { ...s, experiments: { ...s.experiments, [eid]: e } };
}

function pushConsole(s: RunState, line: ConsoleLine): RunState {
  const next = [...s.console, line];
  return { ...s, console: next.length > CONSOLE_LIMIT ? next.slice(-CONSOLE_LIMIT) : next };
}

function pushFeed(s: RunState, item: FeedItem): RunState {
  const next = [item, ...s.feed];
  return { ...s, feed: next.length > FEED_LIMIT ? next.slice(0, FEED_LIMIT) : next };
}

function mapRankings(
  s: RunState,
  fn: (r: RankingView) => RankingView,
): Record<string, RankingView> {
  const out: Record<string, RankingView> = {};
  for (const [k, r] of Object.entries(s.rankings)) out[k] = fn(r);
  return out;
}

/* ------------------------------------------------------------------ */
/* Event handlers                                                      */
/* ------------------------------------------------------------------ */

type Handler = (s: RunState, p: Obj, ev: RunEvent) => RunState;

const handlers: Record<string, Handler> = {
  'run.started': (s, p, ev) => ({
    ...s,
    runId: ev.run_id || s.runId,
    lab: str(p.lab, s.lab),
    goal: str(p.goal, s.goal),
    config: obj(p.config),
    startedAt: strOrNull(p.started_at),
    status: 'running',
  }),

  'profile.done': (s, p, ev) => {
    const evidence = arr(p.evidence)
      .map(readEvidence)
      .filter((e): e is Evidence => e !== null);
    const map = { ...s.evidence };
    for (const e of evidence) map[e.id] = e;
    const next = { ...s, evidence: map };
    if (evidence.length === 0) return next;
    return pushFeed(next, {
      kind: 'profile',
      seq: ev.seq,
      author: evidence[0]?.author_role ?? 'profiler',
      claims: evidence.map((e) => e.claim),
    });
  },

  'question.opened': (s, p, ev) => {
    const q = obj(p.question);
    const id = str(q.id);
    if (!id) return s;
    const next = withQuestion(s, id, ev.seq, (cur) => ({
      ...cur,
      text: str(q.text, cur.text),
      kind: (str(q.kind, cur.kind) as QuestionKind) || 'categorical',
      impact: num(q.impact, cur.impact),
      raisedBy: str(q.raised_by, cur.raisedBy),
      status: (str(q.status, cur.status) as QuestionStatus) || 'open',
      openedSeq: ev.seq,
    }));
    return pushFeed(next, {
      kind: 'question',
      seq: ev.seq,
      author: str(q.raised_by, 'agent'),
      questionId: id,
      text: str(q.text, id),
    });
  },

  'hypothesis.proposed': (s, p, ev) => {
    const h = obj(p.hypothesis);
    const id = str(h.id);
    const qid = str(h.question_id);
    if (!id || !qid) return s;
    return withQuestion(s, qid, ev.seq, (q) => {
      const afterSurprise = q.surprise !== null && ev.seq > q.surprise.seq;
      return withOption(q, id, ev.seq, (o) => {
        const prior = num(h.prior, o.prior);
        const posterior = num(h.posterior, prior);
        const status = str(h.status, 'active');
        return {
          ...o,
          statement: str(h.statement, o.statement),
          author: str(h.author, o.author),
          rationale: str(h.rationale, o.rationale),
          answerKey: strOrNull(h.answer_key),
          isNull: bool(h.is_null, o.isNull),
          hasModel: typeof h.model_code === 'string' && h.model_code.length > 0,
          prior,
          posterior,
          history: o.history.length ? o.history : [posterior],
          historyStart: o.history.length ? o.historyStart : q.updates,
          status: status === 'rejected' || status === 'accepted' ? status : 'active',
          proposedSeq: ev.seq,
          regenerated: o.regenerated || afterSurprise,
        };
      });
    });
  },

  'hypothesis.merged': (s, p, ev) => {
    const qid = str(p.question_id);
    const kept = str(p.kept);
    if (!qid || !kept) return s;
    const merged = strArr(p.merged).filter((m) => m !== kept);
    const priors = probMap(p.priors);
    return withQuestion(s, qid, ev.seq, (q) => {
      let next = q;
      for (const m of merged) {
        next = withOption(next, m, ev.seq, (o) => ({
          ...o,
          status: 'merged',
          mergedInto: kept,
          prior: 0,
          posterior: 0,
        }));
      }
      next = withOption(next, kept, ev.seq, (o) => ({
        ...o,
        mergedFrom: [...o.mergedFrom, ...merged.filter((m) => !o.mergedFrom.includes(m))],
      }));
      for (const [hid, pr] of Object.entries(priors)) {
        next = withOption(next, hid, ev.seq, (o) => ({
          ...o,
          prior: pr,
          posterior: pr,
          history: [...o.history.slice(0, -1), pr],
        }));
      }
      return next;
    });
  },

  'experiment.proposed': (s, p, ev) => {
    const e = obj(p.experiment);
    const id = str(e.id);
    if (!id) return s;
    return withExperiment(s, id, ev.seq, (cur) => ({
      ...cur,
      questionIds: strArr(e.question_ids).length ? strArr(e.question_ids) : cur.questionIds,
      kind: str(e.kind, cur.kind),
      goal: str(e.goal, cur.goal),
      language: strOrNull(e.language) ?? cur.language,
      query: strOrNull(e.query) ?? cur.query,
      bins: readBins(e.bins).length ? readBins(e.bins) : cur.bins,
      estCostS: numOrNull(e.est_cost_s) ?? cur.estCostS,
      estCostUsd: numOrNull(e.est_cost_usd) ?? cur.estCostUsd,
      eigBits: numOrNull(e.eig_bits) ?? cur.eigBits,
      score: numOrNull(e.score) ?? cur.score,
      proposedSeq: ev.seq,
    }));
  },

  'experiment.scored': (s, p, ev) => {
    const qid = str(p.question_id);
    if (!qid) return s;
    const rows: RankingRow[] = arr(p.ranking).map((r) => {
      const o = obj(r);
      return {
        experiment_id: str(o.experiment_id),
        kind: str(o.kind, 'compute'),
        goal: str(o.goal),
        eig_bits: num(o.eig_bits),
        est_cost_s: num(o.est_cost_s),
        est_cost_usd: num(o.est_cost_usd),
        score: num(o.score),
      };
    });
    const chosen = str(p.chosen);
    const prev = s.rankings[qid];
    let next: RunState = {
      ...s,
      activeQuestionId: qid,
      rankings: {
        ...s.rankings,
        [qid]: {
          questionId: qid,
          rows,
          chosen,
          seq: ev.seq,
          round: (prev?.round ?? 0) + 1,
          phase: 'chosen',
          runningId: null,
          observed: null,
          verdict: null,
        },
      },
    };
    for (const r of rows) {
      if (!r.experiment_id || !next.experiments[r.experiment_id]) continue;
      next = withExperiment(next, r.experiment_id, ev.seq, (e) => ({
        ...e,
        eigBits: r.eig_bits,
        score: r.score,
        status: r.experiment_id === chosen && e.status === 'proposed' ? 'chosen' : e.status,
      }));
    }
    return next;
  },

  'experiment.started': (s, p, ev) => {
    const id = str(p.experiment_id);
    if (!id) return s;
    const qids = strArr(p.question_ids);
    const kind = str(p.kind);
    let next = withExperiment(s, id, ev.seq, (e) => ({
      ...e,
      kind: kind || e.kind,
      questionIds: qids.length ? qids : e.questionIds,
      status: 'running',
    }));
    next = {
      ...next,
      activeQuestionId: qids[0] ?? next.activeQuestionId,
      rankings: mapRankings(next, (r) =>
        qids.includes(r.questionId) && (r.chosen === id || r.phase === 'chosen')
          ? { ...r, phase: 'running', runningId: id }
          : r,
      ),
    };
    return pushConsole(next, {
      seq: ev.seq,
      kind: 'start',
      experimentId: id,
      text: `${id} started${kind ? ` (${kind})` : ''}`,
    });
  },

  'experiment.log': (s, p, ev) => {
    const stream = str(p.stream) === 'stderr' ? 'stderr' : 'stdout';
    return pushConsole(s, {
      seq: ev.seq,
      kind: stream,
      experimentId: strOrNull(p.experiment_id),
      text: str(p.line).slice(0, 500),
    });
  },

  'experiment.finished': (s, p, ev) => {
    const obs = readObservation(p.observation);
    if (!obs) return s;
    const id = obs.experiment_id;
    const observed = obs.bin_id ?? (typeof obs.value === 'number' ? obs.value : null);
    let next = withExperiment(s, id, ev.seq, (e) => ({
      ...e,
      observation: obs,
      status: obs.valid ? 'done' : 'invalid',
    }));
    const exp = next.experiments[id];
    next = {
      ...next,
      rankings: mapRankings(next, (r) =>
        r.runningId === id || (r.chosen === id && r.phase !== 'done')
          ? { ...r, phase: obs.valid ? 'done' : 'invalid', observed, runningId: r.runningId ?? id }
          : r,
      ),
      completed: [
        ...next.completed,
        {
          seq: ev.seq,
          experimentId: id,
          questionIds: exp?.questionIds ?? [],
          kind: exp?.kind ?? 'compute',
          goal: exp?.goal ?? '',
          observed,
          valid: null,
        },
      ],
    };
    const shown =
      observed === null ? 'no outcome' : typeof observed === 'number' ? fmtNum(observed) : observed;
    return pushConsole(next, {
      seq: ev.seq,
      kind: 'finish',
      experimentId: id,
      text: `${id} finished -> ${shown}${obs.seconds ? ` in ${obs.seconds.toFixed(1)} s` : ''}`,
    });
  },

  'referee.verdict': (s, p, ev) => {
    const id = str(p.experiment_id);
    if (!id) return s;
    const verdict = { valid: bool(p.valid, true), note: str(p.note) };
    let next = withExperiment(s, id, ev.seq, (e) => ({
      ...e,
      verdict,
      status: verdict.valid ? e.status : 'invalid',
    }));
    next = {
      ...next,
      rankings: mapRankings(next, (r) =>
        r.runningId === id || r.chosen === id
          ? { ...r, verdict, phase: verdict.valid ? r.phase : 'invalid' }
          : r,
      ),
      completed: next.completed.map((c) =>
        c.experimentId === id && c.valid === null ? { ...c, valid: verdict.valid } : c,
      ),
    };
    return pushConsole(next, {
      seq: ev.seq,
      kind: 'verdict',
      experimentId: id,
      text: verdict.note,
      valid: verdict.valid,
    });
  },

  'belief.updated': (s, p, ev) => {
    const qid = str(p.question_id);
    if (!qid) return s;
    const posterior = probMap(p.posterior);
    const observed = observedValue(p.observed);
    const eid = str(p.experiment_id);
    const next = withQuestion(s, qid, ev.seq, (q) => {
      let cur: QuestionView = {
        ...q,
        updates: q.updates + 1,
        lastObserved: observed,
        lastExperimentId: eid || q.lastExperimentId,
      };
      for (const [hid, post] of Object.entries(posterior)) {
        cur = withOption(cur, hid, ev.seq, (o) => ({
          ...o,
          prevPosterior: o.posterior,
          posterior: post,
          updateSeq: ev.seq,
          history: [...o.history, post],
        }));
      }
      return cur;
    });
    return {
      ...next,
      rankings: mapRankings(next, (r) =>
        r.questionId === qid && r.observed === null && (r.runningId === eid || r.chosen === eid)
          ? { ...r, observed }
          : r,
      ),
    };
  },

  'model.curves': (s, p, ev) => {
    const qid = str(p.question_id);
    if (!qid) return s;
    const families: Record<string, CurveFamily> = {};
    for (const [hid, f] of Object.entries(obj(p.families))) {
      const o = obj(f);
      families[hid] = { mean: numArr(o.mean), sd: numArr(o.sd), weight: num(o.weight) };
    }
    const points: CurvePoint[] = arr(p.points)
      .map((pt) => {
        const o = obj(pt);
        return { x: num(o.x, Number.NaN), y: num(o.y, Number.NaN) };
      })
      .filter((pt) => Number.isFinite(pt.x) && Number.isFinite(pt.y));
    return {
      ...s,
      curves: {
        ...s.curves,
        [qid]: { questionId: qid, x: numArr(p.x), families, points, seq: ev.seq },
      },
    };
  },

  'hypothesis.rejected': (s, p, ev) => {
    const qid = str(p.question_id);
    const hid = str(p.hypothesis_id);
    if (!qid || !hid) return s;
    const reason = str(p.reason);
    const evidenceId = strOrNull(p.evidence_id);
    const next = withQuestion(s, qid, ev.seq, (q) =>
      withOption(q, hid, ev.seq, (o) => ({
        ...o,
        status: 'rejected',
        rejectedReason: reason,
        rejectedEvidence: evidenceId,
      })),
    );
    const rest = next.rejected.filter((r) => !(r.questionId === qid && r.hypothesisId === hid));
    return {
      ...next,
      rejected: [{ seq: ev.seq, questionId: qid, hypothesisId: hid, reason, evidenceId }, ...rest],
    };
  },

  'question.settled': (s, p, ev) => {
    const qid = str(p.question_id);
    const hid = str(p.hypothesis_id);
    if (!qid) return s;
    return withQuestion(s, qid, ev.seq, (q) => {
      let cur: QuestionView = {
        ...q,
        status: 'settled',
        settled: { hypothesisId: hid, posterior: num(p.posterior), seq: ev.seq },
        surprise: q.surprise ? { ...q.surprise, resolved: true } : null,
      };
      // Only one option is accepted at a time.
      for (const oid of cur.optionOrder) {
        const o = cur.options[oid];
        if (o && o.status === 'accepted' && oid !== hid) {
          cur = withOption(cur, oid, ev.seq, (x) => ({ ...x, status: 'active' }));
        }
      }
      if (hid) {
        cur = withOption(cur, hid, ev.seq, (o) => ({
          ...o,
          status: o.status === 'rejected' ? o.status : 'accepted',
        }));
      }
      return cur;
    });
  },

  'question.robust': (s, p, ev) => {
    const qid = str(p.question_id);
    if (!qid) return s;
    return withQuestion(s, qid, ev.seq, (q) => ({
      ...q,
      status: 'robust',
      robust: { answerKey: str(p.answer_key), seq: ev.seq },
      surprise: q.surprise ? { ...q.surprise, resolved: true } : null,
    }));
  },

  'question.unresolved': (s, p, ev) => {
    const qid = str(p.question_id);
    if (!qid) return s;
    return withQuestion(s, qid, ev.seq, (q) => ({
      ...q,
      status: 'unresolved',
      unresolved: { reason: str(p.reason), top: strArr(p.top), seq: ev.seq },
    }));
  },

  'surprise.raised': (s, p, ev) => {
    const qid = str(p.question_id);
    if (!qid) return s;
    const next = withQuestion(s, qid, ev.seq, (q) => {
      let cur: QuestionView = {
        ...q,
        status: q.status === 'settled' ? 'open' : q.status,
        settled: q.status === 'settled' ? null : q.settled,
        surprise: {
          reason: str(p.reason),
          experimentId: str(p.experiment_id),
          zScore: numOrNull(p.z_score),
          nullMass: numOrNull(p.null_mass),
          seq: ev.seq,
          atUpdate: q.updates,
          resolved: false,
        },
      };
      for (const oid of cur.optionOrder) {
        if (cur.options[oid]?.status === 'accepted') {
          cur = withOption(cur, oid, ev.seq, (o) => ({ ...o, status: 'active' }));
        }
      }
      return cur;
    });
    return { ...next, surprises: next.surprises + 1 };
  },

  'search.done': (s, p, ev) =>
    pushFeed(s, {
      kind: 'search',
      seq: ev.seq,
      experimentId: str(p.experiment_id),
      query: str(p.query),
      urls: strArr(p.urls),
      blocked: strArr(p.blocked),
      binId: strOrNull(p.bin_id),
    }),

  'memory.written': (s, p, ev) =>
    pushFeed(s, {
      kind: 'write',
      seq: ev.seq,
      author: str(p.author_role, 'agent'),
      evidenceId: str(p.evidence_id),
      backend: str(p.backend),
    }),

  'memory.recalled': (s, p, ev) =>
    pushFeed(s, {
      kind: 'recall',
      seq: ev.seq,
      reader: str(p.reader_role, 'agent'),
      authors: strArr(p.authors),
      evidenceIds: strArr(p.evidence_ids),
      query: str(p.query),
      backend: str(p.backend),
      latencyMs: numOrNull(p.latency_ms),
    }),

  'artifact.written': (s, p, ev) => {
    const a = readArtifact(p.artifact);
    if (!a) return s;
    const prev = s.artifacts[a.name];
    // The engine re-emits an artifact with the critic's verdict (same sha256); that is the
    // same version with an updated contract_ok, not a new version.
    const sameVersion = Boolean(prev && a.sha256 && prev.sha256 === a.sha256);
    return {
      ...s,
      artifacts: {
        ...s.artifacts,
        [a.name]: {
          ...a,
          firstSeq: prev?.firstSeq ?? ev.seq,
          writtenSeq: ev.seq,
          versions: sameVersion && prev ? prev.versions : (prev?.versions ?? 0) + 1,
        },
      },
      artifactOrder: prev ? s.artifactOrder : [...s.artifactOrder, a.name],
    };
  },

  'critic.finding': (s, p, ev) => {
    const f = obj(p.finding);
    const id = str(f.id);
    const artifact = str(f.artifact);
    if (!id && !artifact) return s;
    const finding: FindingView = {
      id: id || `${artifact}:${str(f.rule)}`,
      artifact,
      severity: str(f.severity) === 'block' ? 'block' : 'warn',
      rule: str(f.rule, 'contract'),
      message: str(f.message),
      reopens_question: strOrNull(f.reopens_question),
      seq: ev.seq,
    };
    return {
      ...s,
      findings: [...s.findings.filter((x) => x.id !== finding.id), finding],
    };
  },

  'lesson.written': (s, p, ev) => {
    const lesson = readEvidence(p.lesson);
    if (!lesson) return s;
    return pushFeed(
      { ...s, evidence: { ...s.evidence, [lesson.id]: lesson } },
      {
        kind: 'lesson',
        seq: ev.seq,
        author: lesson.author_role,
        claim: lesson.claim,
        evidenceId: lesson.id,
      },
    );
  },

  'budget.tick': (s, p) => ({
    ...s,
    budget: {
      costUsd: num(p.cost_usd, s.budget?.costUsd ?? 0),
      llmCalls: num(p.llm_calls, s.budget?.llmCalls ?? 0),
      tokensIn: num(p.tokens_in, s.budget?.tokensIn ?? 0),
      tokensOut: num(p.tokens_out, s.budget?.tokensOut ?? 0),
      seconds: num(p.seconds, s.budget?.seconds ?? 0),
    },
  }),

  'run.finished': (s, p, ev) => {
    const r = obj(p.report);
    const status = str(p.status, 'ok');
    const report: Report = {
      run_id: str(r.run_id, s.runId ?? ''),
      artifacts: arr(r.artifacts)
        .map(readArtifact)
        .filter((a): a is Artifact => a !== null),
      methods_md: str(r.methods_md),
      rejected: arr(r.rejected).map((x) => {
        const o = obj(x);
        return {
          hypothesis: str(o.hypothesis),
          question: str(o.question),
          evidence_id: strOrNull(o.evidence_id),
          reason: str(o.reason),
        };
      }),
      unresolved: strArr(r.unresolved),
      cost_usd: num(r.cost_usd),
      seconds: num(r.seconds),
    };
    // Rejections only present in the report still reach the ledger.
    const extra: RejectedRow[] = report.rejected
      .filter(
        (x) =>
          x.hypothesis &&
          !s.rejected.some((row) => row.hypothesisId === x.hypothesis && row.questionId === x.question),
      )
      .map((x) => ({
        seq: ev.seq,
        questionId: x.question,
        hypothesisId: x.hypothesis,
        reason: x.reason,
        evidenceId: x.evidence_id,
      }));
    const budget: BudgetView = {
      costUsd: Math.max(s.budget?.costUsd ?? 0, report.cost_usd),
      llmCalls: s.budget?.llmCalls ?? 0,
      tokensIn: s.budget?.tokensIn ?? 0,
      tokensOut: s.budget?.tokensOut ?? 0,
      seconds: Math.max(s.budget?.seconds ?? 0, report.seconds),
    };
    return {
      ...s,
      status: 'finished',
      finishStatus:
        status === 'budget_exhausted' || status === 'stopped' ? status : 'ok',
      report,
      budget,
      rejected: [...extra, ...s.rejected],
    };
  },

  'run.error': (s, p, ev) => {
    const error = { message: str(p.message, 'Unknown error'), where: str(p.where) };
    // Whatever was in flight will not finish.
    const experiments: Record<string, ExperimentView> = {};
    for (const [id, e] of Object.entries(s.experiments)) {
      experiments[id] = e.status === 'running' || e.status === 'chosen' ? { ...e, status: 'invalid' } : e;
    }
    const rankings = mapRankings(s, (r) =>
      r.phase === 'running' || r.phase === 'chosen' ? { ...r, phase: 'aborted' } : r,
    );
    return pushConsole(
      { ...s, status: 'error', error, experiments, rankings },
      {
        seq: ev.seq,
        kind: 'error',
        experimentId: null,
        text: `${error.message}${error.where ? ` (${error.where})` : ''}`,
      },
    );
  },
};

function fmtNum(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2);
}

export function isKnownEventType(type: string): boolean {
  return Object.prototype.hasOwnProperty.call(handlers, type);
}

/* ------------------------------------------------------------------ */
/* The reducer                                                         */
/* ------------------------------------------------------------------ */

/**
 * Fold one run event into the state. Pure: returns a new state object and
 * never mutates or throws. Duplicate events (seq already seen for this run,
 * e.g. after a reconnect) are dropped.
 */
export function foldEvent(state: RunState, event: RunEvent | unknown): RunState {
  if (!isObj(event)) return state;
  const type = str(event.type);
  if (!type) return state;
  const seq = num(event.seq, state.lastSeq + 1);
  const runId = str(event.run_id);
  const ts = numOrNull(event.ts);

  if (state.lastSeq >= 0 && seq <= state.lastSeq && (!runId || runId === state.runId)) {
    return state;
  }

  const handler = Object.prototype.hasOwnProperty.call(handlers, type) ? handlers[type] : undefined;
  if (!handler) {
    // Unknown event type: only remember that we have seen this seq.
    return { ...state, lastSeq: Math.max(state.lastSeq, seq) };
  }

  const ev: RunEvent = { run_id: runId, seq, ts: ts ?? 0, type, payload: event.payload };
  const base: RunState = {
    ...state,
    runId: state.runId ?? (runId || null),
    lastSeq: Math.max(state.lastSeq, seq),
    eventCount: state.eventCount + 1,
    firstTs: state.firstTs ?? ts,
    lastTs: ts ?? state.lastTs,
  };
  try {
    return handler(base, obj(event.payload), ev);
  } catch {
    return { ...base, malformed: base.malformed + 1 };
  }
}

export function foldAll(events: Iterable<unknown>, from: RunState = initialState()): RunState {
  let s = from;
  for (const e of events) s = foldEvent(s, e);
  return s;
}
