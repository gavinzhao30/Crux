// Generates the bundled replay fixtures in public/fixtures/.
//
//   node scripts/make-fixtures.mjs
//
// Posteriors, expected information gain and scores in the categorical
// fixture are computed with the formulas in docs/ERD.md section 6.1 from the
// likelihoods carried by each experiment.proposed event, so a replay checker
// can rebuild every belief.updated from the file alone. Output is
// deterministic.

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../public/fixtures');

/* ------------------------------------------------------------------ */
/* Shared helpers                                                      */
/* ------------------------------------------------------------------ */

const r3 = (x) => Math.round(x * 1000) / 1000;
const r4 = (x) => Math.round(x * 10000) / 10000;
const r2 = (x) => Math.round(x * 100) / 100;

function sha(seed) {
  // Deterministic 64-hex string (not a real digest; fixtures only). Folds the whole seed in
  // first, so seeds that share a prefix (e.g. ".../v1" and ".../v2") give different strings.
  let h = 2166136261 >>> 0;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  let out = '';
  for (let i = 0; out.length < 64; i++) {
    h ^= (i + 0x9e3779b9) >>> 0;
    h = Math.imul(h, 16777619) >>> 0;
    out += h.toString(16).padStart(8, '0');
  }
  return out.slice(0, 64);
}

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function gauss(rand) {
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

class Recorder {
  constructor(runId, t0, scale = 1) {
    this.scale = scale;
    this.runId = runId;
    this.t0 = t0;
    this.t = t0;
    this.seq = 0;
    this.events = [];
    this.cost = 0;
    this.calls = 0;
    this.tokIn = 0;
    this.tokOut = 0;
  }
  emit(type, payload, dt = 0.5) {
    this.t += dt * this.scale;
    this.events.push({ run_id: this.runId, seq: this.seq++, ts: r3(this.t), type, payload });
  }
  llm(calls, usd, tokIn, tokOut) {
    this.calls += calls;
    this.cost += usd;
    this.tokIn += tokIn;
    this.tokOut += tokOut;
  }
  tick(dt = 0.2) {
    this.emit(
      'budget.tick',
      {
        cost_usd: r4(this.cost),
        llm_calls: this.calls,
        tokens_in: this.tokIn,
        tokens_out: this.tokOut,
        seconds: r2(this.t + dt - this.t0),
      },
      dt,
    );
  }
  write(name) {
    const body = this.events.map((e) => JSON.stringify(e)).join('\n') + '\n';
    writeFileSync(resolve(OUT, name), body);
    const span = this.events.at(-1).ts - this.events[0].ts;
    console.log(`${name}: ${this.events.length} events, ${span.toFixed(1)} s`);
  }
}

/* ------------------------------------------------------------------ */
/* Categorical scoring (ERD 6.1)                                       */
/* ------------------------------------------------------------------ */

const N_SAMPLES = 8;
const EPS = 0.1;

function likelihoodFromCounts(counts, binIds) {
  const B = binIds.length;
  const out = {};
  binIds.forEach((b, i) => {
    out[b] = (1 - EPS) * (counts[i] + 0.5) / (N_SAMPLES + 0.5 * B) + EPS / B;
  });
  return out;
}

function uniform(binIds) {
  return Object.fromEntries(binIds.map((b) => [b, 1 / binIds.length]));
}

const H = (ps) => -ps.reduce((acc, p) => (p > 0 ? acc + p * Math.log2(p) : acc), 0);

function eig(prior, lik, binIds) {
  const hs = Object.keys(prior);
  const pbar = binIds.map((b) => hs.reduce((acc, h) => acc + prior[h] * lik[h][b], 0));
  const cond = hs.reduce((acc, h) => acc + prior[h] * H(binIds.map((b) => lik[h][b])), 0);
  return H(pbar) - cond;
}

function update(prior, lik, bin) {
  const un = {};
  let z = 0;
  for (const h of Object.keys(prior)) {
    un[h] = prior[h] * lik[h][bin];
    z += un[h];
  }
  return Object.fromEntries(Object.entries(un).map(([h, v]) => [h, r4(v / z)]));
}

const score = (impact, e, costS) => impact * e / (1 + 0.2 * costS / 60);

/* ------------------------------------------------------------------ */
/* Fixture 1: BixBench3 task 9 (categorical)                           */
/* ------------------------------------------------------------------ */

function categorical() {
  const R = new Recorder('r-20261114-153012-a1b2', 1794700212.0, 1.3);
  const art = (name) => `runs/r-20261114-153012-a1b2/artifacts/${name}`;

  R.emit(
    'run.started',
    {
      lab: 'bixbench3',
      goal: 'BixBench3 task 9: differential expression of REST knockout vs control in T98G and HEK293 from a 16-sample count matrix',
      config: {
        ablations: { memory: true, search: true, critic: true, eig_ranking: true, fork_library: true },
        models: {
          proposer: 'gpt-4.1-mini',
          predictor: 'gpt-4.1-mini',
          designer: 'gpt-5.5',
          referee: 'gpt-5.5',
          critic: 'gpt-5.5',
          reporter: 'gpt-5.5',
        },
        budget_usd: 2.0,
        max_experiments_per_question: 4,
        seed: 7,
      },
      started_at: '2026-11-14T15:30:12Z',
    },
    0,
  );
  R.tick(0.3);

  const profile = [
    ['ev-prof-1', 'Count matrix: 58,302 genes x 16 samples, integer counts, no missing values'],
    ['ev-prof-2', 'Sample sheet: T98G Ctrl, C10, F7, G2, D4 and HEK293 Ctrl, D10, E6, two replicates each'],
    ['ev-prof-3', 'Library sizes 18.2M to 31.7M reads; no sample below 60% of the median'],
    ['ev-prof-4', 'In T98G, REST itself reads 4.1x higher in knockout-labelled samples than in Ctrl'],
  ].map(([id, claim]) => ({
    id,
    kind: 'profile',
    claim,
    author_role: 'profiler',
    source_ref: 'profile.py',
    url: null,
    conditions: { input_hash: sha('counts.tsv').slice(0, 12) },
  }));
  R.llm(1, 0.0021, 2400, 380);
  R.emit('profile.done', { evidence: profile }, 2.2);
  R.emit('memory.written', { evidence_id: 'ev-prof-2', author_role: 'profiler', backend: 'past.dev' }, 0.4);
  R.emit('memory.written', { evidence_id: 'ev-prof-4', author_role: 'profiler', backend: 'past.dev' }, 0.3);
  R.tick(0.2);

  R.emit(
    'memory.recalled',
    {
      reader_role: 'proposer-b',
      query: 'REST expression in T98G samples',
      evidence_ids: ['ev-prof-4', 'ev-prof-2'],
      authors: ['profiler', 'profiler'],
      backend: 'past.dev',
      latency_ms: 71.4,
    },
    0.9,
  );

  // ---- Questions and options -------------------------------------
  const hyp = (h) => ({
    answer_key: null,
    model_code: null,
    is_null: false,
    status: 'active',
    rejected_by: [],
    posterior: h.prior,
    ...h,
  });
  const nullOpt = (qid, prior = 0.15) =>
    hyp({
      id: 'h0',
      question_id: qid,
      statement: 'None of these',
      author: 'engine',
      rationale: 'Reserved belief for an explanation nobody proposed.',
      is_null: true,
      prior,
    });

  const questions = {
    q1: {
      id: 'q1',
      text: 'Are the T98G control and knockout labels as given, or inverted?',
      kind: 'categorical',
      impact: 0.9,
      status: 'open',
      raised_by: 'proposer-b',
    },
    q2: {
      id: 'q2',
      text: 'Should the model be fit per cell line or jointly?',
      kind: 'categorical',
      impact: 0.6,
      status: 'open',
      raised_by: 'proposer-c',
    },
    q3: {
      id: 'q3',
      text: 'Pre-filter low-count genes before DESeq2?',
      kind: 'categorical',
      impact: 0.3,
      status: 'open',
      raised_by: 'fork-library',
    },
  };

  R.llm(3, 0.0062, 9100, 1650);
  R.emit('question.opened', { question: questions.q1 }, 1.8);
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: hyp({
        id: 'h1',
        question_id: 'q1',
        statement: 'Labels are as given',
        author: 'proposer-a',
        rationale: 'The sample sheet is the only label source; CRISPR knockouts can still express REST mRNA.',
        answer_key: 'labels_as_given',
        prior: 0.2833,
      }),
    },
    0.7,
  );
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: hyp({
        id: 'h2',
        question_id: 'q1',
        statement: 'Control and knockout are inverted in T98G',
        author: 'proposer-b',
        rationale: 'REST reads 4.1x higher in the knockout-labelled samples, the opposite of a knockout.',
        answer_key: 'labels_inverted',
        prior: 0.2833,
      }),
    },
    0.7,
  );
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: hyp({
        id: 'h3',
        question_id: 'q1',
        statement: 'T98G Ctrl and knockout columns are swapped',
        author: 'proposer-c',
        rationale: 'The REST level points to a column swap in the T98G block.',
        answer_key: 'labels_inverted',
        prior: 0.2833,
      }),
    },
    0.7,
  );
  R.emit('hypothesis.proposed', { hypothesis: nullOpt('q1') }, 0.4);

  let prior = { h1: 0.425, h2: 0.425, h0: 0.15 };
  R.llm(1, 0.0018, 2100, 240);
  R.emit('hypothesis.merged', { question_id: 'q1', kept: 'h2', merged: ['h3'], priors: prior }, 1.1);
  R.tick(0.2);

  R.emit(
    'memory.recalled',
    {
      reader_role: 'proposer-c',
      query: 'library sizes and dispersion by cell line',
      evidence_ids: ['ev-prof-2'],
      authors: ['profiler'],
      backend: 'past.dev',
      latency_ms: 66.1,
    },
    0.8,
  );
  R.llm(2, 0.0041, 6400, 1100);
  R.emit('question.opened', { question: questions.q2 }, 1.4);
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: hyp({
        id: 'h1',
        question_id: 'q2',
        statement: 'Fit DESeq2 separately per cell line',
        author: 'proposer-c',
        rationale: 'T98G and HEK293 may differ in dispersion; a joint fit would borrow the wrong variance.',
        answer_key: 'per_line',
        prior: 0.425,
      }),
    },
    0.7,
  );
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: hyp({
        id: 'h2',
        question_id: 'q2',
        statement: 'Fit one joint model: ~ cell_line + condition',
        author: 'proposer-a',
        rationale: 'Sixteen samples give steadier dispersion estimates than ten or six.',
        answer_key: 'joint',
        prior: 0.425,
      }),
    },
    0.7,
  );
  R.emit('hypothesis.proposed', { hypothesis: nullOpt('q2') }, 0.4);

  R.emit('question.opened', { question: questions.q3 }, 0.8);
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: hyp({
        id: 'h1',
        question_id: 'q3',
        statement: 'Keep genes with 10+ reads in 2+ samples',
        author: 'fork-library',
        rationale: 'A common default that speeds fitting and trims noise.',
        answer_key: 'prefilter_10',
        prior: 0.425,
      }),
    },
    0.5,
  );
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: hyp({
        id: 'h2',
        question_id: 'q3',
        statement: 'No pre-filter; rely on independent filtering',
        author: 'fork-library',
        rationale: 'DESeq2 filters low-count genes when it adjusts p-values.',
        answer_key: 'no_prefilter',
        prior: 0.425,
      }),
    },
    0.5,
  );
  R.emit('hypothesis.proposed', { hypothesis: nullOpt('q3') }, 0.4);
  R.tick(0.2);

  // ---- Experiments -------------------------------------------------
  const experiments = {};
  function propose(spec, dt = 0.6) {
    const binIds = spec.bins.map((b) => b.id);
    const likelihood = {};
    for (const [h, counts] of Object.entries(spec.counts)) likelihood[h] = likelihoodFromCounts(counts, binIds);
    likelihood.h0 = uniform(binIds);
    const exp = {
      id: spec.id,
      question_ids: [spec.q],
      kind: spec.kind,
      goal: spec.goal,
      language: spec.language ?? null,
      code: spec.code ?? null,
      query: spec.query ?? null,
      design: null,
      bins: spec.bins,
      likelihood: Object.fromEntries(
        Object.entries(likelihood).map(([h, m]) => [h, Object.fromEntries(Object.entries(m).map(([b, p]) => [b, r4(p)]))]),
      ),
      likelihood_source: 'sampled',
      est_cost_s: spec.costS,
      est_cost_usd: spec.costUsd,
      eig_bits: null,
      score: null,
      spec_hash: sha(spec.id + JSON.stringify(spec.bins)),
      status: 'proposed',
    };
    experiments[spec.id] = { ...exp, _lik: likelihood, _bins: binIds, _impact: questions[spec.q].impact };
    R.emit('experiment.proposed', { experiment: exp }, dt);
  }

  function rank(qid, ids, priorMap, expectChosen, dt = 0.8) {
    const ranking = ids
      .map((id) => {
        const e = experiments[id];
        const bits = eig(priorMap, e._lik, e._bins);
        return {
          experiment_id: id,
          kind: e.kind,
          goal: e.goal,
          eig_bits: r4(bits),
          est_cost_s: e.est_cost_s,
          est_cost_usd: e.est_cost_usd,
          score: r4(score(e._impact, bits, e.est_cost_s)),
        };
      })
      .sort((a, b) => b.score - a.score);
    const chosen = ranking[0].experiment_id;
    if (chosen !== expectChosen) {
      throw new Error(`ranking for ${qid} chose ${chosen}, expected ${expectChosen}: ${JSON.stringify(ranking)}`);
    }
    R.emit('experiment.scored', { question_id: qid, ranking, chosen }, dt);
    return ranking;
  }

  function run(id, logs, value, bin, seconds, costUsd, verdictNote, dtStart = 0.6) {
    const e = experiments[id];
    R.emit('experiment.started', { experiment_id: id, question_ids: e.question_ids, kind: e.kind }, dtStart);
    for (const [stream, line, dt] of logs) {
      R.emit('experiment.log', { experiment_id: id, stream, line }, dt ?? 0.45);
    }
    R.cost += costUsd;
    R.emit(
      'experiment.finished',
      {
        observation: {
          experiment_id: id,
          valid: true,
          value,
          bin_id: bin,
          surprise: 0,
          referee_note: '',
          log_ref: `runs/r-20261114-153012-a1b2/logs/${id}.log`,
          seconds,
          cost_usd: costUsd,
        },
      },
      0.5,
    );
    R.llm(1, 0.0094, 5200, 310);
    R.emit('referee.verdict', { experiment_id: id, valid: true, note: verdictNote }, 1.2);
  }

  function believe(qid, id, bin, priorMap) {
    const e = experiments[id];
    const posterior = update(priorMap, e._lik, bin);
    R.emit(
      'belief.updated',
      { question_id: qid, experiment_id: id, observed: bin, prior: priorMap, posterior },
      0.5,
    );
    return posterior;
  }

  R.llm(4, 0.0213, 15800, 4200);
  propose({
    id: 'rest_targets',
    q: 'q1',
    kind: 'compute',
    language: 'r',
    goal: 'Are canonical REST targets (SYP, SNAP25, CHGA, BDNF, STMN2) higher in knockout-labelled T98G samples?',
    code: 'source("diagnostics/de_from_counts/rest_targets.R")',
    bins: [
      { id: 'up_in_ko', description: 'REST targets higher in knockout-labelled samples', rule: 'median_log2fc_targets > 0.5' },
      { id: 'up_in_ctrl', description: 'REST targets higher in Ctrl-labelled samples', rule: 'median_log2fc_targets < -0.5' },
      { id: 'no_change', description: 'No clear shift in REST targets', rule: 'abs(median_log2fc_targets) <= 0.5' },
    ],
    counts: { h1: [7, 0, 1], h2: [0, 7, 1] },
    costS: 18,
    costUsd: 0.003,
  }, 1.6);
  propose({
    id: 'sample_corr',
    q: 'q1',
    kind: 'compute',
    language: 'python',
    goal: 'On 412 REST-repressed genes, do the T98G Ctrl replicates cluster with true controls or with the knockouts?',
    code: 'python diagnostics/de_from_counts/sample_corr.py --genes rest_repressed.txt',
    bins: [
      { id: 'with_controls', description: 'Ctrl replicates cluster with the HEK293 controls', rule: "ctrl_cluster == 'controls'" },
      { id: 'with_knockouts', description: 'Ctrl replicates cluster with the knockouts', rule: "ctrl_cluster == 'knockouts'" },
      { id: 'mixed', description: 'No clean clustering', rule: "ctrl_cluster == 'mixed'" },
    ],
    counts: { h1: [7, 0, 1], h2: [0, 7, 1] },
    costS: 30,
    costUsd: 0.004,
  }, 0.5);
  propose({
    id: 'multiverse_labels',
    q: 'q1',
    kind: 'multiverse',
    language: 'r',
    goal: 'Run DE with labels as given and inverted; which one recovers the REST-repressed gene set?',
    code: 'source("diagnostics/de_from_counts/multiverse_labels.R")',
    bins: [
      { id: 'given_recovers', description: 'Labels as given recover the gene set', rule: 'auc_given > auc_inverted + 0.1' },
      { id: 'inverted_recovers', description: 'Inverted labels recover the gene set', rule: 'auc_inverted > auc_given + 0.1' },
      { id: 'neither', description: 'Neither orientation recovers it', rule: 'max(auc_given, auc_inverted) < 0.6' },
    ],
    counts: { h1: [6, 1, 1], h2: [1, 6, 1] },
    costS: 300,
    costUsd: 0.09,
  }, 0.5);
  propose({
    id: 'filter_sweep',
    q: 'q3',
    kind: 'compute',
    language: 'r',
    goal: 'Does a 10-read pre-filter change the top 500 genes by padj?',
    code: 'source("diagnostics/de_from_counts/filter_sweep.R")',
    bins: [
      { id: 'changes', description: 'Top 500 changes by more than 5%', rule: 'jaccard_top500 < 0.95' },
      { id: 'same', description: 'Top 500 essentially unchanged', rule: 'jaccard_top500 >= 0.95' },
    ],
    counts: { h1: [4, 4], h2: [3, 5] },
    costS: 40,
    costUsd: 0.005,
  }, 0.5);
  R.tick(0.2);

  // q3 closes before any spend: no check is worth running.
  const q3prior = { h1: 0.425, h2: 0.425, h0: 0.15 };
  const fsBits = eig(q3prior, experiments.filter_sweep._lik, experiments.filter_sweep._bins);
  R.emit(
    'question.unresolved',
    {
      question_id: 'q3',
      reason: `Best check is worth ${fsBits.toFixed(3)} bits, under the 0.05 floor; both choices are reported in the methods.`,
      top: ['h2', 'h1'],
    },
    0.9,
  );

  // ---- Q1, round 1 -----------------------------------------------
  rank('q1', ['rest_targets', 'sample_corr', 'multiverse_labels'], prior, 'rest_targets');
  R.tick(0.2);
  run(
    'rest_targets',
    [
      ['stdout', '$ Rscript diagnostics/de_from_counts/rest_targets.R --counts counts.tsv --samples samples.tsv', 0.5],
      ['stdout', 'loaded 58302 genes x 16 samples', 0.6],
      ['stdout', 'size factors (median of ratios): 0.71 .. 1.24', 0.5],
      ['stdout', 'T98G: Ctrl (2) vs knockout clones C10, F7, G2, D4 (8)', 0.5],
      ['stdout', 'REST targets present: SYP SNAP25 CHGA BDNF STMN2 SCG3 VGF (7/7)', 0.6],
      ['stdout', 'median log2FC on targets (knockout vs Ctrl): -1.62', 0.7],
      ['stdout', 'REST log2FC (knockout vs Ctrl): +2.04', 0.4],
      ['stdout', 'RESULT: {"median_log2fc_targets": -1.62, "n_targets": 7, "rest_log2fc": 2.04}', 0.4],
    ],
    { median_log2fc_targets: -1.62, n_targets: 7, rest_log2fc: 2.04 },
    'up_in_ctrl',
    3.8,
    0.0031,
    'Ran on the staged matrix; spec hash matches; 7 of 7 targets present; bin computed from the RESULT line.',
  );
  prior = believe('q1', 'rest_targets', 'up_in_ctrl', prior);
  R.emit('memory.written', { evidence_id: 'ev-obs-rest_targets', author_role: 'experimenter', backend: 'past.dev' }, 0.4);
  R.tick(0.2);

  // ---- Q1, round 2 -----------------------------------------------
  R.llm(1, 0.0088, 7300, 520);
  rank('q1', ['sample_corr', 'multiverse_labels'], prior, 'sample_corr', 1.4);
  run(
    'sample_corr',
    [
      ['stdout', '$ python diagnostics/de_from_counts/sample_corr.py --genes rest_repressed.txt', 0.5],
      ['stdout', 'vst on 16 samples; 412 REST-repressed genes', 0.6],
      ['stdout', 'average-linkage clustering on 1 - Pearson r', 0.5],
      ['stderr', 'UserWarning: 37 genes with zero variance dropped before clustering', 0.3],
      ['stdout', 'cluster A: T98G Ctrl_1 Ctrl_2 F7_1 F7_2 G2_1 G2_2 D4_1 D4_2', 0.6],
      ['stdout', 'cluster B: T98G C10_1 C10_2 + HEK293 Ctrl_1 Ctrl_2', 0.5],
      ['stdout', 'Ctrl_1, Ctrl_2 join the knockout cluster at height 0.03; C10 joins the controls at 0.06', 0.5],
      ['stdout', 'RESULT: {"ctrl_cluster": "knockouts", "r_ctrl_to_ko": 0.97, "r_ctrl_to_ref_ctrl": 0.41}', 0.4],
    ],
    { ctrl_cluster: 'knockouts', r_ctrl_to_ko: 0.97, r_ctrl_to_ref_ctrl: 0.41 },
    'with_knockouts',
    2.6,
    0.0024,
    'Clustering used only the agent-visible matrix; 412 of 412 genes found; result line parsed by code.',
  );
  prior = believe('q1', 'sample_corr', 'with_knockouts', prior);
  R.emit(
    'hypothesis.rejected',
    {
      question_id: 'q1',
      hypothesis_id: 'h1',
      evidence_id: 'ev-obs-sample_corr',
      reason: 'REST targets read 3.1x higher in Ctrl-labelled samples, and both Ctrl replicates cluster with the knockouts (r = 0.97).',
    },
    0.6,
  );
  R.emit('question.settled', { question_id: 'q1', hypothesis_id: 'h2', posterior: prior.h2 }, 0.6);
  R.emit('memory.written', { evidence_id: 'ev-obs-sample_corr', author_role: 'experimenter', backend: 'past.dev' }, 0.4);
  R.tick(0.2);

  // ---- Q2 ------------------------------------------------------------
  R.emit(
    'memory.recalled',
    {
      reader_role: 'designer',
      query: 'T98G labels and dispersion across cell lines',
      evidence_ids: ['ev-obs-rest_targets', 'ev-obs-sample_corr', 'ev-prof-3'],
      authors: ['experimenter', 'experimenter', 'profiler'],
      backend: 'past.dev',
      latency_ms: 88.2,
    },
    1.0,
  );
  R.llm(3, 0.0172, 13900, 3100);
  propose({
    id: 'docs_deseq2',
    q: 'q2',
    kind: 'search',
    goal: 'What do the DESeq2 docs recommend for two cell lines with different dispersion?',
    query: 'DESeq2 multiple cell lines one joint model or subset per cell line dispersion',
    bins: [
      { id: 'docs_joint', description: 'Docs recommend one joint model with a cell-line term', rule: "stance == 'joint'" },
      { id: 'docs_per_line', description: 'Docs recommend subsetting per cell line', rule: "stance == 'per_line'" },
      { id: 'docs_either', description: 'Docs say either works; check the dispersion', rule: "stance == 'either'" },
    ],
    counts: { h1: [1, 6, 1], h2: [6, 1, 1] },
    costS: 6,
    costUsd: 0.002,
  }, 1.5);
  propose({
    id: 'multiverse_fit',
    q: 'q2',
    kind: 'multiverse',
    language: 'r',
    goal: 'Fit T98G both ways and compare the padj < 0.05 gene sets',
    code: 'source("diagnostics/de_from_counts/multiverse_fit.R")',
    bins: [
      { id: 'concordant', description: 'Same genes either way (Jaccard 0.9+, no sign flips)', rule: 'jaccard >= 0.9 and sign_flips == 0' },
      { id: 'per_line_more', description: 'Per-line fit finds 20%+ more genes', rule: 'n_per_line > 1.2 * n_joint' },
      { id: 'joint_more', description: 'Joint fit finds 20%+ more genes', rule: 'n_joint > 1.2 * n_per_line' },
    ],
    counts: { h1: [1, 6, 1], h2: [6, 1, 1] },
    costS: 120,
    costUsd: 0.06,
  }, 0.5);
  propose({
    id: 'dispersion_trend',
    q: 'q2',
    kind: 'compute',
    language: 'r',
    goal: 'Do T98G and HEK293 have different mean-dispersion trends?',
    code: 'source("diagnostics/de_from_counts/dispersion_trend.R")',
    bins: [
      { id: 'differ', description: 'Trends differ by more than 2x', rule: 'max_ratio > 2' },
      { id: 'similar', description: 'Trends within 2x', rule: 'max_ratio <= 2' },
    ],
    counts: { h1: [6, 2], h2: [3, 5] },
    costS: 40,
    costUsd: 0.005,
  }, 0.5);
  R.tick(0.2);

  let prior2 = { h1: 0.425, h2: 0.425, h0: 0.15 };
  rank('q2', ['docs_deseq2', 'multiverse_fit', 'dispersion_trend'], prior2, 'docs_deseq2');
  R.cost += 0.0021;
  R.emit('experiment.started', { experiment_id: 'docs_deseq2', question_ids: ['q2'], kind: 'search' }, 0.6);
  R.emit('experiment.log', { experiment_id: 'docs_deseq2', stream: 'stdout', line: 'exa.search("DESeq2 multiple cell lines one joint model or subset per cell line dispersion", k=5)' }, 0.5);
  R.emit('experiment.log', { experiment_id: 'docs_deseq2', stream: 'stdout', line: '5 results; leakage policy removed 1 (benchmark repository)' }, 1.1);
  R.emit('experiment.log', { experiment_id: 'docs_deseq2', stream: 'stdout', line: 'reading bioconductor.org DESeq2 vignette: sections "Interactions" and "Frequently asked questions"' }, 0.5);
  R.emit('experiment.log', { experiment_id: 'docs_deseq2', stream: 'stdout', line: 'DESeq2 FAQ: subset when one group has much higher within-group variability; otherwise one model' }, 0.6);
  R.emit(
    'search.done',
    {
      experiment_id: 'docs_deseq2',
      query: 'DESeq2 multiple cell lines one joint model or subset per cell line dispersion',
      urls: [
        'https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html',
        'https://support.bioconductor.org/p/deseq2-subset-or-full-model/',
        'https://f1000research.com/articles/4-1070',
        'https://www.bioconductor.org/help/course-materials/rnaseq-gene-workflow.html',
      ],
      blocked: ['https://github.com/EdisonScientific/BixBench3/tree/main/tasks/task9'],
      bin_id: 'docs_either',
    },
    0.5,
  );
  R.emit(
    'experiment.finished',
    {
      observation: {
        experiment_id: 'docs_deseq2',
        valid: true,
        value: { stance: 'either', sources: 4 },
        bin_id: 'docs_either',
        surprise: 0,
        referee_note: '',
        log_ref: 'runs/r-20261114-153012-a1b2/logs/docs_deseq2.log',
        seconds: 2.1,
        cost_usd: 0.0021,
      },
    },
    0.4,
  );
  R.llm(1, 0.0081, 4800, 290);
  R.emit(
    'referee.verdict',
    { experiment_id: 'docs_deseq2', valid: true, note: 'Four allowed sources, one blocked by policy; stance read from the vignette and FAQ text.' },
    1.1,
  );
  prior2 = believe('q2', 'docs_deseq2', 'docs_either', prior2);
  R.emit('memory.written', { evidence_id: 'ev-search-docs_deseq2', author_role: 'searcher', backend: 'past.dev' }, 0.4);
  R.tick(0.2);

  R.llm(1, 0.0079, 6900, 480);
  rank('q2', ['multiverse_fit', 'dispersion_trend'], prior2, 'multiverse_fit', 1.3);
  run(
    'multiverse_fit',
    [
      ['stdout', '$ Rscript diagnostics/de_from_counts/multiverse_fit.R --labels corrected', 0.5],
      ['stdout', 'design A: per cell line, ~ condition (T98G, 10 samples)', 0.5],
      ['stdout', 'estimating size factors', 0.3],
      ['stdout', 'estimating dispersions', 0.4],
      ['stderr', 'Warning: some variables in design formula are characters, converting to factors', 0.3],
      ['stdout', 'fitting model and testing', 0.6],
      ['stdout', 'design B: joint, ~ cell_line + condition (16 samples)', 0.5],
      ['stdout', 'estimating dispersions', 0.4],
      ['stderr', "note: parametric dispersion trend fit poorly; using fitType='local'", 0.3],
      ['stdout', 'fitting model and testing', 0.6],
      ['stdout', 'padj < 0.05: per-line 2,184 genes, joint 2,262 genes, overlap 2,141', 0.5],
      ['stdout', 'Jaccard 0.93, sign flips 0', 0.3],
      ['stdout', 'RESULT: {"jaccard": 0.93, "sign_flips": 0, "n_per_line": 2184, "n_joint": 2262}', 0.4],
    ],
    { jaccard: 0.93, sign_flips: 0, n_per_line: 2184, n_joint: 2262 },
    'concordant',
    7.4,
    0.0512,
    'Both fits used the corrected labels and the same counts; warnings are benign; bin computed by code.',
  );
  prior2 = believe('q2', 'multiverse_fit', 'concordant', prior2);
  R.emit('question.robust', { question_id: 'q2', answer_key: 'deg_t98g' }, 0.7);
  R.emit('memory.written', { evidence_id: 'ev-obs-multiverse_fit', author_role: 'experimenter', backend: 'past.dev' }, 0.4);
  R.tick(0.2);

  // ---- Deliverables and the critic ---------------------------------
  R.llm(2, 0.0215, 11800, 2600);
  R.emit(
    'artifact.written',
    {
      artifact: {
        name: 'de_results_T98G.csv',
        path: art('de_results_T98G.csv'),
        sha256: sha('de_results_T98G.csv/v1'),
        contract_ok: false,
        findings: [],
      },
    },
    1.6,
  );
  R.emit(
    'artifact.written',
    {
      artifact: {
        name: 'methods.md',
        path: art('methods.md'),
        sha256: sha('methods.md/v1'),
        contract_ok: true,
        findings: [],
      },
    },
    0.6,
  );
  R.emit(
    'memory.recalled',
    {
      reader_role: 'critic',
      query: 'de_results contract columns and label orientation',
      evidence_ids: ['ev-obs-multiverse_fit', 'ev-obs-sample_corr'],
      authors: ['experimenter', 'experimenter'],
      backend: 'past.dev',
      latency_ms: 64.9,
    },
    0.8,
  );
  R.llm(1, 0.0142, 9400, 610);
  R.emit(
    'critic.finding',
    {
      finding: {
        id: 'cf-1',
        artifact: 'de_results_T98G.csv',
        severity: 'block',
        rule: 'required_columns',
        message: 'Missing column padj; the contract needs gene_id, baseMean, log2FoldChange, pvalue, padj.',
        reopens_question: null,
      },
    },
    1.4,
  );
  R.emit(
    'critic.finding',
    {
      finding: {
        id: 'cf-2',
        artifact: 'methods.md',
        severity: 'warn',
        rule: 'states_versions',
        message: 'Methods name DESeq2 but not its version.',
        reopens_question: null,
      },
    },
    0.6,
  );
  R.tick(0.2);
  R.llm(1, 0.0098, 6200, 900);
  R.emit(
    'artifact.written',
    {
      artifact: {
        name: 'de_results_T98G.csv',
        path: art('de_results_T98G.csv'),
        sha256: sha('de_results_T98G.csv/v2'),
        contract_ok: true,
        findings: [],
      },
    },
    1.8,
  );
  R.tick(0.3);

  R.emit(
    'memory.recalled',
    {
      reader_role: 'reporter',
      query: 'what settled each question and what it cost',
      evidence_ids: ['ev-obs-rest_targets', 'ev-obs-sample_corr', 'ev-search-docs_deseq2', 'ev-obs-multiverse_fit'],
      authors: ['experimenter', 'experimenter', 'searcher', 'experimenter'],
      backend: 'past.dev',
      latency_ms: 92.6,
    },
    0.9,
  );
  R.llm(1, 0.0064, 5100, 420);
  R.emit(
    'lesson.written',
    {
      lesson: {
        id: 'ev-lesson-1',
        kind: 'lesson',
        claim: 'Check REST-target direction before trusting control labels: a $0.003 script moved belief from 43% to 79%.',
        author_role: 'reporter',
        source_ref: 'ev-obs-rest_targets',
        url: null,
        conditions: {},
      },
    },
    1.2,
  );
  R.emit('memory.written', { evidence_id: 'ev-lesson-1', author_role: 'reporter', backend: 'past.dev' }, 0.4);
  R.tick(0.3);

  const seconds = r2(R.t + 0.6 - R.t0);
  R.emit(
    'run.finished',
    {
      report: {
        run_id: R.runId,
        artifacts: [
          { name: 'de_results_T98G.csv', path: art('de_results_T98G.csv'), sha256: sha('de_results_T98G.csv/v2'), contract_ok: true, findings: [] },
          { name: 'methods.md', path: art('methods.md'), sha256: sha('methods.md/v1'), contract_ok: true, findings: ['cf-2'] },
        ],
        methods_md: 'T98G labels corrected (Ctrl and knockout inverted; settled at 92%). DESeq2 per cell line and joint fits agree (Jaccard 0.93), so the question is robust. Pre-filtering left unresolved and reported both ways.',
        rejected: [
          {
            hypothesis: 'h1',
            question: 'q1',
            evidence_id: 'ev-obs-sample_corr',
            reason: 'REST targets read 3.1x higher in Ctrl-labelled samples, and both Ctrl replicates cluster with the knockouts (r = 0.97).',
          },
        ],
        unresolved: ['q3'],
        cost_usd: r4(R.cost),
        seconds,
      },
      status: 'ok',
    },
    0.6,
  );

  R.write('demo_categorical.jsonl');
  return R;
}

/* ------------------------------------------------------------------ */
/* Fixture 2: screen-to-camera bench (numeric, executable models)      */
/* ------------------------------------------------------------------ */

function numeric() {
  const R = new Recorder('r-20261114-161540-c3d4', 1794703940.0);
  const rand = rng(42);
  const SIGMA = 1.6;
  const truth = (x) => 4 + 247 * Math.pow(x / 255, 0.62);
  const GAIN = 0.8;
  const tinted = (x) => 3 + GAIN * 247 * Math.pow(x / 255, 0.62);

  const X = [];
  for (let x = 0; x <= 248; x += 8) X.push(x);
  X.push(255);

  // Family fits by grid search plus least squares on one linear parameter.
  const fitScale = (pts, f) => {
    const num = pts.reduce((a, p) => a + f(p.x) * p.y, 0);
    const den = pts.reduce((a, p) => a + f(p.x) * f(p.x), 0);
    return den > 0 ? num / den : 1;
  };
  const families = {
    h1: {
      name: 'linear',
      prior: (x) => x,
      fit(pts) {
        if (pts.length < 2) {
          const a = pts.length ? fitScale(pts, (x) => x) : 1;
          return (x) => a * x;
        }
        const n = pts.length;
        const mx = pts.reduce((a, p) => a + p.x, 0) / n;
        const my = pts.reduce((a, p) => a + p.y, 0) / n;
        const b = pts.reduce((a, p) => a + (p.x - mx) * (p.y - my), 0) / pts.reduce((a, p) => a + (p.x - mx) ** 2, 0);
        return (x) => my + b * (x - mx);
      },
    },
    h2: {
      name: 'power',
      prior: (x) => 255 * Math.pow(x / 255, 0.8),
      fit(pts) {
        let best = null;
        for (let g = 0.3; g <= 1.5; g += 0.01) {
          const f = (x) => Math.pow(x / 255, g);
          const A = fitScale(pts, f);
          const sse = pts.reduce((a, p) => a + (A * f(p.x) - p.y) ** 2, 0);
          if (!best || sse < best.sse) best = { sse, A, g };
        }
        return (x) => best.A * Math.pow(x / 255, best.g);
      },
    },
    h3: {
      name: 'log',
      prior: (x) => 127.6 * Math.log(1 + x / 40),
      fit(pts) {
        let best = null;
        for (let k = 2; k <= 600; k += 2) {
          const f = (x) => Math.log(1 + x / k);
          const a = fitScale(pts, f);
          const sse = pts.reduce((acc, p) => acc + (a * f(p.x) - p.y) ** 2, 0);
          if (!best || sse < best.sse) best = { sse, a, k };
        }
        return (x) => best.a * Math.log(1 + x / best.k);
      },
    },
    h4: {
      name: 'saturating',
      prior: (x) => 355 * x / (100 + x),
      fit(pts) {
        let best = null;
        for (let k = 5; k <= 1500; k += 5) {
          const f = (x) => x / (k + x);
          const m = fitScale(pts, f);
          const sse = pts.reduce((a, p) => a + (m * f(p.x) - p.y) ** 2, 0);
          if (!best || sse < best.sse) best = { sse, m, k };
        }
        return (x) => best.m * x / (best.k + x);
      },
    },
  };

  const curveFor = (fn, pts, spread) =>
    X.map((x) => {
      const d = pts.length ? Math.min(...pts.map((p) => Math.abs(p.x - x))) : 128;
      const misfit = pts.length
        ? Math.sqrt(pts.reduce((a, p) => a + (fn(p.x) - p.y) ** 2, 0) / pts.length)
        : 0;
      const g = 0.25 + 0.75 * Math.min(1, d / 64) ** 2;
      return { mean: fn(x), sd: Math.sqrt(misfit ** 2 * 0.6 + (spread / Math.sqrt(pts.length + 1)) ** 2 * g + SIGMA ** 2) };
    });

  function curvesPayload(pts, weights, fitPts, extra = {}) {
    const fams = {};
    for (const [id, fam] of Object.entries(families)) {
      const fn = fitPts.length ? fam.fit(fitPts) : fam.prior;
      const c = curveFor(fn, fitPts, 18);
      fams[id] = { mean: c.map((v) => r2(v.mean)), sd: c.map((v) => r2(v.sd)), weight: weights[id] };
    }
    for (const [id, spec] of Object.entries(extra)) {
      const c = curveFor(spec.fn, spec.pts, 24);
      fams[id] = { mean: c.map((v) => r2(v.mean)), sd: c.map((v) => r2(v.sd)), weight: weights[id] };
    }
    fams.h0 = {
      mean: X.map((x) => r2(x * 0.9 + 12)),
      sd: X.map(() => 60),
      weight: weights.h0,
    };
    return { question_id: 'q1', x: X, families: fams, points: pts.map((p) => ({ x: p.x, y: p.y })) };
  }

  // Expected information gain proxy from the current predictive spread.
  const CANDIDATES = [4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 255];
  function rankDesigns(weights, fitPts, exclude, extra = {}) {
    const fns = Object.fromEntries(
      Object.entries(families).map(([id, fam]) => [id, fitPts.length ? fam.fit(fitPts) : fam.prior]),
    );
    for (const [id, spec] of Object.entries(extra)) fns[id] = spec.fn;
    const rows = CANDIDATES.filter((x) => !exclude.includes(x)).map((x) => {
      const ids = Object.keys(fns);
      const wsum = ids.reduce((a, id) => a + weights[id], 0);
      const mbar = ids.reduce((a, id) => a + weights[id] * fns[id](x), 0) / wsum;
      const between = ids.reduce((a, id) => a + weights[id] * (fns[id](x) - mbar) ** 2, 0) / wsum;
      const bits = 0.5 * Math.log2(1 + between / (SIGMA ** 2 + 144));
      return {
        experiment_id: `x=${x}`,
        kind: 'measure',
        goal: `Show grey level ${x} and measure the patch`,
        eig_bits: r4(bits),
        est_cost_s: 1.4,
        est_cost_usd: 0,
        score: r4(bits / (1 + 0.2 * 1.4 / 60)),
      };
    });
    return rows.sort((a, b) => b.score - a.score);
  }

  R.emit(
    'run.started',
    {
      lab: 'bench',
      goal: 'Find the model that maps a displayed grey level to the camera reading, in as few measurements as possible',
      config: {
        ablations: { memory: true, search: false, critic: false, eig_ranking: true },
        models: { proposer: 'gpt-4.1-mini', designer: 'gpt-5.5', referee: 'gpt-5.5', reporter: 'gpt-5.5' },
        settle_threshold: 0.95,
        sigma: SIGMA,
        n_particles: 500,
        seed: 3,
      },
      started_at: '2026-11-14T16:15:40Z',
    },
    0,
  );
  R.tick(0.3);
  R.llm(1, 0.0012, 1300, 210);
  R.emit(
    'profile.done',
    {
      evidence: [
        {
          id: 'ev-prof-1',
          kind: 'profile',
          claim: 'Camera 1280x720, exposure and white balance locked; black level reads 4.1 at grey 0',
          author_role: 'profiler',
          source_ref: 'GET /health',
          url: null,
          conditions: { environment: 'jetson-orin-nano' },
        },
      ],
    },
    1.2,
  );
  R.emit(
    'memory.recalled',
    {
      reader_role: 'proposer-a',
      query: 'screen to camera response models',
      evidence_ids: ['ev-lesson-0412'],
      authors: ['reporter'],
      backend: 'past.dev',
      latency_ms: 58.3,
    },
    0.7,
  );
  R.llm(4, 0.0094, 7600, 2300);
  R.emit(
    'question.opened',
    {
      question: {
        id: 'q1',
        text: 'Which model maps the screen grey level to the camera reading?',
        kind: 'model',
        impact: 1.0,
        status: 'open',
        raised_by: 'proposer-a',
      },
    },
    1.6,
  );

  const fam = (id, statement, author, rationale, code) => ({
    hypothesis: {
      id,
      question_id: 'q1',
      statement,
      author,
      rationale,
      answer_key: null,
      model_code: code,
      is_null: false,
      prior: 0.2125,
      posterior: 0.2125,
      status: 'active',
      rejected_by: [],
    },
  });
  R.emit('hypothesis.proposed', fam('h1', 'Linear: y = a + b·x', 'proposer-a', 'Cameras in linear mode report light proportionally.', 'def predict(x, p):\n    return p["a"] + p["b"] * x'), 0.5);
  R.emit('hypothesis.proposed', fam('h2', 'Power law: y = A·(x/255)^γ', 'proposer-b', 'Display gamma and camera gamma compose into one power law.', 'def predict(x, p):\n    return p["A"] * (x / 255) ** p["g"]'), 0.5);
  R.emit('hypothesis.proposed', fam('h3', 'Logarithmic: y = a·log(1 + x/k)', 'proposer-c', 'Auto-tone curves often compress highlights logarithmically.', 'def predict(x, p):\n    return p["a"] * math.log(1 + x / p["k"])'), 0.5);
  R.emit('hypothesis.proposed', fam('h4', 'Saturating: y = m·x / (k + x)', 'proposer-a', 'Sensor wells saturate near the top of the range.', 'def predict(x, p):\n    return p["m"] * x / (p["k"] + x)'), 0.5);
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: {
        id: 'h0',
        question_id: 'q1',
        statement: 'None of these',
        author: 'engine',
        rationale: 'A broad family that catches anything the proposals miss.',
        answer_key: null,
        model_code: null,
        is_null: true,
        prior: 0.15,
        posterior: 0.15,
        status: 'active',
        rejected_by: [],
      },
    },
    0.4,
  );
  R.tick(0.2);

  // Posterior trajectory over the eight measurements.
  const trajectory = [
    { h1: 0.1, h2: 0.31, h3: 0.25, h4: 0.27, h0: 0.07 },
    { h1: 0.03, h2: 0.39, h3: 0.27, h4: 0.26, h0: 0.05 },
    { h1: 0.012, h2: 0.478, h3: 0.27, h4: 0.2, h0: 0.04 },
    { h1: 0.006, h2: 0.584, h3: 0.22, h4: 0.16, h0: 0.03 },
    { h1: 0.004, h2: 0.706, h3: 0.14, h4: 0.13, h0: 0.02 },
    { h1: 0.003, h2: 0.822, h3: 0.07, h4: 0.09, h0: 0.015 },
    { h1: 0.002, h2: 0.911, h3: 0.018, h4: 0.058, h0: 0.011 },
    { h1: 0.001, h2: 0.953, h3: 0.012, h4: 0.028, h0: 0.006 },
  ];

  let weights = { h1: 0.2125, h2: 0.2125, h3: 0.2125, h4: 0.2125, h0: 0.15 };
  const pts = [];
  const rejectedBy = {};
  for (let i = 0; i < trajectory.length; i++) {
    const ranking = rankDesigns(weights, pts, pts.map((p) => p.x));
    const chosen = ranking[0].experiment_id;
    const x = Number(chosen.slice(2));
    R.llm(1, 0.0031, 2900, 160);
    R.emit('experiment.scored', { question_id: 'q1', ranking, chosen }, i === 0 ? 1.2 : 0.9);
    R.emit('experiment.started', { experiment_id: chosen, question_ids: ['q1'], kind: 'measure' }, 0.5);
    const y = r2(truth(x) + SIGMA * gauss(rand));
    const sd = r2(1.4 + rand() * 0.5);
    R.emit(
      'experiment.log',
      { experiment_id: chosen, stream: 'stdout', line: `patch level=${x} shown; 8 frames, centre 80x80: mean ${y.toFixed(2)} std ${sd.toFixed(2)}` },
      1.2,
    );
    R.emit(
      'experiment.finished',
      {
        observation: {
          experiment_id: chosen,
          valid: true,
          value: y,
          bin_id: null,
          surprise: r3(0.4 + rand() * 0.8),
          referee_note: '',
          log_ref: '',
          seconds: 1.3,
          cost_usd: 0,
        },
      },
      0.4,
    );
    pts.push({ x, y });
    const posterior = trajectory[i];
    R.emit('belief.updated', { question_id: 'q1', experiment_id: chosen, observed: y, prior: weights, posterior }, 0.4);
    weights = posterior;
    R.emit('model.curves', curvesPayload(pts, weights, pts), 0.3);
    if (i === 2) {
      R.emit(
        'hypothesis.rejected',
        {
          question_id: 'q1',
          hypothesis_id: 'h1',
          evidence_id: (rejectedBy.h1 = `ev-obs-${chosen}`),
          reason: `Three readings bend away from any straight line; the best linear fit misses grey ${x} by ${(Math.abs(families.h1.fit(pts)(x) - y) / SIGMA).toFixed(0)} sd.`,
        },
        0.5,
      );
    }
    if (i === 6) {
      R.emit(
        'hypothesis.rejected',
        {
          question_id: 'q1',
          hypothesis_id: 'h3',
          evidence_id: (rejectedBy.h3 = `ev-obs-${chosen}`),
          reason: 'The log curve overshoots the dark end and undershoots mid-grey at the same time; posterior fell under 2%.',
        },
        0.5,
      );
    }
    if (i % 2 === 1) R.tick(0.2);
  }
  R.emit('question.settled', { question_id: 'q1', hypothesis_id: 'h2', posterior: weights.h2 }, 0.6);
  R.emit('memory.written', { evidence_id: 'ev-obs-q1-fit', author_role: 'experimenter', backend: 'past.dev' }, 0.4);
  R.tick(0.3);

  // Validation reading after the operator puts a tint over the screen.
  const vId = 'x=128#check';
  R.emit('experiment.started', { experiment_id: vId, question_ids: ['q1'], kind: 'measure' }, 1.8);
  R.emit('experiment.log', { experiment_id: vId, stream: 'stdout', line: 'validation pass: re-measuring grey 128 with the settled model' }, 0.6);
  const yv = r2(tinted(128) + SIGMA * gauss(rand));
  const predV = families.h2.fit(pts)(128);
  const predSd = curveFor(families.h2.fit(pts), pts, 18)[16].sd;
  const z = (yv - predV) / predSd;
  R.emit('experiment.log', { experiment_id: vId, stream: 'stdout', line: `patch level=128 shown; 8 frames, centre 80x80: mean ${yv.toFixed(2)} std 1.71` }, 1.2);
  R.emit('experiment.log', { experiment_id: vId, stream: 'stderr', line: `reading is ${Math.abs(z).toFixed(1)} predictive sd below the settled model` }, 0.4);
  R.emit(
    'experiment.finished',
    {
      observation: {
        experiment_id: vId,
        valid: true,
        value: yv,
        bin_id: null,
        surprise: r3(0.5 * z * z),
        referee_note: '',
        log_ref: '',
        seconds: 1.3,
        cost_usd: 0,
      },
    },
    0.4,
  );
  R.llm(1, 0.0088, 4100, 260);
  R.emit(
    'referee.verdict',
    { experiment_id: vId, valid: true, note: 'Exposure, gain and white balance unchanged since calibration; the reading is valid, so the scene changed.' },
    1.4,
  );
  let post = { h1: 0.0005, h2: 0.49, h3: 0.006, h4: 0.031, h0: 0.4725 };
  R.emit('belief.updated', { question_id: 'q1', experiment_id: vId, observed: yv, prior: weights, posterior: post }, 0.5);
  weights = post;
  R.emit(
    'surprise.raised',
    {
      question_id: 'q1',
      experiment_id: vId,
      reason: `Measured ${yv.toFixed(1)} at grey 128; the settled power law predicted ${predV.toFixed(1)} ± ${predSd.toFixed(1)} (z = ${z.toFixed(1)}). Something in the setup changed.`,
      z_score: r2(z),
      null_mass: post.h0,
    },
    0.6,
  );
  R.llm(3, 0.0152, 9800, 1900);
  R.emit(
    'hypothesis.proposed',
    {
      hypothesis: {
        id: 'h5',
        question_id: 'q1',
        statement: 'Tinted power law: y = c + s·A·(x/255)^γ',
        author: 'proposer-b',
        rationale: 'A neutral tint scales the light by s and lifts the black level; γ is unchanged. Explains every reading if s drops at the validation pass.',
        answer_key: null,
        model_code: 'def predict(x, p):\n    return p["c"] + p["s"] * p["A"] * (x / 255) ** p["g"]',
        is_null: false,
        prior: 0.2,
        posterior: 0.2,
        status: 'active',
        rejected_by: [],
      },
    },
    2.0,
  );
  post = { h1: 0.0004, h2: 0.27, h3: 0.004, h4: 0.018, h5: 0.58, h0: 0.1276 };
  R.emit('belief.updated', { question_id: 'q1', experiment_id: vId, observed: yv, prior: { ...weights, h5: 0.2 }, posterior: post }, 0.8);
  weights = post;
  pts.push({ x: 128, y: yv });
  const before = pts.slice(0, -1);
  const tintFn = (x) => tinted(x) + 0.6;
  R.emit('model.curves', curvesPayload(pts, weights, before, { h5: { fn: tintFn, pts: [pts.at(-1)] } }), 0.4);
  R.tick(0.2);

  const ranking = rankDesigns(weights, before, [128], { h5: { fn: tintFn } });
  const chosen = ranking[0].experiment_id;
  const xf = Number(chosen.slice(2));
  R.llm(1, 0.0033, 3000, 170);
  R.emit('experiment.scored', { question_id: 'q1', ranking, chosen }, 1.1);
  R.emit('experiment.started', { experiment_id: chosen, question_ids: ['q1'], kind: 'measure' }, 0.5);
  const yf = r2(tinted(xf) + SIGMA * gauss(rand));
  R.emit('experiment.log', { experiment_id: chosen, stream: 'stdout', line: `patch level=${xf} shown; 8 frames, centre 80x80: mean ${yf.toFixed(2)} std 1.66` }, 1.2);
  R.emit(
    'experiment.finished',
    {
      observation: { experiment_id: chosen, valid: true, value: yf, bin_id: null, surprise: 0.61, referee_note: '', log_ref: '', seconds: 1.3, cost_usd: 0 },
    },
    0.4,
  );
  post = { h1: 0.0001, h2: 0.019, h3: 0.0009, h4: 0.004, h5: 0.958, h0: 0.018 };
  R.emit('belief.updated', { question_id: 'q1', experiment_id: chosen, observed: yf, prior: weights, posterior: post }, 0.4);
  weights = post;
  pts.push({ x: xf, y: yf });
  R.emit('model.curves', curvesPayload(pts, weights, before, { h5: { fn: tintFn, pts: pts.slice(-2) } }), 0.3);
  R.emit('question.settled', { question_id: 'q1', hypothesis_id: 'h5', posterior: weights.h5 }, 0.6);
  R.tick(0.3);

  R.emit(
    'run.finished',
    {
      report: {
        run_id: R.runId,
        artifacts: [],
        methods_md: `Power law settled at 95% after 8 measurements. A validation reading at grey 128 was ${Math.abs(z).toFixed(0)} sd low; the regenerated tinted power law explains every reading and settled at 96%.`,
        rejected: [
          { hypothesis: 'h1', question: 'q1', evidence_id: rejectedBy.h1, reason: 'Readings bend away from any straight line.' },
          { hypothesis: 'h3', question: 'q1', evidence_id: rejectedBy.h3, reason: 'Log curve misfits both ends.' },
        ],
        unresolved: [],
        cost_usd: r4(R.cost),
        seconds: r2(R.t + 0.5 - R.t0),
      },
      status: 'ok',
    },
    0.5,
  );
  R.write('demo_numeric.jsonl');
  return R;
}

/* ------------------------------------------------------------------ */
/* Fixture 3: a run that fails (exercises run.error)                    */
/* ------------------------------------------------------------------ */

function failing() {
  const R = new Recorder('r-20261114-170205-e5f6', 1794708125.0);
  R.emit(
    'run.started',
    {
      lab: 'bixbench3',
      goal: 'BixBench3 task 9 on fault case 07 (library down-sampled)',
      config: { ablations: { memory: true, search: true, critic: true, eig_ranking: true }, budget_usd: 2.0, seed: 11 },
      started_at: '2026-11-14T17:02:05Z',
    },
    0,
  );
  R.tick(0.3);
  R.llm(1, 0.0019, 2300, 300);
  R.emit(
    'profile.done',
    {
      evidence: [
        { id: 'ev-prof-1', kind: 'profile', claim: 'Library sizes 0.4M to 30.9M reads; one sample is 50x below the median', author_role: 'profiler', source_ref: 'profile.py', url: null, conditions: {} },
      ],
    },
    1.8,
  );
  R.llm(2, 0.0044, 5100, 900);
  R.emit(
    'question.opened',
    { question: { id: 'q1', text: 'Keep or drop the shallow library HEK293 E6_2?', kind: 'categorical', impact: 0.7, status: 'open', raised_by: 'proposer-a' } },
    1.4,
  );
  const h = (id, statement, prior, isNull = false) => ({
    hypothesis: { id, question_id: 'q1', statement, author: isNull ? 'engine' : 'proposer-a', rationale: '', answer_key: null, model_code: null, is_null: isNull, prior, posterior: prior, status: 'active', rejected_by: [] },
  });
  R.emit('hypothesis.proposed', h('h1', 'Keep it; size factors absorb depth', 0.425), 0.5);
  R.emit('hypothesis.proposed', h('h2', 'Drop it; too shallow to estimate dispersion', 0.425), 0.5);
  R.emit('hypothesis.proposed', h('h0', 'None of these', 0.15, true), 0.4);
  R.emit(
    'experiment.scored',
    {
      question_id: 'q1',
      ranking: [
        { experiment_id: 'leave_one_out', kind: 'multiverse', goal: 'Fit with and without E6_2; compare the gene sets', eig_bits: 0.31, est_cost_s: 90, est_cost_usd: 0.05, score: 0.168 },
        { experiment_id: 'depth_check', kind: 'compute', goal: 'Is E6_2 an outlier on Cook distance?', eig_bits: 0.12, est_cost_s: 20, est_cost_usd: 0.003, score: 0.079 },
      ],
      chosen: 'leave_one_out',
    },
    1.2,
  );
  R.emit('experiment.started', { experiment_id: 'leave_one_out', question_ids: ['q1'], kind: 'multiverse' }, 0.6);
  R.emit('experiment.log', { experiment_id: 'leave_one_out', stream: 'stdout', line: '$ Rscript diagnostics/de_from_counts/leave_one_out.R' }, 0.5);
  R.emit('experiment.log', { experiment_id: 'leave_one_out', stream: 'stderr', line: 'Killed' }, 2.0);
  R.tick(0.2);
  R.emit('run.error', { message: 'Sandbox exited with code 137 (out of memory) while running leave_one_out', where: 'execution/sandbox.py' }, 0.4);
  R.write('demo_error.jsonl');
  return R;
}

mkdirSync(OUT, { recursive: true });
categorical();
numeric();
failing();
writeFileSync(
  resolve(OUT, 'index.json'),
  JSON.stringify(
    [
      { name: 'demo_categorical', title: 'BixBench3 task 9 (categorical)' },
      { name: 'demo_numeric', title: 'Screen-to-camera bench (numeric)' },
      { name: 'demo_error', title: 'Failed run (sandbox out of memory)' },
    ],
    null,
    2,
  ) + '\n',
);
