// The latest dna.verdict with its variant names (from dna.variant) and who
// wrote the sentence: the Jetson's local model or the template.

import type { ReactNode } from 'react';
import type { RunState } from '../../state/fold';
import { isNullVariant, verdictAuthor, verdictStanding } from '../../state/selectors';
import { cls, pct } from '../../lib/format';
import { Empty, Panel } from '../Panel';

function Field({ label, children, mono }: { label: string; children: ReactNode; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <div className="text-[0.66rem] font-semibold uppercase tracking-[0.16em] text-mute">{label}</div>
      <div className={cls('mt-[0.15rem] truncate text-[0.92rem] text-fg', mono && 'font-mono')}>{children}</div>
    </div>
  );
}

/** "HbS (sickle)" -> ["HbS", "sickle"]. */
export function splitName(name: string): [string, string | null] {
  const m = /^(.*?)\s*\(([^()]*)\)\s*$/.exec(name);
  return m && m[1] ? [m[1], m[2] ?? null] : [name, null];
}

const STANDING = {
  'new-strip': 'previous strip',
  surprise: 'reopened by surprise',
} as const;

export function VerdictCard({ state }: { state: RunState }) {
  const v = state.dna.verdict;
  if (!v) {
    return (
      <Panel title="Verdict">
        <Empty>Crux writes the verdict when the question settles.</Empty>
      </Panel>
    );
  }
  const variant = state.dna.variants[v.hypothesisId];
  const none = isNullVariant(state, v.hypothesisId);
  const author = verdictAuthor(v.author);
  const standing = verdictStanding(state);
  const superseded = standing !== 'current';
  const [name, gloss] = splitName(v.name);
  const latency =
    v.latencyMs === null ? 'n/a' : v.latencyMs >= 1000 ? `${(v.latencyMs / 1000).toFixed(2)} s` : `${Math.round(v.latencyMs)} ms`;
  const legacy = variant?.legacy || v.legacy;

  return (
    <Panel
      title="Verdict"
      meta={
        <span
          data-testid="verdict-author"
          className={cls(
            'inline-flex items-center rounded-[0.35rem] border px-[0.5rem] py-[0.12rem] text-[0.74rem] font-semibold',
            author.onJetson ? 'border-accent/60 bg-accent/10 text-accent' : 'border-line-strong text-dim',
          )}
        >
          {author.label}
        </span>
      }
      bodyClassName="flex flex-col gap-[0.6rem]"
    >
      <div key={v.seq} className={cls('flex flex-col gap-[0.6rem] animate-fade-up', superseded && 'opacity-50')} data-testid="verdict">
        <div className="flex min-w-0 items-baseline gap-[0.7rem]">
          <span className={cls('shrink-0 text-[2.6rem] font-bold leading-none', superseded ? 'text-dim line-through' : 'text-accent')}>
            {name}
          </span>
          {gloss && <span className="shrink-0 text-[1.1rem] text-dim">{gloss}</span>}
          <span className="tabular shrink-0 text-[1.5rem] font-bold text-fg">{pct(v.posterior)}</span>
          <span className="truncate text-[0.9rem] text-dim">
            {v.reads} {v.reads === 1 ? 'read' : 'reads'}
            {v.strip !== null && ` · strip ${v.strip}`}
          </span>
          {superseded && (
            <span className="ml-auto shrink-0 rounded-[0.3rem] border border-warn/60 px-[0.4rem] py-[0.05rem] text-[0.68rem] font-bold uppercase tracking-[0.12em] text-warn">
              {STANDING[standing]}
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-x-[1rem] gap-y-[0.45rem]">
          <Field label="HGVS" mono>
            {none ? 'none of these' : variant?.hgvs || 'n/a'}
          </Field>
          <Field label="Legacy">{none ? 'none of these' : legacy || 'n/a'}</Field>
          <Field label="dbSNP" mono>
            {variant?.rsid ? (
              <a
                href={`https://www.ncbi.nlm.nih.gov/snp/${encodeURIComponent(variant.rsid)}`}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-line-strong underline-offset-2 hover:text-accent"
              >
                {variant.rsid}
              </a>
            ) : none ? (
              'none of these'
            ) : (
              'none (reference)'
            )}
          </Field>
          <Field label="Codon 6" mono>
            {none ? 'none of these' : variant?.codon ? `${variant.codon} · ${variant.aminoAcid}` : 'n/a'}
          </Field>
        </div>
        {v.text && (
          <blockquote className="line-clamp-4 border-l-2 border-accent/60 pl-[0.7rem] text-[0.98rem] leading-snug text-fg">
            {v.text}
          </blockquote>
        )}
        <div className="flex items-center gap-[0.8rem] text-[0.8rem] text-mute">
          <span>latency {latency}</span>
          <span className="font-mono">PDB {v.pdbId || 'n/a'}</span>
        </div>
      </div>
    </Panel>
  );
}
