// The HBB reference in SeqViz (npm seqviz, MIT, Lattice Automation): the
// three strip windows, codon 6, the translation, the windows read so far and
// the engine's mismatch positions in red. The reads themselves are listed
// underneath, letter by letter.

import { Suspense, lazy, useEffect, useRef, useState, type RefObject } from 'react';
import type { DnaReadView, RunState } from '../../state/fold';
import { readMatches, sequenceMarks, variantName } from '../../state/selectors';
import { CODON6, HBB_ACCESSION, HBB_REF, WINDOWS } from '../../lib/hbb';
import { cls } from '../../lib/format';
import { Panel } from '../Panel';
import { ReadCells } from './bits';

// Lazy: other views never load SeqViz.
const SeqViz = lazy(() => import('seqviz').then((m) => ({ default: m.SeqViz })));

const WINDOW_COLORS = ['#8d99a6', '#8d99a6', '#c3ccd6'];
const CODON6_COLOR = '#9085e9';
const READ_COLOR = 'rgba(70, 224, 180, 0.22)';
const MISMATCH_COLOR = 'rgba(255, 107, 107, 0.85)';
const MISMATCH_TEXT = '#ff6b6b';

/** SeqViz props for the reference and the current marks. Pure, for tests. */
export function seqvizProps(state: RunState, zoom = 70) {
  const marks = sequenceMarks(state);
  const annotations = [
    ...WINDOWS.map((w, i) => ({
      name: w.label,
      start: w.start,
      end: w.end,
      direction: 1,
      color: WINDOW_COLORS[i] ?? '#8d99a6',
    })),
    { name: 'Glu6', start: CODON6.start, end: CODON6.end, direction: 1, color: CODON6_COLOR },
  ];
  const highlights = [
    ...marks.windowsRead
      .map((n) => WINDOWS.find((w) => w.window === n))
      .filter((w): w is (typeof WINDOWS)[number] => w !== undefined)
      .map((w) => ({ start: w.start, end: w.end, color: READ_COLOR })),
    ...marks.mismatches.map((i) => ({ start: i, end: i + 1, color: MISMATCH_COLOR })),
  ];
  const bpColors: Record<number, string> = {};
  for (const i of marks.mismatches) bpColors[i] = MISMATCH_TEXT;
  return {
    name: `HBB ${HBB_ACCESSION}`,
    seq: HBB_REF,
    seqType: 'dna' as const,
    viewer: 'linear' as const,
    annotations,
    translations: [{ name: 'beta-globin', start: 0, end: HBB_REF.length, direction: 1 }],
    highlights,
    bpColors,
    showComplement: false,
    showIndex: true,
    disableExternalFonts: true,
    zoom: { linear: zoom },
  };
}

function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

/**
 * SeqViz zoom that keeps all 27 nt on one row: at zoom > 70 SeqViz fits
 * round(width * 1.4 / 18 * 70 / zoom) bases per row (zoom is capped at 100).
 */
export function zoomForWidth(width: number, bases = HBB_REF.length): number {
  if (!(width > 0)) return 70;
  return Math.max(1, Math.min(100, Math.floor((width * 1.4 * 70) / (18 * (bases + 0.5)))));
}

function useWidth(): [RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

function Loading({ text }: { text: string }) {
  return <div className="flex h-full items-center justify-center text-[0.9rem] text-mute">{text}</div>;
}

export function SequencePanel({ state }: { state: RunState }) {
  const mounted = useMounted();
  const [box, width] = useWidth();
  const props = seqvizProps(state, zoomForWidth(width));
  const reads = state.dna.reads.slice(-3).reverse();
  const total = state.dna.reads.length;

  return (
    <Panel
      title={`HBB · ${HBB_ACCESSION}`}
      meta="first 27 nt of the coding sequence · codon 6 = nt 19-21"
      bodyClassName="flex flex-col gap-[0.55rem]"
    >
      <div ref={box} data-testid="seqviz" className="seqviz-dark h-[166px] shrink-0 overflow-hidden rounded-[0.6rem] border border-line bg-[#0a0e13]">
        {mounted && width > 0 ? (
          <Suspense fallback={<Loading text="Loading SeqViz…" />}>
            <SeqViz {...props} style={{ height: '100%', width: '100%' }} />
          </Suspense>
        ) : (
          <Loading text={HBB_REF.match(/.{3}/g)?.join(' ') ?? HBB_REF} />
        )}
      </div>
      <div className="flex items-center gap-[0.9rem] text-[0.72rem] text-mute">
        <Legend swatch="bg-[#9085e9]" text="codon 6 (Glu6)" />
        <Legend swatch="bg-accent/30" text="window read" />
        <Legend swatch="bg-warn/40" text="differs from reference" />
        <Legend swatch="bg-bad/70" text="mismatch vs closest option" />
        <span className="ml-auto font-semibold uppercase tracking-[0.14em]">{total} reads</span>
      </div>
      <ol className="flex min-h-0 flex-col gap-[0.4rem]" data-testid="reads">
        {reads.length === 0 && <li className="text-[0.9rem] text-mute">No reads yet.</li>}
        {reads.map((r, i) => {
          const n = total - i;
          const earlierStrip = r.seq <= state.dna.stripSince;
          return (
            <li
              key={r.seq}
              className={cls(
                'flex min-w-0 items-center gap-[0.7rem] rounded-[0.5rem] border px-[0.6rem] py-[0.35rem]',
                i === 0 ? 'border-line-strong bg-raised animate-fade-up' : 'border-line',
                i > 0 && (earlierStrip ? 'opacity-50' : 'opacity-80'),
                // Short screens (720p) keep the two newest reads.
                i > 1 && '[@media(max-height:800px)]:hidden',
              )}
              title={`${r.experimentId}${earlierStrip ? ' · earlier strip' : ''}`}
            >
              <span className="tabular w-[1.6rem] shrink-0 text-[0.8rem] font-bold text-mute">#{n}</span>
              <span className="shrink-0 font-mono text-[0.85rem] font-semibold text-dim">
                {r.strip !== null && <span className="text-mute">S{r.strip} </span>}W{r.window}
                {r.replicate && <span className="text-warn"> r</span>}
              </span>
              <ReadCells read={r} size={i === 0 ? 'md' : 'sm'} />
              <ReadVerdict state={state} read={r} />
            </li>
          );
        })}
      </ol>
    </Panel>
  );
}

const shortName = (name: string) => name.replace(/\s*\(.*\)\s*$/, '');

/** "matches HbS", or the closest option and its mismatch count. */
export function readSummary(state: RunState, read: DnaReadView): { text: string; ok: boolean } {
  const matches = readMatches(read);
  if (matches.length) return { text: `matches ${matches.map((h) => shortName(variantName(state, h))).join(', ')}`, ok: true };
  if (!Object.keys(read.expected).length) return { text: '', ok: true };
  const n = read.mismatchPositions.length;
  const closest = read.closest ? ` · closest ${shortName(variantName(state, read.closest))}, ${n} mismatch${n === 1 ? '' : 'es'}` : '';
  return { text: `matches no option${closest}`, ok: false };
}

function ReadVerdict({ state, read }: { state: RunState; read: DnaReadView }) {
  const { text, ok } = readSummary(state, read);
  return (
    <span className={cls('min-w-0 flex-1 truncate text-right text-[0.85rem]', ok ? 'text-dim' : 'font-semibold text-bad')} title={text}>
      {text}
    </span>
  );
}

function Legend({ swatch, text }: { swatch: string; text: string }) {
  return (
    <span className="inline-flex items-center gap-[0.3rem]">
      <span className={cls('inline-block h-[0.6rem] w-[0.9rem] rounded-[0.15rem]', swatch)} />
      {text}
    </span>
  );
}
