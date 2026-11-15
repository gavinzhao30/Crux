// Small shared pieces of the One Letter view.

import type { DnaReadView } from '../../state/fold';
import { readCells } from '../../state/selectors';
import { cls } from '../../lib/format';

/** Shown whenever the reads are not coming from the camera right now. */
export function RecordedBadge({ fakeReader }: { fakeReader: boolean }) {
  return (
    <span
      data-testid="recorded-badge"
      title={
        fakeReader
          ? 'Run on the fake reader: the reads are scripted, not taken from the camera.'
          : 'Replay of a recorded run: nothing here is live.'
      }
      className="inline-flex items-center gap-[0.4rem] rounded-[0.35rem] border border-warn/60 bg-warn/10 px-[0.55rem] py-[0.2rem] text-[0.74rem] font-bold uppercase tracking-[0.14em] text-warn"
    >
      <span className="inline-block h-[0.5rem] w-[0.5rem] rounded-full bg-warn" />
      Recorded{fakeReader ? ' · fake reader' : ''}
    </span>
  );
}

/** The nine letters of one read, mismatches boxed in red. */
export function ReadCells({ read, size = 'md' }: { read: DnaReadView; size?: 'sm' | 'md' }) {
  const cells = readCells(read);
  return (
    <span className="inline-flex gap-[0.18rem] font-mono">
      {cells.map((c) => (
        <span
          key={c.index}
          title={[
            `position ${c.index + 1}: read ${c.base}, reference ${c.ref || '?'}`,
            c.confidence !== null ? `confidence ${c.confidence.toFixed(2)}` : '',
            c.mismatch ? 'mismatch against the closest option' : '',
          ]
            .filter(Boolean)
            .join(' · ')}
          className={cls(
            'inline-flex items-center justify-center rounded-[0.25rem] border font-bold leading-none',
            size === 'sm' ? 'h-[1.35rem] w-[1.1rem] text-[0.82rem]' : 'h-[1.75rem] w-[1.45rem] text-[1.05rem]',
            c.mismatch
              ? 'border-bad bg-bad/20 text-bad shadow-[0_0_0_1px_rgb(255_107_107/0.6)]'
              : c.differsFromRef
                ? 'border-warn/70 bg-warn/10 text-warn'
                : 'border-line-strong bg-raised text-fg',
          )}
        >
          {c.base}
        </span>
      ))}
    </span>
  );
}
