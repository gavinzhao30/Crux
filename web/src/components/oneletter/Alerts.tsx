// The surprise banner (surprise.raised, styled like the question card's
// surprise box) and the lesson toast (dna.lesson).

import { useEffect, useState } from 'react';
import type { RunState } from '../../state/fold';
import { activeSurprise, variantName } from '../../state/selectors';
import { pct } from '../../lib/format';
import { ReadCells } from './bits';

export function SurpriseBanner({ state }: { state: RunState }) {
  const sp = activeSurprise(state);
  if (!sp) return null;
  // The read that raised it: the latest one at or before the surprise.
  const read = [...state.dna.reads].reverse().find((r) => r.seq <= sp.seq) ?? null;
  return (
    <div
      key={sp.seq}
      role="alert"
      data-testid="surprise-banner"
      className="flex min-w-0 items-center gap-[1rem] rounded-[0.9rem] border border-warn/70 bg-warn/10 px-[1.1rem] py-[0.6rem] animate-flash"
    >
      <span className="shrink-0 text-[0.8rem] font-bold uppercase tracking-[0.16em] text-warn">Surprise</span>
      {read && <ReadCells read={read} />}
      <span className="min-w-0 flex-1 truncate text-[1.02rem] text-fg" title={sp.reason}>
        {sp.reason}
      </span>
      {sp.nullMass !== null && (
        <span className="tabular shrink-0 text-[0.92rem] font-semibold text-warn">none of these {pct(sp.nullMass)}</span>
      )}
    </div>
  );
}

const TOAST_MS = 12_000;

export function LessonToast({ state }: { state: RunState }) {
  const lesson = state.dna.lesson;
  const [hidden, setHidden] = useState<number | null>(null);

  useEffect(() => {
    if (!lesson) return;
    const t = setTimeout(() => setHidden(lesson.seq), TOAST_MS);
    return () => clearTimeout(t);
  }, [lesson]);

  if (!lesson || hidden === lesson.seq) return null;
  const names = lesson.added.map((h) => variantName(state, h));
  return (
    <div
      role="status"
      data-testid="lesson-toast"
      className="fixed bottom-[1.4rem] right-[1.4rem] z-40 max-w-[30rem] rounded-[0.8rem] border border-accent/60 bg-raised/95 px-[1.1rem] py-[0.8rem] shadow-2xl backdrop-blur animate-fade-up"
    >
      <div className="text-[0.72rem] font-bold uppercase tracking-[0.16em] text-accent">Lesson saved</div>
      <div className="mt-[0.25rem] text-[1.02rem] leading-snug text-fg">
        {names.length ? (
          <>
            <span className="font-semibold">{names.join(', ')}</span> joins the hypotheses: the next run starts with it.
          </>
        ) : (
          'No new hypothesis this run.'
        )}
      </div>
      <div className="mt-[0.2rem] text-[0.8rem] text-mute">
        {lesson.lessonsTotal} {lesson.lessonsTotal === 1 ? 'lesson' : 'lessons'} on file
      </div>
    </div>
  );
}
