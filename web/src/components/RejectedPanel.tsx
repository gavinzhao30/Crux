import type { RunState } from '../state/fold';
import { evidenceText, optionStatement, questionLabel } from '../state/selectors';
import { Empty, Panel } from './Panel';

export function RejectedPanel({ state }: { state: RunState }) {
  const rows = state.rejected;
  return (
    <Panel
      title="Rejected ledger"
      meta={rows.length ? `${rows.length} ruled out` : undefined}
    >
      {rows.length === 0 ? (
        <Empty>Options ruled out by evidence are recorded here.</Empty>
      ) : (
        <ul className="flex flex-col gap-[0.65rem]">
          {rows.map((r) => {
            const ev = evidenceText(state, r.evidenceId);
            return (
              <li key={`${r.questionId}-${r.hypothesisId}`} className="animate-fade-up">
                <div className="flex min-w-0 items-baseline gap-[0.5rem]">
                  <span className="shrink-0 font-bold text-bad">✕</span>
                  <span className="min-w-0 flex-1 truncate text-[0.98rem] font-medium text-dim line-through decoration-mute decoration-2">
                    {optionStatement(state, r.questionId, r.hypothesisId)}
                  </span>
                  <span className="shrink-0 font-mono text-[0.8rem] font-bold text-mute">
                    {questionLabel(state, r.questionId)}
                  </span>
                </div>
                <p className="mt-[0.15rem] line-clamp-2 pl-[1.1rem] text-[0.86rem] leading-snug text-fg/85">
                  {r.reason}
                </p>
                {(r.evidenceId || ev) && (
                  <p className="mt-[0.1rem] truncate pl-[1.1rem] text-[0.78rem] text-mute" title={ev ?? undefined}>
                    evidence{' '}
                    {r.evidenceId && <span className="font-mono text-dim">{r.evidenceId}</span>}
                    {ev && <span> · {ev}</span>}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
