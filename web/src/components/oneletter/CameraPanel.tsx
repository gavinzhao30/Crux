// The reader's annotated MJPEG (labs/oneletter/reader.py, GET /annotated_stream),
// with the engine's "show window N" prompt from dna.request. Recorded runs
// never show the live camera: the image would not match the reads.

import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { RunState } from '../../state/fold';
import { isFakeReader, pendingRequest, readerProgress, readerUrls } from '../../state/selectors';
import { bits, cls } from '../../lib/format';
import { LiveDot, Panel } from '../Panel';
import { RecordedBadge } from './bits';

const RETRY_MS = 5000;

function LiveCamera({ url }: { url: string }) {
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);
  const img = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!failed) return;
    const t = setTimeout(() => {
      setFailed(false);
      setAttempt((a) => a + 1);
    }, RETRY_MS);
    return () => clearTimeout(t);
  }, [failed]);

  // Close the MJPEG connection when the view goes away.
  useEffect(() => {
    const el = img.current;
    return () => {
      if (el) el.src = '';
    };
  }, []);

  const src = attempt === 0 ? url : `${url}${url.includes('?') ? '&' : '?'}retry=${attempt}`;
  return (
    <>
      <Placeholder>
        {failed ? (
          <>
            Camera unreachable at <span className="font-mono text-dim">{url}</span>
            <br />
            retrying every {RETRY_MS / 1000} s
          </>
        ) : (
          <>Connecting to the reader camera…</>
        )}
      </Placeholder>
      {!failed && (
        <img
          ref={img}
          key={attempt}
          src={src}
          alt="Annotated camera view from the Jetson reader"
          onError={() => setFailed(true)}
          className="absolute inset-0 h-full w-full object-contain"
        />
      )}
    </>
  );
}

function Placeholder({ children }: { children: ReactNode }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center p-[1rem] text-center text-[0.92rem] leading-relaxed text-mute">
      <div>{children}</div>
    </div>
  );
}

export function CameraPanel({ state, recorded }: { state: RunState; recorded: boolean }) {
  const urls = readerUrls(state);
  const req = pendingRequest(state);
  const progress = readerProgress(state);
  const fake = isFakeReader(state);

  return (
    <Panel
      title="Camera · Jetson reader"
      meta={
        recorded ? (
          <RecordedBadge fakeReader={fake} />
        ) : (
          <span className="inline-flex items-center gap-[0.4rem] text-accent">
            <LiveDot /> live
          </span>
        )
      }
      bodyClassName="flex flex-col gap-[0.6rem]"
    >
      <div
        data-testid="camera"
        className="relative min-h-0 flex-1 overflow-hidden rounded-[0.6rem] border border-line bg-[#05070a]"
      >
        {recorded ? (
          <Placeholder>
            {fake ? 'Fake reader: the reads are scripted, there is no camera image.' : 'Recorded run: the live camera is not shown.'}
            <br />
            The letters below come from the event log.
          </Placeholder>
        ) : urls.annotated ? (
          <LiveCamera url={urls.annotated} />
        ) : (
          <Placeholder>No reader URL in the run config.</Placeholder>
        )}

        {req && (
          <div
            key={req.seq}
            data-testid="window-prompt"
            className={cls(
              'absolute inset-x-[0.7rem] bottom-[0.7rem] rounded-[0.6rem] border px-[0.9rem] py-[0.6rem] backdrop-blur animate-fade-up',
              req.replicate ? 'border-warn/70 bg-[#1b1608]/90' : 'border-accent/70 bg-[#071a14]/90',
            )}
          >
            <div className="flex items-baseline gap-[0.7rem]">
              <span className={cls('text-[1.7rem] font-bold leading-none', req.replicate ? 'text-warn' : 'text-accent')}>
                Show window {req.window}
                {req.replicate ? ' again' : ''}
              </span>
              {req.replicate && (
                <span className="rounded-[0.3rem] border border-warn/60 px-[0.4rem] py-[0.05rem] text-[0.68rem] font-bold uppercase tracking-[0.12em] text-warn">
                  replicate
                </span>
              )}
              {req.eigBits !== null && (
                <span className="tabular ml-auto shrink-0 text-[0.9rem] font-semibold text-fg">{bits(req.eigBits)} bits</span>
              )}
            </div>
            {req.reason && <p className="mt-[0.3rem] line-clamp-2 text-[0.88rem] leading-snug text-dim">{req.reason}</p>}
            {progress && (
              <p className="mt-[0.25rem] truncate font-mono text-[0.78rem] text-mute" title={progress} data-testid="reader-progress">
                {progress}
              </p>
            )}
          </div>
        )}
        {state.dna.strip !== null && (
          <span className="absolute left-[0.7rem] top-[0.6rem] rounded-[0.3rem] border border-line-strong bg-panel/80 px-[0.45rem] py-[0.08rem] text-[0.7rem] font-bold uppercase tracking-[0.12em] text-dim">
            strip {state.dna.strip}
          </span>
        )}
      </div>
      {/* The presenter opens the strip page on the phone; a recorded run has no reader behind it. */}
      {urls.strip && !recorded && (
        <div className="flex min-w-0 items-baseline gap-[0.5rem] text-[0.8rem] text-mute" data-testid="strip-link">
          <span className="shrink-0 font-semibold uppercase tracking-[0.12em]">strip display</span>
          <a
            href={urls.strip}
            target="_blank"
            rel="noreferrer"
            className="min-w-0 truncate font-mono text-dim underline decoration-line-strong underline-offset-2 hover:text-fg"
          >
            {urls.strip}
          </a>
        </div>
      )}
    </Panel>
  );
}
