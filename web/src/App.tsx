import { useEffect, useState } from 'react';
import { useRunStream } from './data/useRunStream';
import { focusQuestionId } from './state/selectors';
import { Header } from './components/Header';
import { QuestionsPanel } from './components/QuestionsPanel';
import { RankingPanel } from './components/RankingPanel';
import { CurvesPanel } from './components/CurvesPanel';
import { ConsolePanel } from './components/ConsolePanel';
import { MemoryPanel } from './components/MemoryPanel';
import { CriticPanel } from './components/CriticPanel';
import { RejectedPanel } from './components/RejectedPanel';
import { ControlBar } from './components/ControlBar';
import { Boundary } from './components/Boundary';

function readSpeed(raw: string | null): number {
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.min(16, n) : 1;
}

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
}

function toggleFullscreen() {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen?.();
}

export default function App() {
  const stream = useRunStream();
  const { state, playLocal, playApiReplay, connectLive, togglePause, setSpeed, restart } = stream;
  const [showBar, setShowBar] = useState(false);

  // URL entry points: ?replay=<fixture>&speed=2 (offline), ?apireplay=<name>,
  // ?run=<id> (live). For offline replays, ?delay=<s> waits before starting,
  // ?seek=<n> applies the first n events at once and ?pause=1 starts paused.
  // The callbacks are stable, so this runs once (twice under StrictMode in
  // development, where the second run simply replaces the first source).
  useEffect(() => {
    const qs = new URLSearchParams(window.location.search);
    const speed = readSpeed(qs.get('speed'));
    const replay = qs.get('replay');
    const apiReplay = qs.get('apireplay');
    const run = qs.get('run');
    if (replay) {
      playLocal(replay, speed, {
        startDelayMs: Math.max(0, Number(qs.get('delay')) || 0) * 1000,
        seekTo: Math.max(0, Number(qs.get('seek')) || 0),
        paused: qs.get('pause') === '1',
      });
    } else if (apiReplay) {
      playApiReplay(apiReplay, speed);
    } else if (run) {
      connectLive(run);
    }
    setShowBar(qs.get('controls') === '1' || (!replay && !apiReplay && !run));
  }, [playLocal, playApiReplay, connectLive]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target)) return;
      switch (e.key) {
        case '`':
          e.preventDefault();
          setShowBar((v) => !v);
          break;
        case ' ':
          e.preventDefault();
          togglePause();
          break;
        case '1':
        case '2':
        case '4':
          setSpeed(Number(e.key));
          break;
        case 'r':
        case 'R':
          restart();
          break;
        case 'f':
        case 'F':
          toggleFullscreen();
          break;
        default:
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [togglePause, setSpeed, restart]);

  const k = state.lastSeq;
  const focus = focusQuestionId(state);
  const curveQuestion =
    focus && state.curves[focus] ? focus : (Object.keys(state.curves)[0] ?? null);

  return (
    <div className="flex h-screen w-screen flex-col gap-[0.85rem] overflow-hidden bg-bg p-[1.05rem] font-sans text-fg">
      <Header state={state} source={stream.source} />
      <main className="grid min-h-0 flex-1 grid-cols-[minmax(0,1.32fr)_minmax(0,1fr)_minmax(0,0.84fr)] gap-[0.85rem]">
        <Boundary name="Questions" resetKey={k}>
          <QuestionsPanel state={state} />
        </Boundary>
        <div className="grid min-h-0 grid-rows-[minmax(0,1.25fr)_minmax(0,1fr)] gap-[0.85rem]">
          <Boundary name="Experiments" resetKey={k}>
            {curveQuestion ? (
              <CurvesPanel state={state} questionId={curveQuestion} />
            ) : (
              <RankingPanel state={state} />
            )}
          </Boundary>
          <Boundary name="Console" resetKey={k}>
            <ConsolePanel state={state} />
          </Boundary>
        </div>
        <div className="grid min-h-0 grid-rows-[minmax(0,1.2fr)_minmax(0,0.9fr)_minmax(0,0.95fr)] gap-[0.85rem]">
          <Boundary name="Memory" resetKey={k}>
            <MemoryPanel state={state} />
          </Boundary>
          <Boundary name="Critic" resetKey={k}>
            <CriticPanel state={state} />
          </Boundary>
          <Boundary name="Rejected ledger" resetKey={k}>
            <RejectedPanel state={state} />
          </Boundary>
        </div>
      </main>
      {showBar && <ControlBar stream={stream} onClose={() => setShowBar(false)} />}
    </div>
  );
}
