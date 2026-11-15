import { useEffect, useState } from 'react';
import { useRunStream } from './data/useRunStream';
import { isFakeReader, isOneLetterLab, isRecorded } from './state/selectors';
import { Header } from './components/Header';
import { ControlBar } from './components/ControlBar';
import { RunView } from './components/RunView';
import { RecordedBadge } from './components/oneletter/bits';

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

  // One Letter runs carry a "recorded" badge whenever the reads are not live.
  const recordedBadge =
    isOneLetterLab(state) && isRecorded(state, stream.source.kind) ? (
      <RecordedBadge fakeReader={isFakeReader(state)} />
    ) : undefined;

  return (
    <div className="flex h-screen w-screen flex-col gap-[0.85rem] overflow-hidden bg-bg p-[1.05rem] font-sans text-fg">
      <Header state={state} source={stream.source} extra={recordedBadge} />
      <RunView state={state} source={stream.source} />
      {showBar && <ControlBar stream={stream} onClose={() => setShowBar(false)} />}
    </div>
  );
}
