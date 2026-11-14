// React binding: owns the current event source and folds its events into
// RunState. Events are flushed in small batches so a burst (an SSE backlog
// on connect) renders once instead of once per event.

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { foldAll, initialState, type RunState } from '../state/fold';
import type { RunEvent } from '../types';
import {
  API_URL,
  LocalReplay,
  SseSource,
  apiReplayUrl,
  liveUrl,
  loadFixture,
  type ConnStatus,
  type SourceCallbacks,
} from './sources';

export type SourceKind = 'none' | 'local' | 'api-replay' | 'live';

export interface LocalOptions {
  startDelayMs?: number;
  seekTo?: number;
  paused?: boolean;
}

export interface SourceInfo {
  kind: SourceKind;
  name: string;
  speed: number;
  status: ConnStatus;
}

type Action = { type: 'reset' } | { type: 'events'; events: RunEvent[] };

function reducer(state: RunState, action: Action): RunState {
  switch (action.type) {
    case 'reset':
      return initialState();
    case 'events':
      return foldAll(action.events, state);
    default:
      return state;
  }
}

export interface RunStream {
  state: RunState;
  source: SourceInfo;
  playLocal: (name: string, speed?: number, opts?: LocalOptions) => void;
  playApiReplay: (name: string, speed?: number) => void;
  connectLive: (runId: string) => void;
  togglePause: () => void;
  setSpeed: (speed: number) => void;
  restart: () => void;
  disconnect: () => void;
}

export function useRunStream(): RunStream {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);
  const [source, setSource] = useState<SourceInfo>({
    kind: 'none',
    name: '',
    speed: 1,
    status: { state: 'idle' },
  });

  const current = useRef<LocalReplay | SseSource | null>(null);
  const token = useRef(0);
  const queue = useRef<RunEvent[]>([]);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sourceRef = useRef(source);
  sourceRef.current = source;

  const flush = useCallback(() => {
    flushTimer.current = null;
    if (queue.current.length === 0) return;
    const events = queue.current;
    queue.current = [];
    dispatch({ type: 'events', events });
  }, []);

  const callbacks = useCallback(
    (myToken: number): SourceCallbacks => ({
      onEvent: (ev) => {
        if (token.current !== myToken) return;
        queue.current.push(ev);
        if (!flushTimer.current) flushTimer.current = setTimeout(flush, 16);
      },
      onStatus: (status) => {
        if (token.current !== myToken) return;
        setSource((s) => ({ ...s, status }));
      },
      onReset: () => {
        if (token.current !== myToken) return;
        queue.current = [];
        dispatch({ type: 'reset' });
      },
    }),
    [flush],
  );

  const teardown = useCallback(() => {
    token.current += 1;
    current.current?.stop();
    current.current = null;
    queue.current = [];
    if (flushTimer.current) clearTimeout(flushTimer.current);
    flushTimer.current = null;
  }, []);

  const playLocal = useCallback(
    (name: string, speed = 1, opts: LocalOptions = {}) => {
      teardown();
      dispatch({ type: 'reset' });
      const myToken = token.current;
      setSource({ kind: 'local', name, speed, status: { state: 'loading' } });
      loadFixture(name)
        .then((events) => {
          if (token.current !== myToken) return;
          const replay = new LocalReplay(events, callbacks(myToken), speed, opts);
          current.current = replay;
          replay.start();
        })
        .catch((err: unknown) => {
          if (token.current !== myToken) return;
          setSource((s) => ({ ...s, status: { state: 'error', detail: String(err) } }));
        });
    },
    [callbacks, teardown],
  );

  const playApiReplay = useCallback(
    (name: string, speed = 1) => {
      teardown();
      dispatch({ type: 'reset' });
      const myToken = token.current;
      setSource({ kind: 'api-replay', name, speed, status: { state: 'connecting' } });
      const sse = new SseSource((from) => apiReplayUrl(name, speed, from, API_URL), callbacks(myToken));
      current.current = sse;
      sse.start();
    },
    [callbacks, teardown],
  );

  const connectLive = useCallback(
    (runId: string) => {
      teardown();
      dispatch({ type: 'reset' });
      const myToken = token.current;
      setSource({ kind: 'live', name: runId, speed: 1, status: { state: 'connecting' } });
      const sse = new SseSource((from) => liveUrl(runId, from, API_URL), callbacks(myToken));
      current.current = sse;
      sse.start();
    },
    [callbacks, teardown],
  );

  const togglePause = useCallback(() => {
    const src = current.current;
    if (src instanceof LocalReplay) {
      src.toggle();
      return;
    }
    // An API replay pauses by disconnecting and resumes from the last seq.
    const info = sourceRef.current;
    if (info.kind === 'api-replay' && src instanceof SseSource) {
      if (info.status.state === 'paused') {
        src.start();
      } else {
        src.stop();
        setSource((s) => ({ ...s, status: { state: 'paused' } }));
      }
    }
  }, []);

  const setSpeed = useCallback(
    (speed: number) => {
      const src = current.current;
      const info = sourceRef.current;
      setSource((s) => ({ ...s, speed }));
      if (src instanceof LocalReplay) {
        src.setSpeed(speed);
      } else if (info.kind === 'api-replay' && src instanceof SseSource) {
        // Reconnect at the new speed, resuming after the last event seen.
        const from = src.resumeSeq - 1;
        src.stop();
        const myToken = token.current;
        const next = new SseSource(
          (f) => apiReplayUrl(info.name, speed, f, API_URL),
          callbacks(myToken),
          from,
        );
        current.current = next;
        next.start();
      }
    },
    [callbacks],
  );

  const restart = useCallback(() => {
    const src = current.current;
    const info = sourceRef.current;
    if (src instanceof LocalReplay) {
      src.restart();
    } else if (info.kind === 'api-replay') {
      playApiReplay(info.name, info.speed);
    } else if (info.kind === 'local') {
      playLocal(info.name, info.speed);
    }
  }, [playApiReplay, playLocal]);

  const disconnect = useCallback(() => {
    teardown();
    setSource((s) => ({ ...s, kind: 'none', status: { state: 'idle' } }));
  }, [teardown]);

  useEffect(() => teardown, [teardown]);

  return {
    state,
    source,
    playLocal,
    playApiReplay,
    connectLive,
    togglePause,
    setSpeed,
    restart,
    disconnect,
  };
}
