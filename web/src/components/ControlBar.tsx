import { useEffect, useState, type ReactNode } from 'react';
import type { RunStream, SourceKind } from '../data/useRunStream';
import {
  API_URL,
  fetchFixtureIndex,
  fetchReplays,
  fetchRuns,
  startRun,
  type NamedItem,
} from '../data/sources';
import { cls } from '../lib/format';

const SPEEDS = [1, 2, 4];
const LABS = ['bixbench3', 'boxing', 'bench'];

type Tab = Exclude<SourceKind, 'none'>;

function Btn({
  children,
  onClick,
  active,
  disabled,
  title,
}: {
  children: ReactNode;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cls(
        'rounded-[0.4rem] border px-[0.7rem] py-[0.3rem] text-[0.82rem] font-semibold transition-colors',
        active ? 'border-accent/70 bg-accent/15 text-accent' : 'border-line-strong text-dim hover:text-fg',
        disabled && 'cursor-not-allowed opacity-40',
      )}
    >
      {children}
    </button>
  );
}

function Select({
  items,
  value,
  onChange,
  placeholder,
}: {
  items: NamedItem[];
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="max-w-[22rem] rounded-[0.4rem] border border-line-strong bg-panel px-[0.5rem] py-[0.3rem] text-[0.82rem] text-fg"
    >
      {items.length === 0 && <option value="">{placeholder}</option>}
      {items.map((it) => (
        <option key={it.name} value={it.name}>
          {it.title}
        </option>
      ))}
    </select>
  );
}

export function ControlBar({ stream, onClose }: { stream: RunStream; onClose: () => void }) {
  const { source } = stream;
  const [tab, setTab] = useState<Tab>(source.kind === 'none' ? 'local' : source.kind);
  const [fixtures, setFixtures] = useState<NamedItem[]>([]);
  const [replays, setReplays] = useState<NamedItem[]>([]);
  const [runs, setRuns] = useState<NamedItem[]>([]);
  const [fixture, setFixture] = useState('demo_categorical');
  const [replay, setReplay] = useState('');
  const [runId, setRunId] = useState('');
  const [lab, setLab] = useState('bixbench3');
  const [seed, setSeed] = useState(0);
  const [note, setNote] = useState('');

  useEffect(() => {
    void fetchFixtureIndex().then((items) => {
      setFixtures(items);
      setFixture((cur) => (items.some((i) => i.name === cur) ? cur : (items[0]?.name ?? '')));
    });
  }, []);

  useEffect(() => {
    setNote('');
    if (tab === 'api-replay') {
      fetchReplays()
        .then((items) => {
          setReplays(items);
          if (items[0]) setReplay((r) => r || items[0]!.name);
        })
        .catch(() => setNote(`API unreachable at ${API_URL}`));
    } else if (tab === 'live') {
      fetchRuns()
        .then((items) => {
          setRuns(items);
          if (items[0]) setRunId((r) => r || items[items.length - 1]!.name);
        })
        .catch(() => setNote(`API unreachable at ${API_URL}`));
    }
  }, [tab]);

  const isReplay = source.kind === 'local' || source.kind === 'api-replay';
  const paused = source.status.state === 'paused';
  const ended = source.status.state === 'ended';

  const onStartRun = () => {
    setNote('Starting run…');
    startRun({ lab, config: {}, seed })
      .then((id) => {
        setNote('');
        setRunId(id);
        stream.connectLive(id);
      })
      .catch((err: unknown) => setNote(String(err instanceof Error ? err.message : err)));
  };

  return (
    <div className="fixed inset-x-0 bottom-[1rem] z-50 flex justify-center px-[1rem]">
      <div className="flex max-w-full flex-wrap items-center gap-[0.6rem] rounded-[0.8rem] border border-line-strong bg-raised/95 px-[0.9rem] py-[0.6rem] text-[0.82rem] shadow-2xl backdrop-blur">
        <div className="flex gap-[0.3rem]">
          <Btn active={tab === 'local'} onClick={() => setTab('local')} title="Bundled fixture, works offline">
            Offline replay
          </Btn>
          <Btn active={tab === 'api-replay'} onClick={() => setTab('api-replay')} title="Replay streamed by the API">
            API replay
          </Btn>
          <Btn active={tab === 'live'} onClick={() => setTab('live')} title="Live run over SSE">
            Live
          </Btn>
        </div>
        <span className="h-[1.4rem] w-px bg-line-strong" />

        {tab === 'local' && (
          <>
            <Select items={fixtures} value={fixture} onChange={setFixture} placeholder="no fixtures" />
            <Btn onClick={() => stream.playLocal(fixture, source.speed)} disabled={!fixture}>
              Play
            </Btn>
          </>
        )}
        {tab === 'api-replay' && (
          <>
            <Select items={replays} value={replay} onChange={setReplay} placeholder="no replays" />
            <Btn onClick={() => stream.playApiReplay(replay, source.speed)} disabled={!replay}>
              Play
            </Btn>
          </>
        )}
        {tab === 'live' && (
          <>
            <Select items={runs} value={runId} onChange={setRunId} placeholder="no runs" />
            <input
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              placeholder="run id"
              className="w-[13rem] rounded-[0.4rem] border border-line-strong bg-panel px-[0.5rem] py-[0.3rem] font-mono text-[0.8rem] text-fg"
            />
            <Btn onClick={() => stream.connectLive(runId)} disabled={!runId}>
              Connect
            </Btn>
            <span className="h-[1.4rem] w-px bg-line-strong" />
            <select
              value={lab}
              onChange={(e) => setLab(e.target.value)}
              className="rounded-[0.4rem] border border-line-strong bg-panel px-[0.5rem] py-[0.3rem] text-[0.82rem] text-fg"
            >
              {LABS.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value) || 0)}
              className="w-[4.5rem] rounded-[0.4rem] border border-line-strong bg-panel px-[0.5rem] py-[0.3rem] text-[0.82rem] text-fg"
              title="seed"
            />
            <Btn onClick={onStartRun}>New run</Btn>
          </>
        )}

        {isReplay && (
          <>
            <span className="h-[1.4rem] w-px bg-line-strong" />
            <Btn onClick={ended ? stream.restart : stream.togglePause} title="space">
              {ended ? 'Replay' : paused ? 'Resume' : 'Pause'}
            </Btn>
            <Btn onClick={stream.restart} title="r">
              Restart
            </Btn>
            <div className="flex gap-[0.2rem]">
              {SPEEDS.map((s) => (
                <Btn key={s} active={source.speed === s} onClick={() => stream.setSpeed(s)} title={String(s)}>
                  {s}×
                </Btn>
              ))}
            </div>
          </>
        )}

        <span className="min-w-[8rem] text-mute">
          {note ||
            `${source.status.state}${source.status.detail ? ` · ${source.status.detail}` : ''}`}
        </span>
        <span className="hidden text-[0.74rem] text-mute xl:inline">
          ` hide · space pause · 1 2 4 speed · r restart · f full screen
        </span>
        <Btn onClick={onClose} title="`">
          Hide
        </Btn>
      </div>
    </div>
  );
}
