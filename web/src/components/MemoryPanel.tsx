import type { ReactNode } from 'react';
import type { FeedItem, RunState } from '../state/fold';
import { questionLabel } from '../state/selectors';
import { cls, shortUrl } from '../lib/format';
import { Empty, Panel, Role } from './Panel';

function uniq(xs: string[]): string[] {
  return [...new Set(xs.filter(Boolean))];
}

function joinRoles(roles: string[]) {
  return roles.map((r, i) => (
    <span key={r}>
      {i > 0 && <span className="text-mute">{i === roles.length - 1 ? ' and ' : ', '}</span>}
      <Role name={r} strong />
    </span>
  ));
}

function Item({ item, state, fresh }: { item: FeedItem; state: RunState; fresh: boolean }) {
  const wrap = (body: ReactNode, sub?: ReactNode, oneLine = false) => (
    <li className={cls('border-b border-line/70 pb-[0.45rem]', fresh && 'animate-fade-up')}>
      <div className={cls('leading-[1.7] text-dim', oneLine && 'truncate')}>{body}</div>
      {sub && <div className="mt-[0.1rem] text-[0.8rem] leading-snug text-mute">{sub}</div>}
    </li>
  );

  switch (item.kind) {
    case 'recall': {
      const authors = uniq(item.authors);
      const n = item.evidenceIds.length;
      return wrap(
        <>
          <Role name={item.reader} strong /> recalled{' '}
          <span className="font-semibold text-fg">
            {n} record{n === 1 ? '' : 's'}
          </span>
          {authors.length > 0 && <> from {joinRoles(authors)}</>}
        </>,
        <>
          “{item.query}” · {item.backend}
          {item.latencyMs !== null && ` · ${Math.round(item.latencyMs)} ms`}
        </>,
      );
    }
    case 'write':
      return wrap(
        <>
          <Role name={item.author} /> wrote <span className="font-mono text-[0.82rem] text-dim">{item.evidenceId}</span>
          <span className="text-mute"> → {item.backend}</span>
        </>,
        undefined,
        true,
      );
    case 'search':
      return wrap(
        <>
          <Role name="search" strong /> {item.urls.length} source{item.urls.length === 1 ? '' : 's'}
          {item.blocked.length > 0 && (
            <span className="text-bad/90">, {item.blocked.length} blocked by the leakage policy</span>
          )}
        </>,
        <>
          <div className="truncate">“{item.query}”</div>
          <ul className="mt-[0.15rem] space-y-[0.05rem] font-mono text-[0.76rem]">
            {item.urls.slice(0, 3).map((u) => (
              <li key={u} className="truncate text-dim" title={u}>
                {shortUrl(u)}
              </li>
            ))}
            {item.urls.length > 3 && <li className="text-mute">+{item.urls.length - 3} more</li>}
            {item.blocked.map((u) => (
              <li key={u} className="truncate text-bad/80 line-through decoration-bad/70" title={`blocked: ${u}`}>
                {shortUrl(u)}
              </li>
            ))}
          </ul>
        </>,
      );
    case 'lesson':
      return wrap(
        <>
          <Role name={item.author} strong /> wrote a lesson
        </>,
        <span className="text-dim">{item.claim}</span>,
      );
    case 'profile':
      return wrap(
        <>
          <Role name={item.author} strong /> profiled the data:{' '}
          <span className="text-fg">{item.claims.length} facts</span>
        </>,
        <span className="line-clamp-2">{item.claims.slice(0, 2).join(' · ')}</span>,
      );
    case 'question':
      return wrap(
        <>
          <Role name={item.author} /> raised{' '}
          <span className="font-mono font-bold text-fg">{questionLabel(state, item.questionId)}</span>
          <span className="text-mute"> · {item.text}</span>
        </>,
        undefined,
        true,
      );
    default:
      return null;
  }
}

export function MemoryPanel({ state }: { state: RunState }) {
  const items = state.feed.slice(0, 14);
  const recalls = state.feed.filter((f) => f.kind === 'recall').length;
  const newest = items[0]?.seq;
  return (
    <Panel
      title="Agents & memory"
      meta={recalls > 0 ? `${recalls} recall${recalls === 1 ? '' : 's'}` : undefined}
    >
      {items.length === 0 ? (
        <Empty>Shared memory reads and writes appear here.</Empty>
      ) : (
        <ul className="flex flex-col gap-[0.45rem] text-[0.9rem]">
          {items.map((it) => (
            <Item key={`${it.kind}-${it.seq}`} item={it} state={state} fresh={it.seq === newest} />
          ))}
        </ul>
      )}
    </Panel>
  );
}
