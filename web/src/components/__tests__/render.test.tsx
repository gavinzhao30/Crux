import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { foldEvent, initialState, type RunState } from '../../state/fold';
import { parseJsonl } from '../../data/sources';
import { Header } from '../Header';
import { QuestionsPanel } from '../QuestionsPanel';
import { RankingPanel } from '../RankingPanel';
import { CurvesPanel } from '../CurvesPanel';
import { ConsolePanel } from '../ConsolePanel';
import { MemoryPanel } from '../MemoryPanel';
import { CriticPanel } from '../CriticPanel';
import { RejectedPanel } from '../RejectedPanel';

const FIXTURES = fileURLToPath(new URL('../../../public/fixtures/', import.meta.url));
const source = { kind: 'local' as const, name: 'fixture', speed: 2, status: { state: 'playing' as const } };

function renderAll(s: RunState): string {
  const curveQ = Object.keys(s.curves)[0];
  return [
    <Header state={s} source={source} />,
    <QuestionsPanel state={s} />,
    curveQ ? <CurvesPanel state={s} questionId={curveQ} /> : <RankingPanel state={s} />,
    <ConsolePanel state={s} />,
    <MemoryPanel state={s} />,
    <CriticPanel state={s} />,
    <RejectedPanel state={s} />,
  ]
    .map((el) => renderToStaticMarkup(el))
    .join('');
}

describe('every panel renders after every event', () => {
  for (const name of ['demo_categorical', 'demo_numeric', 'demo_error']) {
    it(name, () => {
      const events = parseJsonl(readFileSync(`${FIXTURES}${name}.jsonl`, 'utf8'));
      let s = initialState();
      expect(() => renderAll(s)).not.toThrow();
      for (const ev of events) {
        s = foldEvent(s, ev);
        expect(() => renderAll(s), `after seq ${ev.seq} (${ev.type})`).not.toThrow();
      }
      const html = renderAll(s);
      if (name === 'demo_categorical') {
        expect(html).toContain('Control and knockout are inverted in T98G');
        expect(html).toContain('required_columns');
        expect(html).toContain('Replay ×2');
      }
    });
  }
});
