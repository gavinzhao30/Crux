import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { foldEvent, initialState } from '../../state/fold';
import { QuestionsPanel } from '../QuestionsPanel';
import { RankingPanel } from '../RankingPanel';

const ref = {
  trace_id: 'trace-1', run_id: 'r', role: 'designer', step: 'design:q',
  url: 'https://cloud.langfuse.com/project/project-1/traces/trace-1',
};

function event(seq: number, type: string, payload: unknown) {
  return { run_id: 'r', seq, ts: seq, type, payload };
}

describe('decision trace links', () => {
  it('opens the exact model trace from question and ranking cards after replay', () => {
    let state = foldEvent(initialState(), event(0, 'question.opened', {
      question: { id: 'q', text: 'Which method?', kind: 'categorical', status: 'open', impact: 1 },
    }));
    state = foldEvent(state, event(1, 'experiment.scored', {
      question_id: 'q', chosen: 'exp', ranking: [{ experiment_id: 'exp', goal: 'Check method',
        kind: 'compute', eig_bits: 1, score: 1, est_cost_s: 1, est_cost_usd: 0.01 }], traces: [ref],
    }));
    for (const panel of [<QuestionsPanel state={state} />, <RankingPanel state={state} />]) {
      const html = renderToStaticMarkup(panel);
      expect(html).toContain(`href="${ref.url}"`);
      expect(html).toContain('target="_blank"');
      expect(html).toContain('rel="noopener noreferrer"');
      expect(html).toContain('Model trace · designer');
    }
  });

  it('ignores unsafe URLs and traces from other runs', () => {
    const state = foldEvent(initialState(), event(0, 'experiment.scored', {
      question_id: 'q', ranking: [], traces: [
        { ...ref, url: 'javascript:alert(1)' },
        { ...ref, run_id: 'other' }, { ...ref, url: null },
      ],
    }));
    expect(state.decisionTraces).toEqual({});
  });
});
