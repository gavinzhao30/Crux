// One Letter (labs/oneletter): camera and window ranking on the left, the HBB
// sequence and the posteriors in the centre, the structure and the verdict on
// the right. Everything renders from RunState.

import type { RunState } from '../../state/fold';
import type { SourceInfo } from '../../data/useRunStream';
import { isRecorded } from '../../state/selectors';
import { Boundary } from '../Boundary';
import { QuestionsPanel } from '../QuestionsPanel';
import { RankingPanel } from '../RankingPanel';
import { LessonToast, SurpriseBanner } from './Alerts';
import { CameraPanel } from './CameraPanel';
import { SequencePanel } from './SequencePanel';
import { StructurePanel } from './StructurePanel';
import { VerdictCard } from './VerdictCard';

export function OneLetterView({ state, source }: { state: RunState; source: SourceInfo }) {
  const k = state.lastSeq;
  const recorded = isRecorded(state, source.kind);
  return (
    <div data-layout="oneletter" className="flex min-h-0 flex-1 flex-col gap-[0.85rem]">
      <SurpriseBanner state={state} />
      <main className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_minmax(0,1fr)] gap-[0.85rem]">
        <div className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto] gap-[0.85rem]">
          <Boundary name="Camera" resetKey={k}>
            <CameraPanel state={state} recorded={recorded} />
          </Boundary>
          <Boundary name="Window ranking" resetKey={k}>
            <RankingPanel state={state} />
          </Boundary>
        </div>
        <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-[0.85rem]">
          <Boundary name="Sequence" resetKey={k}>
            <SequencePanel state={state} />
          </Boundary>
          <Boundary name="Questions" resetKey={k}>
            <QuestionsPanel state={state} maxDensity="normal" />
          </Boundary>
        </div>
        <div className="grid min-h-0 grid-rows-[minmax(0,1.15fr)_minmax(0,1fr)] gap-[0.85rem]">
          <Boundary name="Structure" resetKey={k}>
            <StructurePanel state={state} />
          </Boundary>
          <Boundary name="Verdict" resetKey={k}>
            <VerdictCard state={state} />
          </Boundary>
        </div>
      </main>
      <LessonToast state={state} />
    </div>
  );
}
