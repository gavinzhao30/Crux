import type { TraceRef } from '../types';

export function DecisionTraces({ traces = [] }: { traces?: TraceRef[] }) {
  const latest = new Map<string, TraceRef>();
  for (const trace of traces) if (trace.url) latest.set(trace.role, trace);
  if (!latest.size) return null;
  return (
    <span className="flex flex-wrap gap-2 text-[0.75rem] text-accent">
      {[...latest.values()].map((trace) => (
        <a key={trace.trace_id} href={trace.url!} target="_blank" rel="noopener noreferrer"
          title={`${trace.role}: ${trace.step}`} className="underline">
          Model trace · {trace.role}
        </a>
      ))}
    </span>
  );
}
