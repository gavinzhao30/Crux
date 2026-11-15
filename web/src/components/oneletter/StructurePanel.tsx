// 3D haemoglobin in PDBe Molstar (npm pdbe-molstar, Apache-2.0, EMBL-EBI):
// the entry named by the latest dna.verdict, with beta residue 6 picked out
// on every beta chain. Loaded on demand; offline it shows a placeholder.

import { useEffect, useRef, useState } from 'react';
import type { DnaVerdictView, RunState } from '../../state/fold';
import { structureInfo } from '../../lib/hbb';
import { verdictSuperseded } from '../../state/selectors';
import { loadMolstar, type MolstarQuery, type MolstarViewer } from '../../lib/molstar';
import { Panel } from '../Panel';

const LOAD_TIMEOUT_MS = 30_000;
const BG = '#0d1116';

type Status = 'idle' | 'loading' | 'ready' | 'offline';

/** Beta chains (label_asym_id) of an entry: known table first, then PDBe entry metadata. */
export async function betaChains(pdbId: string): Promise<string[]> {
  const known = structureInfo(pdbId);
  if (known) return known.betaChains;
  const id = pdbId.toLowerCase();
  const res = await fetch(`https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`PDBe molecules ${res.status}`);
  const data = (await res.json()) as Record<string, { molecule_name?: string[]; in_struct_asyms?: string[] }[]>;
  const beta = (data[id] ?? []).find((m) => (m.molecule_name ?? []).some((n) => /subunit beta|beta chain/i.test(n)));
  return beta?.in_struct_asyms ?? [];
}

/**
 * Selection for residue `residue` on each beta chain: coloured and drawn as
 * spheres so it reads at whole-molecule zoom (no camera focus, so the
 * audience sees the full haemoglobin).
 */
export function residueSelection(chains: string[], residue: number, color: string): MolstarQuery[] {
  return chains.map((c) => ({
    label_asym_id: c,
    beg_label_seq_id: residue,
    end_label_seq_id: residue,
    color,
    representation: 'spacefill',
    representationColor: color,
  }));
}

/** Everything but residue 6 in a muted tone. */
const REST_COLOR = '#56616e';

/** Red when the entry carries the variant residue, green when it shows the normal one. */
function residueColor(v: DnaVerdictView): string {
  const info = structureInfo(v.pdbId);
  return info && info.beta6 !== 'Glu' ? '#ff6b6b' : '#46e0b4';
}

function caption(v: DnaVerdictView): string {
  const info = structureInfo(v.pdbId);
  if (!info) return `PDB ${v.pdbId} · beta residue ${v.residue}`;
  const base = `${info.title}, ${info.resolution} · β${v.residue} ${info.beta6} on chains ${info.betaChains.join(', ')}`;
  // 1A3N is normal haemoglobin: say so when the verdict is not HbA.
  const short = v.name.replace(/\s*\(.*\)\s*$/, '');
  if (info.beta6 === 'Glu' && short && short !== 'HbA') {
    const what = v.hypothesisId === 'h0' ? 'the variant read here' : short;
    return `${base}. This entry is normal haemoglobin; ${what} changes β${v.residue}.`;
  }
  return base;
}

export function StructurePanel({ state }: { state: RunState }) {
  const verdict = state.dna.verdict;
  const pdbId = verdict?.pdbId || null;
  const residue = verdict?.residue ?? 6;
  const host = useRef<HTMLDivElement | null>(null);
  const viewer = useRef<MolstarViewer | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [detail, setDetail] = useState('');

  useEffect(() => {
    if (!pdbId || !verdict) return;
    let cancelled = false;
    let sub: { unsubscribe(): void } | null = null;
    setStatus('loading');
    setDetail('');
    const timer = setTimeout(() => {
      if (!cancelled) {
        setStatus('offline');
        setDetail(`PDB ${pdbId} did not load within ${LOAD_TIMEOUT_MS / 1000} s`);
      }
    }, LOAD_TIMEOUT_MS);

    (async () => {
      const [Ctor, chains] = await Promise.all([loadMolstar(), betaChains(pdbId)]);
      if (cancelled || !host.current) return;
      const color = residueColor(verdict);
      const selection = { data: residueSelection(chains, residue, color), nonSelectedColor: REST_COLOR };
      const params = {
        moleculeId: pdbId.toLowerCase(),
        bgColor: BG,
        hideControls: true,
        hideCanvasControls: ['expand', 'selection', 'animation', 'controlToggle', 'controlInfo', 'trajectory'],
        sequencePanel: false,
        leftPanel: false,
        rightPanel: false,
        logPanel: false,
        pdbeLink: false,
        loadingOverlay: false,
        landscape: true,
        selectInteraction: false,
        visualStyle: 'cartoon',
        hideStructure: ['water'],
        highlightColor: '#f5b83d',
        selection,
      };
      const done = (ok: boolean) => {
        if (cancelled) return;
        clearTimeout(timer);
        setStatus(ok ? 'ready' : 'offline');
        if (!ok) setDetail(`PDB ${pdbId} could not be fetched from PDBe`);
        if (ok && viewer.current) {
          void viewer.current.visual
            .tooltips({ data: chains.map((c) => ({ label_asym_id: c, beg_label_seq_id: residue, end_label_seq_id: residue, tooltip: `β${residue} (codon ${residue})` })) })
            .catch(() => undefined);
        }
      };
      if (!viewer.current) {
        const v = new Ctor();
        viewer.current = v;
        sub = v.events.loadComplete.subscribe(done);
        await v.render(host.current, params);
      } else {
        sub = viewer.current.events.loadComplete.subscribe(done);
        await viewer.current.visual.update(params, true);
      }
    })().catch((err: unknown) => {
      if (cancelled) return;
      clearTimeout(timer);
      setStatus('offline');
      setDetail(err instanceof Error ? err.message : String(err));
    });

    return () => {
      cancelled = true;
      clearTimeout(timer);
      sub?.unsubscribe();
    };
    // The structure depends only on the entry, the residue and the verdict name.
  }, [pdbId, residue, verdict?.name]);

  // Dispose of the plugin with the view.
  useEffect(
    () => () => {
      const v = viewer.current;
      viewer.current = null;
      try {
        v?.plugin?.dispose?.();
      } catch {
        // already gone
      }
    },
    [],
  );

  // Before the first effect runs (and on the server) a verdict means loading.
  const shown: Status = pdbId && status === 'idle' ? 'loading' : status;

  return (
    <Panel
      title="Structure · PDBe Molstar"
      meta={
        pdbId ? (
          <span className="font-mono font-semibold text-dim">
            {verdictSuperseded(state) && <span className="font-sans text-warn">previous verdict · </span>}
            PDB {pdbId} · β{residue}
          </span>
        ) : undefined
      }
      bodyClassName="flex flex-col gap-[0.5rem]"
    >
      <div
        data-testid="molstar"
        className="relative min-h-0 flex-1 overflow-hidden rounded-[0.6rem] border border-line"
        style={{ background: BG }}
      >
        <div ref={host} className="absolute inset-0" />
        {shown !== 'ready' && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-[1rem] text-center text-[0.92rem] leading-relaxed text-mute">
            {shown === 'idle' && (
              <div>
                The structure loads when Crux settles.
                <br />
                <span className="text-dim">2HBS (sickle, β6 Val) or 1A3N (normal, β6 Glu)</span>
              </div>
            )}
            {shown === 'loading' && <div>Loading PDB {pdbId} from PDBe…</div>}
            {shown === 'offline' && (
              <div>
                3D view unavailable (offline?)
                <br />
                <span className="text-dim">{detail}</span>
              </div>
            )}
          </div>
        )}
      </div>
      {verdict && pdbId && <p className="line-clamp-2 text-[0.82rem] leading-snug text-dim">{caption(verdict)}</p>}
    </Panel>
  );
}
