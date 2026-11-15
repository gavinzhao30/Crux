// Lazy loader for PDBe Molstar (npm pdbe-molstar, Apache-2.0, EMBL-EBI).
//
// The package's library entry is CommonJS over the whole molstar source tree,
// so the One Letter view loads the prebuilt, self-contained bundle instead:
// a dynamic import resolves the bundle's URL (Vite emits it as a separate
// asset), then a <script> tag defines window.PDBeMolstarPlugin. Other views
// never fetch it.

/** The subset of the PDBeMolstarPlugin API (pdbe-molstar 3.12.0) this app uses. */
export interface MolstarQuery {
  label_asym_id?: string;
  beg_label_seq_id?: number;
  end_label_seq_id?: number;
  color?: string;
  sideChain?: boolean;
  representation?: string;
  representationColor?: string;
  focus?: boolean;
  tooltip?: string;
}

export interface MolstarInitParams {
  moleculeId: string;
  bgColor: string;
  hideControls: boolean;
  hideCanvasControls: string[];
  sequencePanel: boolean;
  leftPanel: boolean;
  rightPanel: boolean;
  logPanel: boolean;
  pdbeLink: boolean;
  loadingOverlay: boolean;
  landscape: boolean;
  selectInteraction: boolean;
  visualStyle: string;
  hideStructure: string[];
  highlightColor: string;
  selection: { data: MolstarQuery[]; nonSelectedColor?: string };
}

export interface MolstarViewer {
  render(target: HTMLElement, options: Partial<MolstarInitParams>): Promise<void>;
  events: { loadComplete: { subscribe(fn: (ok: boolean) => void): { unsubscribe(): void } } };
  visual: {
    update(options: Partial<MolstarInitParams>, fullLoad?: boolean): Promise<boolean>;
    select(params: { data: MolstarQuery[]; nonSelectedColor?: string }): Promise<void>;
    tooltips(params: { data: MolstarQuery[] }): Promise<void>;
    focus(selection: MolstarQuery[]): Promise<void>;
  };
  clear(): Promise<void>;
  plugin?: { dispose?: () => void };
}

type MolstarCtor = new () => MolstarViewer;

declare global {
  interface Window {
    PDBeMolstarPlugin?: MolstarCtor;
  }
}

let pending: Promise<MolstarCtor> | null = null;

function addTag<T extends HTMLElement>(el: T, done: () => void, fail: (e: Error) => void): void {
  el.addEventListener('load', () => done(), { once: true });
  el.addEventListener('error', () => fail(new Error('Molstar bundle failed to load')), { once: true });
  document.head.appendChild(el);
}

/** Load the PDBe Molstar bundle once; later calls share the same promise. */
export function loadMolstar(): Promise<MolstarCtor> {
  if (typeof window === 'undefined') return Promise.reject(new Error('no window'));
  if (window.PDBeMolstarPlugin) return Promise.resolve(window.PDBeMolstarPlugin);
  if (pending) return pending;
  pending = (async () => {
    const [{ default: scriptUrl }, { default: cssUrl }] = await Promise.all([
      import('pdbe-molstar/build/pdbe-molstar-plugin.js?url'),
      // pdbe-molstar.css is the dark skin (the light one is pdbe-molstar-light.css).
      import('pdbe-molstar/build/pdbe-molstar.css?url'),
    ]);
    await new Promise<void>((resolve) => {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = cssUrl;
      // The stylesheet is cosmetic: carry on without it.
      addTag(link, resolve, () => resolve());
    });
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = scriptUrl;
      script.async = true;
      addTag(script, resolve, reject);
    });
    if (!window.PDBeMolstarPlugin) throw new Error('PDBeMolstarPlugin missing after load');
    return window.PDBeMolstarPlugin;
  })().catch((err: unknown) => {
    pending = null;
    throw err;
  });
  return pending;
}
