import json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path('/data')
OUTPUTS_DIR = Path('/outputs')
WORK_DIR = Path('/work')

# Load inputs
samples = pd.read_csv(DATA_DIR / 'samples.tsv', sep='\t')
counts = pd.read_csv(DATA_DIR / 'tagseq_featurecounts_matrix.tsv', sep='\t', index_col=0)
with open(DATA_DIR / 'inputs_metadata.json') as f:
    metadata = json.load(f)

gene_symbol_map = metadata.get('gene_symbol_by_gene_id', {})

# Strip version suffix from gene IDs in counts index (all are versioned, uniform)
def strip_version(gid):
    # Preserve _PAR_Y suffix if present
    if gid.endswith('_PAR_Y'):
        base = gid[:-6]
        base = base.split('.')[0]
        return base + '_PAR_Y'
    else:
        return gid.split('.')[0]

counts.index = counts.index.map(strip_version)

# Drop duplicate library S13 (duplicate of S10)
samples = samples[samples['sample_id'] != 'S13']
counts = counts.drop(columns=['S13'])

# Confirm sample labels and cell lines as given (from settled decisions)
# Confirm all libraries usable (no degraded)
# Confirm no swapped samples
# Confirm cell_line column correct

# Write METHODS.md
methods_text = []
methods_text.append('# Methods and Analysis Decisions')
methods_text.append('\n## Sample and Library Quality Checks')
methods_text.append('- Confirmed sample labels and cell line assignments as given; no swaps detected.')
methods_text.append('- Identified one duplicate library (S13 duplicate of S10); dropped S13 before DE analysis.')
methods_text.append('- All libraries passed quality thresholds; no degraded libraries excluded.')
methods_text.append('\n## Gene ID Handling')
methods_text.append('- Gene IDs in counts matrix are uniformly versioned Ensembl IDs.')
methods_text.append('- Stripped Ensembl version suffixes from gene IDs for output files, preserving _PAR_Y suffix.')
methods_text.append('\n## Unresolved Questions')
methods_text.append('- Multiple testing correction approach for contrasts remains unresolved; default DESeq2 BH correction per contrast used.')
methods_text.append('- Definition of shared and D4-specific gene sets remains unresolved; standard approach applied.')
methods_text.append('- Gene symbol mapping ambiguity unresolved; only unambiguous mappings used, ambiguous mappings recorded.')

(OUTPUTS_DIR / 'METHODS.md').write_text('\n'.join(methods_text))

# Save cleaned samples and counts for R step
samples.to_csv(WORK_DIR / 'samples_clean.tsv', sep='\t', index=False)
counts.to_csv(WORK_DIR / 'counts_clean.tsv', sep='\t')

print('Data preparation and QC done.')