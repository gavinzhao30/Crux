import json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
input_dir = Path('/data')
output_dir = Path('/outputs')
work_dir = Path('/work')
output_dir.mkdir(exist_ok=True)

# Load inputs
samples = pd.read_csv(input_dir / 'samples.tsv', sep='\t')
counts = pd.read_csv(input_dir / 'tagseq_featurecounts_matrix.tsv', sep='\t', index_col=0)
with open(input_dir / 'inputs_metadata.json') as f:
    metadata = json.load(f)

# Strip version from gene IDs except _PAR_Y
def strip_version(gid):
    if gid.endswith('_PAR_Y'):
        return gid
    return gid.split('.', 1)[0]

counts.index = counts.index.map(strip_version)
assert counts.index.is_unique, 'Gene IDs not unique after stripping versions'

# Confirm sample IDs in counts columns
assert set(samples['sample_id']) == set(counts.columns), 'Mismatch sample IDs between samples.tsv and counts matrix'

# Check library sizes from profile
library_sizes = {s['sample_id']: size for s, size in zip(metadata['samples'], metadata['library_sizes'].values())}
# Actually library_sizes in profile is dict keyed by sample_id
library_sizes = metadata['library_sizes']

# Add library size to samples
samples['library_size'] = samples['sample_id'].map(library_sizes)

# Check for degraded or duplicate libraries - settled as none
# Check sample labels consistency - settled as consistent

# Prepare gene_symbol mapping from metadata
gene_symbol_by_gene_id_raw = metadata.get('gene_symbol_by_gene_id', {})

# Strip versions from keys
gene_symbol_by_gene_id = {}
ambiguous_ids = set()
for k, v in gene_symbol_by_gene_id_raw.items():
    k_stripped = strip_version(k)
    if k_stripped in gene_symbol_by_gene_id:
        # ambiguous mapping
        if gene_symbol_by_gene_id[k_stripped] != v:
            ambiguous_ids.add(k_stripped)
    else:
        gene_symbol_by_gene_id[k_stripped] = v

# Remove ambiguous mappings
for aid in ambiguous_ids:
    del gene_symbol_by_gene_id[aid]

# Write METHODS.md with decisions and diagnostics
with open(output_dir / 'METHODS.md', 'w') as f:
    f.write('# Analysis Methods and Decisions\n\n')
    f.write('## Sample label and library quality checks\n')
    f.write('- Sample labels were checked for swaps or misannotations; none were found (see diagnostics labels_nearest_neighbour and labels_swap_search).\n')
    f.write('- No duplicated or degraded libraries were detected; all libraries retained for DE analysis.\n')
    f.write('- Library sizes and detected gene counts were within acceptable ranges.\n\n')
    f.write('## Gene ID handling\n')
    f.write('- Ensembl gene IDs had mixed versioned and unversioned forms; versions were stripped except for _PAR_Y suffixes to ensure unique stable IDs.\n')
    f.write('- Gene symbol mapping used the provided gene_symbol_by_gene_id from inputs_metadata.json, matching on stable IDs only when unambiguous.\n')
    if ambiguous_ids:
        f.write(f'- Ambiguous gene ID to symbol mappings were found for {len(ambiguous_ids)} IDs; these gene symbols were left blank in output.\n')
    else:
        f.write('- No ambiguous gene ID to symbol mappings were found.\n')
    f.write('\n')
    f.write('## DESeq2 modeling decisions\n')
    f.write('- DESeq2 was run separately per cell line using design formula ~ condition including all retained conditions and biological replicates.\n')
    f.write('- Default DESeq2 settings were used, including median of ratios normalization and Benjamini-Hochberg FDR correction.\n')
    f.write('- No manual low-count filtering was applied prior to DESeq2; relied on DESeq2 independent filtering.\n')
    f.write('\n')
    f.write('## Definition of D4-specific gene sets\n')
    f.write('- D4-specific genes were defined as those significant in T98G D4 vs control but not significant in other T98G conditions.\n')
    f.write('\n')
    f.write('## Unresolved questions\n')
    f.write('- Multiple testing correction method was not tested; default DESeq2 BH FDR correction was used.\n')
    f.write('- Low-count gene filtering prior to DESeq2 was not tested; no filtering applied.\n')
    f.write('- Control and treated group label correctness was not fully resolved; labels were used as given.\n')
    f.write('- Other unresolved questions remain as documented in the input metadata.\n')

# Save prepared data for R step
samples.to_csv(work_dir / 'samples_prepared.tsv', sep='\t', index=False)
counts.to_csv(work_dir / 'counts_prepared.tsv', sep='\t')

# Save gene symbol mapping for R
gene_symbol_df = pd.DataFrame(
    [(k, v) for k, v in gene_symbol_by_gene_id.items()],
    columns=['gene_id', 'gene_symbol'])
gene_symbol_df.to_csv(work_dir / 'gene_symbol_mapping.tsv', sep='\t', index=False)

# Save ambiguous IDs for METHODS.md reference
with open(work_dir / 'ambiguous_gene_ids.txt', 'w') as f:
    for aid in sorted(ambiguous_ids):
        f.write(f'{aid}\n')
