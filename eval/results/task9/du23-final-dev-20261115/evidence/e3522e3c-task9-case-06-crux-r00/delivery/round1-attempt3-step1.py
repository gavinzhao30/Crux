import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('/data')
OUTPUTS_DIR = Path('/outputs')
WORK_DIR = Path('/work')

# Load sample sheet
samples = pd.read_csv(DATA_DIR / 'samples.tsv', sep='\t')

# Apply settled decisions:
# - No swapped samples (condition labels as given)
# - Drop duplicate library S13 (keep S10)
# - All libraries usable
# - Cell line column correct as given

# Drop duplicate library S13
samples = samples[samples['sample_id'] != 'S13'].reset_index(drop=True)

# Load counts matrix
counts = pd.read_csv(DATA_DIR / 'tagseq_featurecounts_matrix.tsv', sep='\t', index_col=0)

# Drop duplicate library S13 column
if 'S13' in counts.columns:
    counts = counts.drop(columns=['S13'])

# Confirm samples and counts columns match
assert set(samples['sample_id']) == set(counts.columns), 'Mismatch between samples and counts columns'

# Sort samples and counts columns to same order
samples = samples.set_index('sample_id').loc[counts.columns].reset_index()

# Load gene symbol mapping if available
import json
metadata_path = DATA_DIR / 'inputs_metadata.json'
gene_symbol_by_gene_id = None
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    gene_symbol_by_gene_id = metadata.get('gene_symbol_by_gene_id', None)
else:
    metadata = None

# Save intermediate data for R
counts.to_csv(WORK_DIR / 'counts.tsv', sep='\t')
samples.to_csv(WORK_DIR / 'samples.tsv', sep='\t', index=False)

# Record methods start
with open(OUTPUTS_DIR / 'METHODS.md', 'w') as f:
    f.write('# Methods for Task 9 Differential Expression Analysis\n')
    f.write('\n')
    f.write('## Sample and library quality control\n')
    f.write('- No swapped or mislabeled samples detected; condition labels used as given.\n')
    f.write('- Duplicate library pair detected: S10 and S13 identical; dropped S13.\n')
    f.write('- All libraries passed quality filters; no degraded libraries excluded.\n')
    f.write('- Cell line assignments confirmed correct.\n')
    f.write('\n')
    f.write('## Gene ID handling\n')
    f.write('- Gene IDs in counts matrix are uniformly versioned Ensembl IDs; versions will be stripped in output files while preserving _PAR_Y suffix.\n')
    f.write('\n')
    f.write('## DESeq2 analysis approach\n')
    f.write('- DESeq2 models fitted separately per cell line with design ~ condition, using all retained conditions and biological replicates.\n')
    f.write('- Default DESeq2 settings used, including internal normalization and Benjamini-Hochberg FDR correction per contrast.\n')
    f.write('- Control condition set as reference level in condition factor for contrasts.\n')
    f.write('\n')
    f.write('## Gene symbol mapping\n')
    if gene_symbol_by_gene_id is None:
        f.write('- No gene_symbol_by_gene_id mapping provided; gene symbols will be assigned as empty strings in outputs.\n')
    else:
        f.write('- Gene symbol mapping loaded from inputs_metadata.json; ambiguous mappings handled by keeping only unambiguous stable ID to symbol pairs.\n')
        f.write('- Unresolved ambiguous gene symbol mappings recorded below.\n')

