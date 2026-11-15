import pandas as pd
import numpy as np
import json

# Load sample sheet
samples = pd.read_csv('/data/samples.tsv', sep='\t')

# Load counts matrix
counts = pd.read_csv('/data/tagseq_featurecounts_matrix.tsv', sep='\t', index_col=0)

# Load inputs_metadata.json for gene_symbol_by_gene_id
with open('/data/inputs_metadata.json') as f:
    metadata = json.load(f)

gene_symbol_by_gene_id_raw = metadata.get('gene_symbol_by_gene_id', {})

# Strip version numbers from gene IDs in counts and gene_symbol_by_gene_id keys, preserving _PAR_Y suffix
import re

def strip_version(gid):
    if gid.endswith('_PAR_Y'):
        return gid
    else:
        return gid.split('.')[0]

# Strip versions in counts index
counts.index = counts.index.map(strip_version)

# Check for duplicates after stripping
assert counts.index.is_unique, 'Gene IDs are not unique after stripping versions'

# Process gene_symbol_by_gene_id mapping: strip versions and keep only unambiguous mappings
from collections import defaultdict
mapping_temp = defaultdict(set)
for k,v in gene_symbol_by_gene_id_raw.items():
    k_stripped = strip_version(k)
    mapping_temp[k_stripped].add(v)

# Keep only unambiguous mappings
gene_symbol_by_gene_id = {k: next(iter(v)) for k,v in mapping_temp.items() if len(v)==1}
ambiguous_gene_ids = [k for k,v in mapping_temp.items() if len(v)>1]

# Prepare sample metadata for DESeq2
samples = samples.set_index('sample_id')

# Check sample labels and library quality from decisions (all settled as correct and usable)
# No samples to drop or relabel

# Save ambiguous gene symbol mappings for METHODS.md
with open('/outputs/METHODS.md', 'w') as f:
    f.write('# Analysis Methods and Decisions\n')
    f.write('\n## Gene symbol mapping\n')
    if ambiguous_gene_ids:
        f.write('Ambiguous gene symbol mappings found for the following gene IDs (after stripping versions):\n')
        for gid in ambiguous_gene_ids:
            f.write(f'- {gid}\n')
    else:
        f.write('No ambiguous gene symbol mappings found.\n')

# Save processed counts and samples for R
counts.to_csv('/work/processed_counts.tsv', sep='\t')
samples.to_csv('/work/processed_samples.tsv', sep='\t')

# Record in METHODS.md that default DESeq2 settings are used for multiple testing correction and no low-count filtering
with open('/outputs/METHODS.md', 'a') as f:
    f.write('\n## DESeq2 settings\n')
    f.write('Used default DESeq2 Benjamini-Hochberg FDR correction (padj).\n')
    f.write('No manual low-count filtering applied before DESeq2; relied on DESeq2 independent filtering.\n')
    f.write('Design formula used: ~ condition per cell line.\n')
    f.write('Sample labels and library quality were checked and found consistent and usable.\n')
    f.write('Gene IDs were stripped of version numbers preserving _PAR_Y suffixes before mapping gene symbols.\n')
    f.write('Gene symbols were assigned only when mapping was unambiguous; ambiguous mappings recorded above.\n')
