import pandas as pd
import numpy as np
import json
import os

# Load sample sheet
samples = pd.read_csv('/data/samples.tsv', sep='\t')

# Load counts matrix
counts = pd.read_csv('/data/tagseq_featurecounts_matrix.tsv', sep='\t', index_col=0)

# Check that sample IDs in counts match sample sheet
assert set(samples['sample_id']) == set(counts.columns), "Sample IDs mismatch between counts and sample sheet"

# Check library sizes from profile
library_sizes = {s['sample_id']: s['library_size'] for s in [{'sample_id': s['sample_id'], 'library_size': s.get('library_size', None)} for s in samples.to_dict(orient='records')]}
# But profile already gives library sizes, use from profile
# Use profile from task input
profile = {"library_sizes": {"S01": 6246310, "S02": 2760487, "S03": 2098027, "S04": 2032928, "S05": 4578610, "S06": 4606376, "S07": 4100361, "S08": 5092791, "S09": 3711939, "S10": 4586059, "S11": 3083279, "S12": 3721873, "S13": 3798055, "S14": 6013006, "S15": 2737095, "S16": 3968618}} 

library_sizes = profile['library_sizes']

# Add library size to samples
samples['library_size'] = samples['sample_id'].map(library_sizes)

# Check for outliers in library size within each cell line
outlier_threshold_low = 0.1  # 10% of median
outlier_threshold_high = 10  # 10x median

methods_lines = []
methods_lines.append("# METHODS.md")
methods_lines.append("\n## Sample and Library Quality Checks")

for cell_line in samples['cell_line'].unique():
    cell_samples = samples[samples['cell_line'] == cell_line]
    median_lib_size = cell_samples['library_size'].median()
    low_cutoff = median_lib_size * outlier_threshold_low
    high_cutoff = median_lib_size * outlier_threshold_high
    low_outliers = cell_samples[cell_samples['library_size'] < low_cutoff]
    high_outliers = cell_samples[cell_samples['library_size'] > high_cutoff]
    if not low_outliers.empty or not high_outliers.empty:
        methods_lines.append(f"Cell line {cell_line} has library size outliers:")
        if not low_outliers.empty:
            methods_lines.append(f"  Low outliers: {list(low_outliers['sample_id'])}")
        if not high_outliers.empty:
            methods_lines.append(f"  High outliers: {list(high_outliers['sample_id'])}")
    else:
        methods_lines.append(f"Cell line {cell_line} has no library size outliers.")

# No samples flagged for exclusion by diagnostics, so keep all

# Strip version numbers from gene IDs in counts
# Preserve _PAR_Y suffix

def strip_version(gene_id):
    if gene_id.endswith('_PAR_Y'):
        return gene_id
    else:
        return gene_id.split('.')[0]

counts.index = counts.index.map(strip_version)

# Check for duplicates after stripping versions
if counts.index.duplicated().any():
    raise ValueError("Duplicate gene IDs after stripping versions")

# Save stripped counts for DESeq2 input
counts.to_csv('/work/stripped_counts.tsv', sep='\t')

# Save filtered samples for DESeq2 input
samples.to_csv('/work/filtered_samples.tsv', sep='\t', index=False)

# Save METHODS.md partial
with open('/outputs/METHODS.md', 'w') as f:
    f.write('\n'.join(methods_lines))
