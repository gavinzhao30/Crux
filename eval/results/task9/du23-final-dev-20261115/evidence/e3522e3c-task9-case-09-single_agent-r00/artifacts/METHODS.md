# Task 9 Downstream Analysis Methods
- Loaded count matrix and sample sheet from /data directory.
- Removed version suffix from Ensembl IDs and aggregated duplicated genes by summing counts.
- Converted condition to factors with 'Ctrl' as baseline.
- Fitted DESeq2 models per cell line with design ~ condition.
- Extracted treatment versus Ctrl contrasts using coefficient-specific LFC shrinkage (apeglm).
- Created T98G shared DEG gene sets by intersecting significant genes with consistent regulation.
- Generated T98G D4-specific upregulated genes excluding genes significant in other treatments.
- gene_symbol columns are NA due to absent mapping metadata.
- Inspected library sizes as surrogate for quality control.
- No unresolved sample label mismatches detected.
- Unresolved: Missing gene_symbol mapping data.

