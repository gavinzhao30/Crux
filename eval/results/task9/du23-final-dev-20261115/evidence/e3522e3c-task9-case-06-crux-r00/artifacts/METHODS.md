# Methods for Task 9 Differential Expression Analysis

## Sample and library quality control
- No swapped or mislabeled samples detected; condition labels used as given.
- Duplicate library pair detected: S10 and S13 identical; dropped S13.
- All libraries passed quality filters; no degraded libraries excluded.
- Cell line assignments confirmed correct.

## Gene ID handling
- Gene IDs in counts matrix are uniformly versioned Ensembl IDs; versions will be stripped in output files while preserving _PAR_Y suffix.

## DESeq2 analysis approach
- DESeq2 models fitted separately per cell line with design ~ condition, using all retained conditions and biological replicates.
- Default DESeq2 settings used, including internal normalization and Benjamini-Hochberg FDR correction per contrast.
- Control condition set as reference level in condition factor for contrasts.

## Gene symbol mapping
- No gene_symbol_by_gene_id mapping provided; gene symbols will be assigned as empty strings in outputs.
