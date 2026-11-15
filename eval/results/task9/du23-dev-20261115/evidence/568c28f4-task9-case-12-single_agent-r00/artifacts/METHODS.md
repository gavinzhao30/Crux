# Methods
Loaded samples.tsv and tagseq_featurecounts_matrix.tsv from /data.
Processed and filtered counts by cell line T98G and HEK293.
Used DESeq2 with design ~ condition for differential expression analysis.
Generated contrasts for all conditions versus control.
Removed gene version numbers except _PAR_Y from gene_id.
Gene symbol mapping not available due to missing inputs_metadata.json.
Saved DE results in required output files.
Extracted T98G D4-specific upregulated genes with padj < 0.05 and log2FoldChange > 0.
Shared DEGs gene_symbol assignment skipped due to missing gene symbol mapping.
Further diagnostics and issues noted in this METHODS file.
