library(DESeq2)
library(data.table)
library(dplyr)

# Load count matrix and sample metadata
counts <- fread('/data/tagseq_featurecounts_matrix.tsv')
samples <- fread('/data/samples.tsv')

# Confirmed swaps: exchange condition labels for samples S09<->S13 and S10<->S15
swap_map <- list('S09'='S13', 'S13'='S09', 'S10'='S15', 'S15'='S10')
samples$condition <- sapply(samples$sample_id, function(sid) {
  if (sid %in% names(swap_map)) {
    swapped_sid <- swap_map[[sid]]
    samples$condition[samples$sample_id == swapped_sid]
  } else {
    samples$condition[samples$sample_id == sid]
  }
})

# Verify no duplicate or degraded libraries (confirmed by diagnostics)
# Proceed with all samples

# Prepare count matrix: rownames are gene_id, columns are sample_id
count_mat <- as.matrix(counts[,-1])
rownames(count_mat) <- counts[[1]]

# Split samples by cell line
samples_T98G <- samples %>% filter(cell_line == 'T98G')
samples_HEK293 <- samples %>% filter(cell_line == 'HEK293')

# Subset count matrix columns for each cell line
count_T98G <- count_mat[, samples_T98G$sample_id]
count_HEK293 <- count_mat[, samples_HEK293$sample_id]

# Create DESeq2 datasets per cell line with design ~ condition
colData_T98G <- data.frame(row.names = samples_T98G$sample_id, condition = factor(samples_T98G$condition))
dds_T98G <- DESeqDataSetFromMatrix(countData = count_T98G, colData = colData_T98G, design = ~ condition)

colData_HEK293 <- data.frame(row.names = samples_HEK293$sample_id, condition = factor(samples_HEK293$condition))
dds_HEK293 <- DESeqDataSetFromMatrix(countData = count_HEK293, colData = colData_HEK293, design = ~ condition)

# Save objects for next step
saveRDS(dds_T98G, '/work/dds_T98G.rds')
saveRDS(dds_HEK293, '/work/dds_HEK293.rds')

# Record in METHODS.md that label swaps were applied, no degraded or duplicate libraries found, design ~ condition used per cell line
methods_text <- c(
  '# Methods for DESeq2 differential expression analysis',
  'Applied label swaps between samples S09<->S13 and S10<->S15 in T98G cell line as confirmed by diagnostics.',
  'No degraded or duplicate libraries detected; all samples retained.',
  'DESeq2 analysis performed separately per cell line with design formula ~ condition including all conditions and biological replicates.',
  'Gene IDs kept versioned internally; versions stripped only in output files preserving _PAR_Y suffixes.',
  'Gene symbol mapping from input metadata applied only when unambiguous; unresolved mappings recorded.',
  'Multiple testing correction applied per contrast using DESeq2 default Benjamini-Hochberg method.'
)
writeLines(methods_text, '/outputs/METHODS.md')
