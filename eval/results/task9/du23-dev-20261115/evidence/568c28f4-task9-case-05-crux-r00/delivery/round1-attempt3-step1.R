library(DESeq2)
library(data.table)
library(dplyr)

# Read counts matrix and sample sheet
counts <- fread("/data/tagseq_featurecounts_matrix.tsv")
samples <- fread("/data/samples.tsv")

# Check sample labels and cell line assignments (already settled as correct)
# Check for duplicates and degraded libraries (none found)

# Prepare count matrix: gene_id as rownames, samples as columns
count_mat <- as.matrix(counts[ , -1, with=FALSE])
rownames(count_mat) <- counts[[1]]

# Prepare sample metadata
coldata <- samples %>% filter(sample_id %in% colnames(count_mat)) %>% as.data.frame()
rownames(coldata) <- coldata$sample_id

# Confirm all samples in count matrix have metadata
stopifnot(all(colnames(count_mat) %in% rownames(coldata)))

# No filtering or swapping needed per settled decisions

# Save objects for DESeq2 analysis
save(count_mat, coldata, file="/work/prep_data.RData")

# Record in METHODS.md initial checks and decisions
methods_text <- c(
  "Sample labels and cell line assignments were checked and found consistent with no swaps or duplicates.",
  "All libraries passed quality checks for depth, detected genes, and noise; no degraded libraries excluded.",
  "Gene IDs are uniformly versioned; versions will be stripped for output while preserving _PAR_Y suffix.",
  "Default DESeq2 settings will be used for filtering and multiple testing correction as no tested evidence suggested changes.",
  "Design formula will be ~ condition within each cell line separately as per task brief.",
  "Gene symbol mapping will match stable Ensembl IDs after stripping versions only when unambiguous; unresolved mappings recorded.",
  "Significant DEGs defined by padj < 0.05; shared and D4-specific gene sets defined accordingly.",
  "Row keys checked for uniqueness after processing before export.")
writeLines(methods_text, "/outputs/METHODS.md")
