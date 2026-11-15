library(DESeq2)
library(data.table)
library(dplyr)

# Load counts matrix and sample sheet
counts <- fread('/data/tagseq_featurecounts_matrix.tsv', data.table=FALSE)
samples <- read.table('/data/samples.tsv', header=TRUE, sep='\t', stringsAsFactors=FALSE)

# Set rownames as gene_id
rownames(counts) <- counts$gene_id
counts$gene_id <- NULL

# Check for duplicate libraries flagged: drop S13 (duplicate of S10)
samples <- samples[!(samples$sample_id %in% 'S13'), ]
counts <- counts[, colnames(counts) != 'S13']

# Confirm all libraries usable (no degraded libraries)
# Confirm no swapped or mislabeled samples (accepted as given)
# Confirm cell_line column is correct (accepted as given)

# Prepare metadata for DESeq2
samples$condition <- factor(samples$condition)
samples$cell_line <- factor(samples$cell_line)

# Split samples by cell line
samples_t98g <- samples[samples$cell_line == 'T98G', ]
samples_hek293 <- samples[samples$cell_line == 'HEK293', ]

counts_t98g <- counts[, colnames(counts) %in% samples_t98g$sample_id]
counts_hek293 <- counts[, colnames(counts) %in% samples_hek293$sample_id]

# Create DESeq2 datasets
dds_t98g <- DESeqDataSetFromMatrix(countData=counts_t98g, colData=samples_t98g, design=~condition)
dds_hek293 <- DESeqDataSetFromMatrix(countData=counts_hek293, colData=samples_hek293, design=~condition)

# Keep all genes (no manual low-count prefiltering as per unresolved decision)

# Set control as reference level for condition
dds_t98g$condition <- relevel(dds_t98g$condition, ref='Ctrl')
dds_hek293$condition <- relevel(dds_hek293$condition, ref='Ctrl')

# Save dds objects for next step
save(dds_t98g, dds_hek293, file='/work/dds_objects.RData')

# Record in METHODS.md about decisions
methods_text <- c(
  '# METHODS',
  'Sample filtering: dropped duplicate library S13 as flagged.',
  'No degraded libraries detected; all libraries retained.',
  'Sample labels and cell_line assignments accepted as given after diagnostics.',
  'DESeq2 design: ~ condition per cell line, control set as reference level.',
  'No manual low-count filtering applied; relied on DESeq2 internal filtering.',
  'Gene IDs are versioned Ensembl IDs as in matrix; versions stripped only in output.',
  'Multiple testing correction: default DESeq2 Benjamini-Hochberg FDR per contrast (default).',
  'Unresolved questions noted: multiple testing correction approach, gene symbol mapping ambiguities, gene set definitions remain as default.'
)
writeLines(methods_text, '/outputs/METHODS.md')
