library(DESeq2)
library(tidyverse)
library(data.table)

# Read sample sheet and counts
samples <- fread('/data/samples.tsv')
counts <- fread('/data/tagseq_featurecounts_matrix.tsv')

# Fix swapped condition labels in T98G samples S09/S13 and S10/S15 as per settled decision
samples_fixed <- samples %>% mutate(condition = case_when(
  sample_id == 'S09' ~ samples$condition[samples$sample_id == 'S13'],
  sample_id == 'S13' ~ samples$condition[samples$sample_id == 'S09'],
  sample_id == 'S10' ~ samples$condition[samples$sample_id == 'S15'],
  sample_id == 'S15' ~ samples$condition[samples$sample_id == 'S10'],
  TRUE ~ condition
))

# Prepare gene count matrix with gene_id as rownames
count_mat <- as.matrix(counts[ , -1, with=FALSE])
rownames(count_mat) <- counts[[1]]

# Store fixed samples for downstream
samples <- samples_fixed

# Save intermediate objects for next step
saveRDS(samples, '/work/samples_fixed.rds')
saveRDS(count_mat, '/work/count_mat.rds')
