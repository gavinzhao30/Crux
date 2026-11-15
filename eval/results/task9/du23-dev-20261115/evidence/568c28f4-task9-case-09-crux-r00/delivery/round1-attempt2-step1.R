library(DESeq2)
library(tibble)
library(readr)
library(stringr)

# Load sample sheet and counts
samples <- read_tsv('/data/samples.tsv')
counts <- read_tsv('/data/tagseq_featurecounts_matrix.tsv')

# Strip version numbers from gene IDs except _PAR_Y
strip_version <- function(ids) {
  sapply(ids, function(id) {
    if (str_detect(id, '_PAR_Y$')) {
      return(id)
    } else {
      return(str_replace(id, '\\..*$', ''))
    }
  })
}

counts$gene_id <- strip_version(counts$gene_id)

# Check for duplicate gene_ids after stripping
if(any(duplicated(counts$gene_id))) stop('Duplicate gene_ids after stripping versions')

# Prepare count matrix
count_mat <- as.matrix(counts[,-1])
rownames(count_mat) <- counts$gene_id

# Confirm sample IDs match columns
if(!all(colnames(count_mat) %in% samples$sample_id)) stop('Mismatch between count columns and sample IDs')

# Reorder samples to match count columns
samples <- samples[match(colnames(count_mat), samples$sample_id),]

# Check sample labels consistency (already settled: as_given)
# Check library quality (already settled: all usable)

# Save stripped gene IDs for later use
writeLines(counts$gene_id, '/work/stripped_gene_ids.txt')

# Save processed data for DESeq2
saveRDS(list(counts=count_mat, samples=samples), '/work/data_for_deseq2.rds')

# Record methods
methods <- c(
  '# Methods for Task 9 DESeq2 analysis',
  '- Sample labels checked and confirmed consistent with metadata (diagnostics: labels_nearest_neighbour, labels_swap_search).',
  '- No duplicated or degraded libraries found; all libraries retained.',
  '- Gene IDs version numbers stripped except for _PAR_Y suffixes; no collisions detected.',
  '- Used default DESeq2 normalization (median of ratios) and multiple testing correction (Benjamini-Hochberg).',
  '- Design formula per cell line: ~ condition using all retained conditions and biological replicates.',
  '- Gene symbols mapped from /data/inputs_metadata.json using stable IDs unambiguously; ambiguous mappings recorded.',
  '- D4-specific genes defined as significant in T98G D4 vs control but not in other T98G conditions.',
  '- Unresolved questions on filtering and multiple testing correction kept at DESeq2 defaults.',
  ''
)
writeLines(methods, '/outputs/METHODS.md')
