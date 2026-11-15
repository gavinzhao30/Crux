library(DESeq2)
library(data.table)
library(dplyr)
library(stringr)
library(jsonlite)

# Load counts matrix and sample sheet
counts <- fread('/data/tagseq_featurecounts_matrix.tsv')
samples <- fread('/data/samples.tsv')

# Convert counts to matrix with gene_id as rownames
gene_ids <- counts[[1]]
count_mat <- as.matrix(counts[,-1, with=FALSE])
rownames(count_mat) <- gene_ids

# Strip version numbers from gene IDs except those ending with '_PAR_Y'
strip_version <- function(gid) {
  if (grepl('_PAR_Y$', gid)) {
    return(gid)
  } else {
    return(sub('\\..*$', '', gid))
  }
}

unversioned_gene_ids <- sapply(rownames(count_mat), strip_version)

# Check for collisions after stripping versions
if(any(duplicated(unversioned_gene_ids))) {
  stop('Gene ID collisions after stripping versions')
}
rownames(count_mat) <- unversioned_gene_ids

# Load gene_symbol_by_gene_id mapping from inputs_metadata.json
metadata <- fromJSON('/data/inputs_metadata.json')
if (!is.null(metadata$gene_symbol_by_gene_id)) {
  gene_symbol_map <- metadata$gene_symbol_by_gene_id
  # Strip versions in keys
  names(gene_symbol_map) <- sapply(names(gene_symbol_map), strip_version)
  # Identify ambiguous mappings
  ambiguous_ids <- names(which(table(names(gene_symbol_map)) > 1))
  # Keep only unambiguous mappings
  unambiguous_map <- gene_symbol_map[!names(gene_symbol_map) %in% ambiguous_ids]
} else {
  unambiguous_map <- list()
  ambiguous_ids <- character(0)
}

# Prepare sample metadata
samples$condition <- factor(samples$condition)
samples$cell_line <- factor(samples$cell_line)

# Check sample labels and library quality (already settled: no swaps, no duplicates, all usable, correct cell_line)
# So proceed with all samples

# Split data by cell line
cell_lines <- unique(samples$cell_line)

# Save variables for later use
save(count_mat, samples, unambiguous_map, ambiguous_ids, cell_lines, file='/work/prep_data.RData')

# Write METHODS.md initial content
methods <- c(
  '# METHODS',
  'Sample labels were checked and found consistent with no swaps or misannotations.',
  'No duplicated or degraded libraries were detected; all libraries were retained.',
  'Gene IDs were stripped of Ensembl version numbers except those ending with _PAR_Y, with no collisions detected.',
  'Gene symbol mapping used unversioned stable IDs; ambiguous mappings were left blank and recorded here.',
  paste('Ambiguous gene ID mappings:', paste(ambiguous_ids, collapse=', '))
)
writeLines(methods, '/outputs/METHODS.md')
