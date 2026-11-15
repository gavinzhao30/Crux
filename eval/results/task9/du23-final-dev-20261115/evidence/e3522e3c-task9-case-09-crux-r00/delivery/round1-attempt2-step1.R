library(DESeq2)
library(readr)
library(stringr)
library(dplyr)

# Read sample sheet and counts matrix
samples <- read_tsv("/data/samples.tsv")
counts <- read_tsv("/data/tagseq_featurecounts_matrix.tsv")

# Strip version numbers from gene_id, preserving _PAR_Y suffix
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(id) {
    if (str_detect(id, "_PAR_Y")) {
      # preserve _PAR_Y suffix, strip version before it
      parts <- str_match(id, "^(ENSG\d+)(?:\\.\d+)?(_PAR_Y)$")
      if (!is.na(parts[1,1])) {
        return(paste0(parts[1,2], parts[1,3]))
      } else {
        return(id) # fallback
      }
    } else {
      # strip version after dot
      return(str_replace(id, "\\.\\d+$", ""))
    }
  }, USE.NAMES=FALSE)
}

counts$gene_id <- strip_version(counts$gene_id)

# Check uniqueness of gene_id after stripping
if(any(duplicated(counts$gene_id))) stop("Duplicated gene_id after stripping versions")

# Prepare counts matrix for DESeq2
count_mat <- counts %>% select(-gene_symbol) %>% column_to_rownames("gene_id")

# Read gene_symbol_by_gene_id mapping if available
mapping_path <- "/data/inputs_metadata.json"
if (file.exists(mapping_path)) {
  library(jsonlite)
  meta <- fromJSON(mapping_path)
  if (!is.null(meta$gene_symbol_by_gene_id)) {
    gene_symbol_map <- meta$gene_symbol_by_gene_id
    # Strip versions from keys
    names(gene_symbol_map) <- strip_version(names(gene_symbol_map))
    # Check for ambiguous mappings
    ambiguous_ids <- names(which(table(names(gene_symbol_map)) > 1))
    unresolved_mappings <- ambiguous_ids[ambiguous_ids %in% names(gene_symbol_map)]
  } else {
    gene_symbol_map <- NULL
    unresolved_mappings <- character(0)
  }
} else {
  gene_symbol_map <- NULL
  unresolved_mappings <- character(0)
}

# Filter samples: use all samples as is (no filtering) per unresolved decision
# Confirm sample labels and library quality settled as accepted

# Prepare sample metadata for DESeq2
samples <- samples %>% filter(sample_id %in% colnames(count_mat))

# Confirm cell_line column matches libraries (accepted as_given)
# Confirm no duplicates or degraded libraries (accepted all_usable)

# Split samples by cell line
cell_lines <- unique(samples$cell_line)

# Save variables for downstream steps
saveRDS(list(count_mat=count_mat, samples=samples, gene_symbol_map=gene_symbol_map, unresolved_mappings=unresolved_mappings), "/work/prep_data.rds")
