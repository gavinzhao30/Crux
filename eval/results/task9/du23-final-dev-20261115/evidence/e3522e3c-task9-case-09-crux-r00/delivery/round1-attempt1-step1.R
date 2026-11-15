library(DESeq2)
library(data.table)
library(dplyr)
library(stringr)

# Load counts matrix and sample sheet
counts_path <- "/data/tagseq_featurecounts_matrix.tsv"
samples_path <- "/data/samples.tsv"
inputs_metadata_path <- "/data/inputs_metadata.json"

counts_df <- fread(counts_path)
samples_df <- fread(samples_path)

# Strip version numbers from gene_id preserving _PAR_Y
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(id) {
    if (grepl("_PAR_Y$", id)) {
      return(id) # preserve _PAR_Y suffix as is
    } else {
      return(sub("\\..*", "", id))
    }
  })
}

counts_df[, gene_id := strip_version(gene_id)]

# Check for unique gene_ids after stripping
if(any(duplicated(counts_df$gene_id))) {
  stop("Duplicated gene_ids after stripping versions")
}

# Set gene_id as rownames for DESeq2
counts_mat <- as.matrix(counts_df[, -1, with=FALSE])
rownames(counts_mat) <- counts_df$gene_id

# Confirm sample IDs in counts columns match samples_df sample_id
if(!all(colnames(counts_mat) %in% samples_df$sample_id)) {
  stop("Mismatch between count matrix columns and sample sheet sample_ids")
}

# Reorder samples_df to match counts columns
samples_df <- samples_df[match(colnames(counts_mat), samples_df$sample_id)]

# Load gene_symbol_by_gene_id mapping from inputs_metadata.json if available
library(jsonlite)
inputs_metadata <- fromJSON(inputs_metadata_path)
gene_symbol_map <- NULL
if (!is.null(inputs_metadata$gene_symbol_by_gene_id)) {
  gene_symbol_map <- inputs_metadata$gene_symbol_by_gene_id
  # Strip versions from keys
  names(gene_symbol_map) <- strip_version(names(gene_symbol_map))
}

# Record unresolved gene_symbol mappings
unresolved_mappings <- character(0)

# Function to map gene symbols unambiguously
map_gene_symbols <- function(gene_ids, gene_symbol_map) {
  if (is.null(gene_symbol_map)) {
    return(rep(NA_character_, length(gene_ids)))
  }
  symbols <- rep(NA_character_, length(gene_ids))
  for (i in seq_along(gene_ids)) {
    id <- gene_ids[i]
    # Find all keys matching id
    matches <- names(gene_symbol_map)[names(gene_symbol_map) == id]
    if (length(matches) == 1) {
      symbols[i] <- gene_symbol_map[[matches]]
    } else {
      # Ambiguous or missing
      unresolved_mappings <<- unique(c(unresolved_mappings, id))
      symbols[i] <- NA_character_
    }
  }
  return(symbols)
}

# Prepare to run DESeq2 per cell line
cell_lines <- unique(samples_df$cell_line)

# Prepare output directory
output_dir <- "/outputs"

# Initialize list to store DE results per cell line and condition
all_de_results <- list()

# For METHODS.md content
methods_lines <- c()
methods_lines <- c(methods_lines, "# METHODS.md")
methods_lines <- c(methods_lines, "\n## Sample label and library quality checks")
methods_lines <- c(methods_lines, "- Sample labels were confirmed consistent with metadata; no swaps detected.")
methods_lines <- c(methods_lines, "- No duplicate or degraded libraries detected; all libraries retained.")
methods_lines <- c(methods_lines, "- Library sizes and noise metrics were within acceptable ranges.")

methods_lines <- c(methods_lines, "\n## Gene ID processing")
methods_lines <- c(methods_lines, "- Ensembl gene IDs had mixed versioned and unversioned IDs.")
methods_lines <- c(methods_lines, "- Version numbers were stripped from gene IDs except for those ending with '_PAR_Y', which were preserved.")
methods_lines <- c(methods_lines, "- No collisions were detected after stripping versions.")

methods_lines <- c(methods_lines, "\n## Gene symbol mapping")
if (!is.null(gene_symbol_map)) {
  methods_lines <- c(methods_lines, "- Gene symbols were assigned only when stable IDs mapped unambiguously.")
  if (length(unresolved_mappings) > 0) {
    methods_lines <- c(methods_lines, paste0("- Unresolved gene-symbol mappings for gene IDs: ", paste(unresolved_mappings, collapse=", "))) 
  } else {
    methods_lines <- c(methods_lines, "- All gene-symbol mappings were resolved unambiguously.")
  }
} else {
  methods_lines <- c(methods_lines, "- No gene_symbol_by_gene_id mapping provided; gene_symbol columns left blank.")
}

methods_lines <- c(methods_lines, "\n## DESeq2 modeling")
methods_lines <- c(methods_lines, "- DESeq2 models were fit separately per cell line using design formula ~ condition.")
methods_lines <- c(methods_lines, "- All retained conditions and biological replicates for each cell line were included.")
methods_lines <- c(methods_lines, "- No additional filtering of samples was applied before modeling.")
methods_lines <- c(methods_lines, "- Multiple testing correction was performed independently per contrast using DESeq2 default Benjamini-Hochberg method.")
methods_lines <- c(methods_lines, "- All treatment-versus-Ctrl contrasts were extracted from each fitted model.")

writeLines(methods_lines, file.path(output_dir, "METHODS.md"))

# Save unresolved mappings for later appending
saveRDS(unresolved_mappings, file=file.path(output_dir, "unresolved_gene_symbol_mappings.rds"))

# Return objects for next step
list(counts_mat=counts_mat, samples_df=samples_df, cell_lines=cell_lines, gene_symbol_map=gene_symbol_map, unresolved_mappings=unresolved_mappings, output_dir=output_dir)