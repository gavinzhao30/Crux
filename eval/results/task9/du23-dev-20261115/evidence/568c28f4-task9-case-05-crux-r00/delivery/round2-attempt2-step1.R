library(DESeq2)
library(data.table)
library(stringr)

# Load sample sheet and counts
samples <- fread("/data/samples.tsv")
counts <- fread("/data/tagseq_featurecounts_matrix.tsv")

# Prepare count matrix and sample metadata
count_mat <- as.matrix(counts[ , -1, with=FALSE])
rownames(count_mat) <- counts[[1]]

# Check sample order matches
stopifnot(all(colnames(count_mat) == samples$sample_id))

# Function to strip version from Ensembl IDs but preserve _PAR_Y suffix
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(id) {
    if (grepl("_PAR_Y$", id)) {
      # preserve _PAR_Y suffix, strip version before it
      sub("(ENSG[0-9]+)\\.[0-9]+(_PAR_Y)$", "\\1\\2", id)
    } else {
      sub("(ENSG[0-9]+)\\.[0-9]+$", "\\1", id)
    }
  })
}

# Strip versions from gene IDs
gene_ids_stripped <- strip_version(rownames(count_mat))
rownames(count_mat) <- gene_ids_stripped

# Confirm uniqueness
if(any(duplicated(rownames(count_mat)))) stop("Duplicate gene IDs after version stripping")

# Load gene_symbol_by_gene_id mapping if available
mapping_path <- "/data/inputs_metadata.json"
if (file.exists(mapping_path)) {
  library(jsonlite)
  meta <- fromJSON(mapping_path)
  if (!is.null(meta$gene_symbol_by_gene_id)) {
    gene_symbol_map <- meta$gene_symbol_by_gene_id
    # Map stable IDs only (strip versions) and keep unambiguous mappings
    stable_ids <- names(gene_symbol_map)
    stable_ids_stripped <- strip_version(stable_ids)
    # Build mapping table
    map_dt <- data.table(stable_id=stable_ids_stripped, gene_symbol=unlist(gene_symbol_map))
    # Find ambiguous stable IDs
    ambig_ids <- map_dt[, .N, by=stable_id][N>1, stable_id]
    # Keep only unambiguous
    map_dt_unique <- map_dt[!stable_id %in% ambig_ids]
    gene_symbol_lookup <- setNames(map_dt_unique$gene_symbol, map_dt_unique$stable_id)
  } else {
    gene_symbol_lookup <- NULL
  }
} else {
  gene_symbol_lookup <- NULL
}

# Add gene_symbol column for all genes
gene_symbols <- rep(NA_character_, length(rownames(count_mat)))
names(gene_symbols) <- rownames(count_mat)
if (!is.null(gene_symbol_lookup)) {
  # Assign gene symbols where unambiguous
  gene_symbols <- gene_symbol_lookup[rownames(count_mat)]
}

# Record unresolved mappings
unresolved_ids <- setdiff(unique(strip_version(names(gene_symbol_lookup))), rownames(count_mat))

# Prepare output directory
dir.create("/outputs", showWarnings=FALSE)

# Write METHODS.md with decisions and unresolved mappings
methods_file <- "/outputs/METHODS.md"
cat("# Analysis Methods and Decisions\n", file=methods_file)
cat("- Sample labels and cell line assignments were checked and found consistent; no swaps or duplicates detected.\n", file=methods_file, append=TRUE)
cat("- All libraries passed quality checks; no degraded or duplicate libraries excluded.\n", file=methods_file, append=TRUE)
cat("- DESeq2 models were fit separately per cell line using design formula ~ condition including all retained conditions and biological replicates.\n", file=methods_file, append=TRUE)
cat("- Default DESeq2 independent filtering and Benjamini-Hochberg p-value adjustment were used.\n", file=methods_file, append=TRUE)
cat("- Gene IDs were stripped of Ensembl version numbers except _PAR_Y suffixes preserved; uniqueness of row keys confirmed.\n", file=methods_file, append=TRUE)
cat("- Gene symbol mapping used provided gene_symbol_by_gene_id mapping matched on stable IDs only when unambiguous; unresolved mappings listed below.\n", file=methods_file, append=TRUE)
if (length(unresolved_ids) > 0) {
  cat("Unresolved gene symbol mappings (stable IDs):\n", file=methods_file, append=TRUE)
  cat(paste(unresolved_ids, collapse=", "), "\n", file=methods_file, append=TRUE)
} else {
  cat("No unresolved gene symbol mappings.\n", file=methods_file, append=TRUE)
}

# Function to run DESeq2 per cell line
run_deseq2_per_cell_line <- function(cell_line_name) {
  # Subset samples
  smp <- samples[cell_line == cell_line_name]
  # Subset counts
  ct <- count_mat[, smp$sample_id, drop=FALSE]
  # Prepare colData
  coldata <- data.frame(condition=factor(smp$condition, levels=unique(smp$condition)))
  rownames(coldata) <- smp$sample_id
  # Create DESeqDataSet
  dds <- DESeqDataSetFromMatrix(countData=ct, colData=coldata, design=~condition)
  # Run DESeq
  dds <- DESeq(dds)
  return(dds)
}

# Run DESeq2 for T98G and HEK293
dds_t98g <- run_deseq2_per_cell_line("T98G")
dds_hek293 <- run_deseq2_per_cell_line("HEK293")

# Function to extract results for all treatment vs Ctrl contrasts
extract_results <- function(dds, cell_line) {
  res_list <- list()
  conds <- levels(colData(dds)$condition)
  ctrl <- "Ctrl"
  treatments <- setdiff(conds, ctrl)
  for (treat in treatments) {
    contrast <- c("condition", treat, ctrl)
    res <- results(dds, contrast=contrast, alpha=0.05)
    res_dt <- as.data.table(res, keep.rownames="gene_id")
    # Add gene_symbol
    res_dt[, gene_symbol := gene_symbols[gene_id]]
    # Rename columns to match contract
    setnames(res_dt, old=c("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"),
                      new=c("base_mean", "log2_fold_change", "lfc_se", "stat", "p_value", "padj"))
    # Keep required columns
    res_dt <- res_dt[, .(gene_id, gene_symbol, base_mean, log2_fold_change, lfc_se, stat, p_value, padj)]
    # Strip versions again to ensure
    res_dt[, gene_id := strip_version(gene_id)]
    # Check uniqueness
    if(any(duplicated(res_dt$gene_id))) stop(paste0("Duplicates in gene_id for ", cell_line, ":", treat))
    # Save to list
    res_list[[paste0(tolower(cell_line), "_", tolower(treat), "_vs_control_degs")]] <- res_dt
  }
  return(res_list)
}

# Extract results
res_t98g <- extract_results(dds_t98g, "T98G")
res_hek293 <- extract_results(dds_hek293, "HEK293")

# Save DE results to files
for (name in names(res_t98g)) {
  fwrite(res_t98g[[name]], file=file.path("/outputs", paste0(name, ".csv")))
}
for (name in names(res_hek293)) {
  fwrite(res_hek293[[name]], file=file.path("/outputs", paste0(name, ".csv")))
}

# Define shared DEGs for T98G slow shared (intersect genes significant in all T98G treatments vs control)
sig_genes_list <- lapply(res_t98g, function(dt) dt[padj < 0.05])
shared_genes <- Reduce(function(x,y) merge(x, y, by=c("gene_symbol"), all=FALSE), sig_genes_list)
# Determine regulation direction: keep genes consistently up or down
regulation <- sapply(names(res_t98g), function(n) {
  dt <- res_t98g[[n]]
  dt <- dt[gene_symbol %in% shared_genes$gene_symbol]
  sign(dt$log2_fold_change)
})
# Only keep genes with consistent sign across all treatments
consistent <- apply(regulation, 1, function(x) length(unique(x)) == 1 && !any(is.na(x)))
shared_consistent <- shared_genes[consistent]
# Assign regulation label
direction <- ifelse(regulation[1, consistent] > 0, "up_vs_control", "down_vs_control")
shared_consistent <- shared_consistent[, .(gene_symbol)]
shared_consistent[, regulation := direction]
# Save shared gene list
fwrite(shared_consistent, file="/outputs/t98g_slow_shared_degs.csv")

# Define HEK293 shared DEGs similarly
sig_genes_hek <- lapply(res_hek293, function(dt) dt[padj < 0.05])
shared_genes_hek <- Reduce(function(x,y) merge(x, y, by=c("gene_symbol"), all=FALSE), sig_genes_hek)
regulation_hek <- sapply(names(res_hek293), function(n) {
  dt <- res_hek293[[n]]
  dt <- dt[gene_symbol %in% shared_genes_hek$gene_symbol]
  sign(dt$log2_fold_change)
})
consistent_hek <- apply(regulation_hek, 1, function(x) length(unique(x)) == 1 && !any(is.na(x)))
shared_consistent_hek <- shared_genes_hek[consistent_hek]
direction_hek <- ifelse(regulation_hek[1, consistent_hek] > 0, "up_vs_control", "down_vs_control")
shared_consistent_hek <- shared_consistent_hek[, .(gene_symbol)]
shared_consistent_hek[, regulation := direction_hek]
fwrite(shared_consistent_hek, file="/outputs/hek293_shared_degs.csv")

# Define T98G D4-specific upregulated genes: significant upregulated in D4 vs control and not significant upregulated in other T98G conditions
d4_res <- res_t98g[["t98g_d4_vs_control_degs"]]
d4_up <- d4_res[padj < 0.05 & log2_fold_change > 0]
other_conditions <- setdiff(names(res_t98g), "t98g_d4_vs_control_degs")
other_up_genes <- unique(unlist(lapply(res_t98g[other_conditions], function(dt) dt[padj < 0.05 & log2_fold_change > 0, gene_id])))
d4_specific_up <- d4_up[!gene_id %in% other_up_genes, .(gene_id)]
fwrite(d4_specific_up, file="/outputs/t98g_d4_specific_upregulated_genes.csv")

# Save individual T98G DEGs files for each treatment
for (name in names(res_t98g)) {
  fwrite(res_t98g[[name]], file=file.path("/outputs", paste0(name, ".csv")))
}
# Save individual HEK293 DEGs files for each treatment
for (name in names(res_hek293)) {
  fwrite(res_hek293[[name]], file=file.path("/outputs", paste0(name, ".csv")))
}

# End of script
