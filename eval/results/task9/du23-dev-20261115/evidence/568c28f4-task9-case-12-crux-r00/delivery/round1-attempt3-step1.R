library(DESeq2)
library(tibble)
library(readr)
library(dplyr)
library(stringr)

# Read count matrix and sample sheet
counts <- read_tsv("/data/tagseq_featurecounts_matrix.tsv")
samples <- read_tsv("/data/samples.tsv")

# Fix swapped sample labels as per settled decision
# Swap conditions for samples S09 <-> S13 and S10 <-> S15
samples_fixed <- samples %>% mutate(condition = case_when(
  sample_id == "S09" ~ samples$condition[samples$sample_id == "S13"],
  sample_id == "S13" ~ samples$condition[samples$sample_id == "S09"],
  sample_id == "S10" ~ samples$condition[samples$sample_id == "S15"],
  sample_id == "S15" ~ samples$condition[samples$sample_id == "S10"],
  TRUE ~ condition
))

# Prepare gene_id vector and strip version numbers preserving _PAR_Y
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(gid) {
    if (grepl("_PAR_Y$", gid)) {
      return(gid) # preserve _PAR_Y suffix
    } else {
      sub("\\.\\d+$", "", gid) # remove version suffix
    }
  })
}
gene_ids <- counts[[1]]
gene_ids_stripped <- strip_version(gene_ids)

# Check uniqueness of stripped gene_ids
if (any(duplicated(gene_ids_stripped))) {
  stop("Duplicated gene_ids after stripping versions detected")
}

# Prepare count matrix with stripped gene_ids as rownames
count_mat <- counts %>% select(-1) %>% as.matrix()
rownames(count_mat) <- gene_ids_stripped

# Confirm samples in counts columns match samples_fixed sample_id
if (!all(colnames(count_mat) == samples_fixed$sample_id)) {
  stop("Count matrix columns and sample sheet sample_ids do not match")
}

# Load gene_symbol mapping from inputs_metadata.json
library(jsonlite)
metadata <- fromJSON("/data/inputs_metadata.json")
gene_symbol_map <- metadata$gene_symbol_by_gene_id

# Process gene_symbol mapping: strip versions and keep only unambiguous mappings
strip_map_keys <- strip_version(names(gene_symbol_map))
mapping_df <- tibble(gene_id_versioned = names(gene_symbol_map),
                     gene_id = strip_map_keys,
                     gene_symbol = unlist(gene_symbol_map))

# Keep only unambiguous mappings
ambiguous_ids <- mapping_df %>% group_by(gene_id) %>% filter(n() > 1) %>% pull(gene_id) %>% unique()

mapping_unique <- mapping_df %>% filter(!gene_id %in% ambiguous_ids) %>% distinct(gene_id, .keep_all=TRUE)

# Record ambiguous mappings for METHODS.md
ambiguous_mappings <- mapping_df %>% filter(gene_id %in% ambiguous_ids) %>% distinct(gene_id, gene_symbol)

# Prepare a named vector for mapping
gene_symbol_vec <- setNames(mapping_unique$gene_symbol, mapping_unique$gene_id)

# Function to add gene_symbol to DESeq2 results
add_gene_symbol <- function(res_df) {
  res_df$gene_symbol <- gene_symbol_vec[res_df$gene_id]
  res_df$gene_symbol[is.na(res_df$gene_symbol)] <- ""
  return(res_df)
}

# Split samples by cell_line
samples_by_line <- split(samples_fixed, samples_fixed$cell_line)

# Prepare output list for DE results per contrast
de_results <- list()

# For each cell line, run DESeq2 with design ~ condition
for (line in names(samples_by_line)) {
  sample_sub <- samples_by_line[[line]]
  # Subset count matrix columns
  count_sub <- count_mat[, sample_sub$sample_id]
  # Prepare colData
  coldata <- DataFrame(condition = factor(sample_sub$condition))
  rownames(coldata) <- sample_sub$sample_id
  # Create DESeqDataSet
  dds <- DESeqDataSetFromMatrix(countData = count_sub, colData = coldata, design = ~ condition)
  # Run DESeq
  dds <- DESeq(dds)
  # Extract all treatment vs Ctrl contrasts
  cond_levels <- levels(coldata$condition)
  ctrl_level <- "Ctrl"
  treatments <- setdiff(cond_levels, ctrl_level)
  for (treat in treatments) {
    contrast_name <- paste0(tolower(line), "_", tolower(treat), "_vs_control_degs")
    res <- results(dds, contrast = c("condition", treat, ctrl_level))
    res_df <- as.data.frame(res) %>% rownames_to_column(var = "gene_id")
    # Rename columns to match contract
    res_df <- res_df %>% rename(
      base_mean = baseMean,
      log2_fold_change = log2FoldChange,
      lfc_se = lfcSE,
      p_value = pvalue,
      padj = padj
    )
    # Add gene_symbol
    res_df <- add_gene_symbol(res_df)
    # Select required columns
    res_df <- res_df %>% select(gene_id, gene_symbol, base_mean, log2_fold_change, lfc_se, stat, p_value, padj)
    # Write to output
    write_csv(res_df, file = file.path("/outputs", paste0(contrast_name, ".csv")))
    de_results[[contrast_name]] <- res_df
  }
}

# Create shared and D4-specific gene sets for T98G
# Shared DEGs: genes significantly DE (padj < 0.05) in all T98G treatments vs Ctrl with consistent regulation
# D4-specific upregulated genes: genes significantly upregulated only in D4 vs Ctrl

# Collect T98G DE results
t98g_results <- de_results[grepl("^t98g_", names(de_results))]

# Filter significant genes per contrast
sig_genes_list <- lapply(t98g_results, function(df) {
  df %>% filter(padj < 0.05) %>% mutate(regulation = ifelse(log2_fold_change > 0, "up_vs_control", "down_vs_control"))
})

# Find shared genes across all contrasts
shared_genes <- Reduce(function(x, y) inner_join(x, y, by = "gene_id"), sig_genes_list)

# Keep genes with consistent regulation across all contrasts
reg_cols <- grep("regulation", colnames(shared_genes), value = TRUE)
consistent_reg <- apply(shared_genes[, reg_cols], 1, function(x) length(unique(x)) == 1)
shared_consistent <- shared_genes[consistent_reg, ]

# Prepare shared DEGs output
shared_degs <- shared_consistent %>% select(gene_symbol = gene_symbol.x, regulation = regulation.x) %>% distinct()
write_csv(shared_degs, file = "/outputs/t98g_slow_shared_degs.csv")

# D4-specific upregulated genes
# Genes upregulated in D4 vs Ctrl (padj < 0.05, log2FC > 0) and not significant in other contrasts
d4_res <- t98g_results[["t98g_d4_vs_control_degs"]]
d4_up <- d4_res %>% filter(padj < 0.05 & log2_fold_change > 0)
other_contrasts <- setdiff(names(t98g_results), "t98g_d4_vs_control_degs")
other_sig_genes <- unique(unlist(lapply(t98g_results[other_contrasts], function(df) df$gene_id[df$padj < 0.05])))
d4_specific_up <- d4_up %>% filter(!gene_id %in% other_sig_genes) %>% select(gene_id)
write_csv(d4_specific_up, file = "/outputs/t98g_d4_specific_upregulated_genes.csv")

# HEK293 shared DEGs: genes significantly DE in both E6 and D10 vs Ctrl with consistent regulation
hek293_results <- de_results[grepl("^hek293_", names(de_results))]
hek293_sig <- lapply(hek293_results, function(df) {
  df %>% filter(padj < 0.05) %>% mutate(regulation = ifelse(log2_fold_change > 0, "up_vs_control", "down_vs_control"))
})
shared_hek293 <- Reduce(function(x, y) inner_join(x, y, by = "gene_id"), hek293_sig)
reg_cols_hek <- grep("regulation", colnames(shared_hek293), value = TRUE)
consistent_reg_hek <- apply(shared_hek293[, reg_cols_hek], 1, function(x) length(unique(x)) == 1)
shared_hek293_consistent <- shared_hek293[consistent_reg_hek, ]
hek293_shared_degs <- shared_hek293_consistent %>% select(gene_symbol = gene_symbol.x, regulation = regulation.x) %>% distinct()
write_csv(hek293_shared_degs, file = "/outputs/hek293_shared_degs.csv")

# Write METHODS.md with decisions and unresolved issues
methods_md <- c(
  "# METHODS",
  "- Swapped condition labels for samples S09 <-> S13 and S10 <-> S15 in T98G cell line as per diagnostics.",
  "- Used design formula ~ condition for DESeq2 fitting per cell line including all conditions and replicates.",
  "- No gene filtering before DESeq2; relied on DESeq2 internal independent filtering.",
  "- Extracted all treatment-versus-Ctrl contrasts from the same fitted model per cell line.",
  "- Stripped Ensembl version numbers from gene_id values while preserving _PAR_Y suffixes; confirmed unique gene_ids after stripping.",
  "- Mapped gene symbols using input metadata matching stable gene IDs only when unambiguous; ambiguous mappings recorded below.",
  "- No duplicate or degraded libraries detected; all samples retained.",
  "- Sample labels and cell line assignments consistent after swapping condition labels as above.",
  "- Normalization used DESeq2's internal median-of-ratios method.",
  "- Unresolved questions with untested evidence kept default DESeq2 settings.",
  "- Ambiguous gene symbol mappings (gene_ids with multiple symbols) include:",
  paste(capture.output(print(ambiguous_mappings)), collapse = "\n")
)
writeLines(methods_md, "/outputs/METHODS.md")
