library(DESeq2)
library(data.table)
library(stringr)
library(dplyr)

# Load sample metadata
samples <- fread("/data/samples.tsv")

# Apply settled decision: swap condition labels for samples S09 and S13, and S10 and S15
swap_map <- list("S09" = "S13", "S13" = "S09", "S10" = "S15", "S15" = "S10")
samples[, condition := ifelse(sample_id %in% names(swap_map), 
                              samples[match(swap_map[sample_id], sample_id), condition], 
                              condition)]

# Load counts matrix
counts <- fread("/data/tagseq_featurecounts_matrix.tsv")
# First column is gene_id
gene_ids <- counts[[1]]
count_mat <- as.matrix(counts[ , -1, with=FALSE])
colnames(count_mat) <- colnames(counts)[-1]
rownames(count_mat) <- gene_ids

# Confirm gene IDs are versioned and uniform (settled decision)
# Strip version numbers for output only
strip_version <- function(gene_id) {
  # Preserve _PAR_Y suffix
  if (grepl("_PAR_Y$", gene_id)) {
    base <- sub("_PAR_Y$", "", gene_id)
    base <- sub("\\.\\d+$", "", base)
    return(paste0(base, "_PAR_Y"))
  } else {
    return(sub("\\.\\d+$", "", gene_id))
  }
}

# Load gene_symbol_by_gene_id mapping from inputs_metadata.json if available
library(jsonlite)
meta <- fromJSON("/data/inputs_metadata.json")
gene_symbol_map <- NULL
if (!is.null(meta$gene_symbol_by_gene_id)) {
  gene_symbol_map <- meta$gene_symbol_by_gene_id
  # Strip versions from keys
  names(gene_symbol_map) <- sapply(names(gene_symbol_map), strip_version)
}

# Prepare output directory
dir.create("/outputs", showWarnings=FALSE)

# Function to map gene symbols unambiguously
map_gene_symbols <- function(gene_ids, gene_symbol_map) {
  stripped_ids <- sapply(gene_ids, strip_version)
  symbols <- rep(NA_character_, length(gene_ids))
  if (!is.null(gene_symbol_map)) {
    # Find unique mappings only
    unique_ids <- unique(stripped_ids)
    # Count how many gene_ids map to each symbol
    symbol_counts <- table(unlist(gene_symbol_map))
    for (i in seq_along(stripped_ids)) {
      gid <- stripped_ids[i]
      if (gid %in% names(gene_symbol_map)) {
        sym <- gene_symbol_map[[gid]]
        # Check if mapping is unambiguous
        if (sum(gene_symbol_map == sym) == 1) {
          symbols[i] <- sym
        }
      }
    }
  }
  return(symbols)
}

# Check sample labels and cell line assignments (settled: consistent after swap)
# No duplicates or degraded libraries (settled)

# Split samples by cell line
cell_lines <- unique(samples$cell_line)

# Prepare a list to store DESeq2 results per cell line and condition
results_list <- list()

for (cl in cell_lines) {
  # Subset samples for this cell line
  smp_cl <- samples[cell_line == cl]
  # Subset count matrix columns
  count_cl <- count_mat[, smp_cl$sample_id, drop=FALSE]
  
  # Create colData
  colData <- data.frame(row.names = smp_cl$sample_id, condition = factor(smp_cl$condition))
  
  # Create DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(countData = count_cl, colData = colData, design = ~ condition)
  
  # No prefiltering (untested decision, keep DESeq2 default)
  
  # Run DESeq
  dds <- DESeq(dds)
  
  # Extract all treatment vs Ctrl contrasts
  cond_levels <- levels(colData$condition)
  ctrl_level <- "Ctrl"
  if (!(ctrl_level %in% cond_levels)) {
    stop(paste0("Ctrl condition not found in cell line ", cl))
  }
  treatments <- setdiff(cond_levels, ctrl_level)
  
  for (trt in treatments) {
    res <- results(dds, contrast = c("condition", trt, ctrl_level))
    res_df <- as.data.frame(res)
    res_df$gene_id <- rownames(res_df)
    # Strip version for output
    res_df$gene_id <- sapply(res_df$gene_id, strip_version)
    
    # Map gene symbols
    res_df$gene_symbol <- map_gene_symbols(res_df$gene_id, gene_symbol_map)
    
    # Rename columns to match contract
    # baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
    # Contract requires: gene_id, gene_symbol, base_mean, log2_fold_change, lfc_se, stat, p_value, padj
    out_df <- data.frame(
      gene_id = res_df$gene_id,
      gene_symbol = res_df$gene_symbol,
      base_mean = res_df$baseMean,
      log2_fold_change = res_df$log2FoldChange,
      lfc_se = res_df$lfcSE,
      stat = res_df$stat,
      p_value = res_df$pvalue,
      padj = res_df$padj
    )
    
    # Check uniqueness of gene_id
    if (any(duplicated(out_df$gene_id))) {
      stop(paste0("Duplicated gene_id after version stripping in cell line ", cl, " contrast ", trt))
    }
    
    # Write to file
    # File names from contract
    fname <- NULL
    if (cl == "T98G") {
      # Map treatment to file name suffix
      if (trt == "C10") fname <- "/outputs/t98g_c10_vs_control_degs.csv"
      else if (trt == "F7") fname <- "/outputs/t98g_f7_vs_control_degs.csv"
      else if (trt == "G2") fname <- "/outputs/t98g_g2_vs_control_degs.csv"
      else if (trt == "D4") fname <- "/outputs/t98g_d4_vs_control_degs.csv"
    } else if (cl == "HEK293") {
      if (trt == "D10") fname <- "/outputs/hek293_d10_vs_control_degs.csv"
      else if (trt == "E6") fname <- "/outputs/hek293_e6_vs_control_degs.csv"
    }
    if (!is.null(fname)) {
      fwrite(out_df, file = fname)
    }
    
    # Store results for downstream gene set extraction
    results_list[[paste0(cl, "_", trt)]] <- out_df
  }
}

# Extract shared and D4-specific gene sets as requested
# For T98G slow shared DEGs: genes significant (padj<0.05) and regulated in same direction in all slow treatments (C10, F7, G2)
slow_treatments <- c("C10", "F7", "G2")
slow_dfs <- lapply(slow_treatments, function(trt) {
  df <- results_list[[paste0("T98G_", trt)]]
  df <- df[!is.na(df$padj) & df$padj < 0.05, ]
  df$regulation <- ifelse(df$log2_fold_change > 0, "up_vs_control", "down_vs_control")
  df[, c("gene_id", "gene_symbol", "regulation")]
})

# Find genes common to all three with same regulation
common_genes <- Reduce(function(x, y) merge(x, y, by = c("gene_symbol", "regulation")), slow_dfs)
# Keep gene_symbol and regulation columns
t98g_slow_shared <- common_genes[, c("gene_symbol", "regulation")]

# Write T98G slow shared DEGs
fwrite(t98g_slow_shared, file = "/outputs/t98g_slow_shared_degs.csv")

# For HEK293 shared DEGs: genes significant and regulated in same direction in both D10 and E6
hek_treatments <- c("D10", "E6")
hek_dfs <- lapply(hek_treatments, function(trt) {
  df <- results_list[[paste0("HEK293_", trt)]]
  df <- df[!is.na(df$padj) & df$padj < 0.05, ]
  df$regulation <- ifelse(df$log2_fold_change > 0, "up_vs_control", "down_vs_control")
  df[, c("gene_id", "gene_symbol", "regulation")]
})

common_hek <- Reduce(function(x, y) merge(x, y, by = c("gene_symbol", "regulation")), hek_dfs)
hek293_shared <- common_hek[, c("gene_symbol", "regulation")]

# Write HEK293 shared DEGs
fwrite(hek293_shared, file = "/outputs/hek293_shared_degs.csv")

# For T98G D4-specific upregulated genes: genes significantly upregulated (padj<0.05, log2FC>0) in D4 but not significant in other treatments
d4_df <- results_list[["T98G_D4"]]
d4_up <- d4_df[!is.na(d4_df$padj) & d4_df$padj < 0.05 & d4_df$log2_fold_change > 0, ]

# Get genes significant in other treatments
other_treatments <- setdiff(slow_treatments, "D4")
other_sig_genes <- unique(unlist(lapply(other_treatments, function(trt) {
  df <- results_list[[paste0("T98G_", trt)]]
  df <- df[!is.na(df$padj) & df$padj < 0.05, ]
  df$gene_id
})))

# Filter D4 upregulated genes to those not significant in other treatments
d4_specific_up <- d4_up[!(d4_up$gene_id %in% other_sig_genes), ]

# Write T98G D4-specific upregulated genes
fwrite(d4_specific_up[, "gene_id", drop=FALSE], file = "/outputs/t98g_d4_specific_upregulated_genes.csv")

# Write METHODS.md recording decisions and diagnostics
methods_text <- c(
  "# METHODS.md",
  "",
  "## Sample label correction",
  "Swapped condition labels for samples S09 and S13, and S10 and S15 in T98G cell line as per settled decision based on nearest neighbour and replicate correlation diagnostics.",
  "",
  "## Library quality",
  "All libraries passed quality checks: no duplicates or degraded libraries detected.",
  "",
  "## Cell line assignments",
  "Cell line assignments were consistent with expression data.",
  "",
  "## Gene ID processing",
  "Gene IDs were versioned uniformly; stripped version numbers for output while preserving _PAR_Y suffixes, ensuring unique row keys.",
  "",
  "## Gene symbol mapping",
  "Mapped gene symbols using input metadata by stable gene ID ignoring version only when mapping was unambiguous; unresolved mappings left blank.",
  "",
  "## DESeq2 analysis",
  "Fitted DESeq2 models per cell line with design formula ~ condition including all conditions and biological replicates.",
  "No manual low-count prefiltering applied; relied on DESeq2's internal independent filtering.",
  "Extracted every treatment-versus-Ctrl contrast from each fitted model.",
  "Multiple testing correction applied per contrast using DESeq2's default Benjamini-Hochberg procedure.",
  "",
  "## Gene set extraction",
  "Identified shared DEGs for T98G slow treatments (C10, F7, G2) and HEK293 treatments (D10, E6) as genes significant and regulated in the same direction across all treatments.",
  "Identified T98G D4-specific upregulated genes as those significantly upregulated in D4 but not significant in other treatments.",
  "",
  "## Unresolved questions",
  "Several analysis design questions remain untested; default DESeq2 settings and simple design formula ~ condition per cell line were used.",
  ""
)
writeLines(methods_text, "/outputs/METHODS.md")
