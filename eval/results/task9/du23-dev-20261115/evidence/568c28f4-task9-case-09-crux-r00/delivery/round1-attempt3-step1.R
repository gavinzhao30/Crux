library(DESeq2)
library(SummarizedExperiment)
library(jsonlite)
library(dplyr)
library(readr)

# Load counts and sample metadata
counts <- read_tsv('/data/tagseq_featurecounts_matrix.tsv')
samples <- read_tsv('/data/samples.tsv')

# Prepare count matrix and sample metadata
count_mat <- as.matrix(counts[,-1])
rownames(count_mat) <- counts[[1]]

# Strip version numbers from gene IDs except _PAR_Y
strip_version <- function(ids) {
  sapply(ids, function(id) {
    if(grepl('_PAR_Y$', id)) {
      return(id)
    } else {
      sub('\\..*$', '', id)
    }
  })
}

unversioned_ids <- strip_version(rownames(count_mat))
if(any(duplicated(unversioned_ids))) {
  stop('Gene ID collisions after stripping versions')
}
rownames(count_mat) <- unversioned_ids

# Load gene_symbol_by_gene_id mapping from inputs_metadata.json
metadata <- fromJSON('/data/inputs_metadata.json')
if(!is.null(metadata$gene_symbol_by_gene_id)) {
  gene_map_raw <- metadata$gene_symbol_by_gene_id
  # gene_map_raw is a named list: names are gene_ids (possibly versioned), values are gene_symbols (possibly multiple)
  # Strip version from keys
  keys <- names(gene_map_raw)
  keys_unversioned <- strip_version(keys)
  # Create a data frame to check ambiguous mappings
  gene_map_df <- data.frame(
    gene_id_versioned = keys,
    gene_id = keys_unversioned,
    gene_symbol_raw = unlist(gene_map_raw),
    stringsAsFactors = FALSE
  )
  # Check ambiguous mappings
  ambiguous_ids <- gene_map_df %>% group_by(gene_id) %>% filter(n_distinct(gene_symbol_raw) > 1) %>% pull(gene_id) %>% unique()
  # Create final mapping: only keep unambiguous
  gene_map_final <- gene_map_df %>% 
    filter(!(gene_id %in% ambiguous_ids)) %>% 
    distinct(gene_id, .keep_all=TRUE) %>% 
    select(gene_id, gene_symbol_raw)
  gene_symbol_map <- setNames(gene_map_final$gene_symbol_raw, gene_map_final$gene_id)
} else {
  gene_symbol_map <- NULL
  ambiguous_ids <- character(0)
}

# Prepare output directory
if(!dir.exists('/outputs')) dir.create('/outputs')

# Prepare METHODS.md content
methods_lines <- c()
methods_lines <- c(methods_lines, 'Analysis methods and decisions:')
methods_lines <- c(methods_lines, '- Sample labels and library quality checked: all usable, no swaps or duplicates detected (settled decisions).')
methods_lines <- c(methods_lines, '- Gene IDs stripped of version numbers except _PAR_Y suffix to avoid collisions (settled decision).')
methods_lines <- c(methods_lines, '- Used DESeq2 default settings with design ~ condition per cell line.')
methods_lines <- c(methods_lines, '- Multiple testing correction: DESeq2 default Benjamini-Hochberg FDR (unresolved, kept default).')
methods_lines <- c(methods_lines, '- No pre-filtering of low-count genes or samples (unresolved, kept default).')
methods_lines <- c(methods_lines, '- Gene symbol mapping used only when unambiguous; ambiguous gene IDs left with blank gene_symbol and recorded below.')
if(length(ambiguous_ids) > 0) {
  methods_lines <- c(methods_lines, paste0('- Ambiguous gene symbol mappings for gene IDs: ', paste(ambiguous_ids, collapse=',')))
} else {
  methods_lines <- c(methods_lines, '- No ambiguous gene symbol mappings detected.')
}

# Prepare to store DE results per cell line and condition
cell_lines <- unique(samples$cell_line)

# Function to run DESeq2 and extract results for all conditions vs Ctrl
run_deseq2 <- function(count_mat, sample_df, cell_line) {
  # Subset samples for cell line
  sub_samples <- sample_df %>% filter(cell_line == cell_line)
  sub_counts <- count_mat[, sub_samples$sample_id, drop=FALSE]
  # Prepare colData
  colData <- data.frame(condition = factor(sub_samples$condition))
  rownames(colData) <- sub_samples$sample_id
  # DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(countData = sub_counts, colData = colData, design = ~ condition)
  # Run DESeq
  dds <- DESeq(dds)
  # Extract results for each condition vs Ctrl
  conditions <- levels(colData$condition)
  conditions <- conditions[conditions != 'Ctrl']
  res_list <- list()
  for(cond in conditions) {
    res <- results(dds, contrast = c('condition', cond, 'Ctrl'))
    res_df <- as.data.frame(res)
    res_df$gene_id <- rownames(res_df)
    # Add gene_symbol
    if(!is.null(gene_symbol_map)) {
      res_df$gene_symbol <- gene_symbol_map[res_df$gene_id]
      res_df$gene_symbol[is.na(res_df$gene_symbol)] <- ''
    } else {
      res_df$gene_symbol <- ''
    }
    # Rename columns to match contract
    colnames(res_df)[colnames(res_df) == 'baseMean'] <- 'base_mean'
    colnames(res_df)[colnames(res_df) == 'log2FoldChange'] <- 'log2_fold_change'
    colnames(res_df)[colnames(res_df) == 'lfcSE'] <- 'lfc_se'
    colnames(res_df)[colnames(res_df) == 'stat'] <- 'stat'
    colnames(res_df)[colnames(res_df) == 'pvalue'] <- 'p_value'
    colnames(res_df)[colnames(res_df) == 'padj'] <- 'padj'
    # Select required columns
    res_df <- res_df %>% select(gene_id, gene_symbol, base_mean, log2_fold_change, lfc_se, stat, p_value, padj)
    # Sort by padj
    res_df <- res_df[order(res_df$padj),]
    # Save to list
    res_list[[cond]] <- res_df
  }
  return(res_list)
}

# Run DESeq2 per cell line
all_results <- list()
for(cl in cell_lines) {
  all_results[[cl]] <- run_deseq2(count_mat, samples, cl)
}

# Write DE result files as per contract
# T98G contrasts: C10, F7, G2, D4
for(cond in c('C10','F7','G2','D4')) {
  res_df <- all_results[['T98G']][[cond]]
  # Write DEGs
  fname <- paste0('/outputs/t98g_', tolower(cond), '_vs_control_degs.csv')
  write_csv(res_df, fname)
}
# HEK293 contrasts: D10, E6
for(cond in c('D10','E6')) {
  res_df <- all_results[['HEK293']][[cond]]
  fname <- paste0('/outputs/hek293_', tolower(cond), '_vs_control_degs.csv')
  write_csv(res_df, fname)
}

# Prepare shared gene sets for T98G slow shared DEGs
# Shared DEGs: genes significant (padj<0.05) in all T98G conditions
sig_genes_list <- lapply(all_results[['T98G']], function(df) df$gene_id[df$padj < 0.05])
shared_genes <- Reduce(intersect, sig_genes_list)
# Determine regulation direction: majority vote across conditions
regulation_df <- data.frame(gene_symbol=character(), regulation=character(), stringsAsFactors=FALSE)
for(gene in shared_genes) {
  lfc_vec <- sapply(all_results[['T98G']], function(df) {
    val <- df$log2_fold_change[df$gene_id == gene]
    if(length(val)==0) return(NA) else return(val)
  })
  # Majority direction
  up_count <- sum(lfc_vec > 0, na.rm=TRUE)
  down_count <- sum(lfc_vec < 0, na.rm=TRUE)
  if(up_count > down_count) {
    reg <- 'up_vs_control'
  } else if(down_count > up_count) {
    reg <- 'down_vs_control'
  } else {
    reg <- 'up_vs_control' # default if tie
  }
  # Get gene_symbol from mapping or blank
  gsym <- if(!is.null(gene_symbol_map) && !is.na(gene_symbol_map[gene])) gene_symbol_map[gene] else ''
  regulation_df <- rbind(regulation_df, data.frame(gene_symbol=gsym, regulation=reg, stringsAsFactors=FALSE))
}
write_csv(regulation_df, '/outputs/t98g_slow_shared_degs.csv')

# Prepare shared gene sets for HEK293
sig_genes_list_hek <- lapply(all_results[['HEK293']], function(df) df$gene_id[df$padj < 0.05])
shared_genes_hek <- Reduce(intersect, sig_genes_list_hek)
regulation_df_hek <- data.frame(gene_symbol=character(), regulation=character(), stringsAsFactors=FALSE)
for(gene in shared_genes_hek) {
  lfc_vec <- sapply(all_results[['HEK293']], function(df) {
    val <- df$log2_fold_change[df$gene_id == gene]
    if(length(val)==0) return(NA) else return(val)
  })
  up_count <- sum(lfc_vec > 0, na.rm=TRUE)
  down_count <- sum(lfc_vec < 0, na.rm=TRUE)
  if(up_count > down_count) {
    reg <- 'up_vs_control'
  } else if(down_count > up_count) {
    reg <- 'down_vs_control'
  } else {
    reg <- 'up_vs_control'
  }
  gsym <- if(!is.null(gene_symbol_map) && !is.na(gene_symbol_map[gene])) gene_symbol_map[gene] else ''
  regulation_df_hek <- rbind(regulation_df_hek, data.frame(gene_symbol=gsym, regulation=reg, stringsAsFactors=FALSE))
}
write_csv(regulation_df_hek, '/outputs/hek293_shared_degs.csv')

# Prepare D4-specific upregulated genes for T98G
# Genes significant (padj<0.05) and upregulated (log2FC>0) in D4 but not significant in other T98G conditions
d4_res <- all_results[['T98G']][['D4']]
d4_sig_up <- d4_res %>% filter(padj < 0.05 & log2_fold_change > 0)
d4_other_conditions <- setdiff(names(all_results[['T98G']]), 'D4')
not_sig_in_others <- sapply(d4_sig_up$gene_id, function(g) {
  any(sapply(d4_other_conditions, function(cond) {
    df <- all_results[['T98G']][[cond]]
    padj_val <- df$padj[df$gene_id == g]
    if(length(padj_val)==0) return(FALSE) else return(padj_val < 0.05)
  })) == FALSE
})
d4_specific_up <- d4_sig_up[not_sig_in_others, ]
# Write only gene_id column
write_csv(d4_specific_up %>% select(gene_id), '/outputs/t98g_d4_specific_upregulated_genes.csv')

# Write METHODS.md
writeLines(methods_lines, '/outputs/METHODS.md')
