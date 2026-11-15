library(DESeq2)
library(data.table)
library(dplyr)

# Load stripped counts and samples
counts <- fread('/work/stripped_counts.tsv', data.table=FALSE)
rownames(counts) <- counts[[1]]
counts[[1]] <- NULL

samples <- read.csv('/work/filtered_samples.tsv', stringsAsFactors=FALSE)

# Function to run DESeq2 for one cell line
run_deseq2 <- function(cell_line, counts, samples) {
  # Subset samples
  sub_samples <- samples[samples$cell_line == cell_line, ]
  sub_counts <- counts[, sub_samples$sample_id, drop=FALSE]

  # Create DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(countData = sub_counts,
                                colData = sub_samples,
                                design = ~ condition)

  # Run DESeq2
  dds <- DESeq(dds)

  # Get all conditions except control
  conditions <- setdiff(unique(sub_samples$condition), 'Ctrl')

  results_list <- list()

  for (cond in conditions) {
    contrast <- c('condition', cond, 'Ctrl')
    res <- results(dds, contrast=contrast)
    res_df <- as.data.frame(res)
    res_df$gene_id <- rownames(res_df)

    # Add gene_symbol column placeholder
    res_df$gene_symbol <- NA_character_

    # Rename columns to match contract
    colnames(res_df)[colnames(res_df) == 'baseMean'] <- 'base_mean'
    colnames(res_df)[colnames(res_df) == 'log2FoldChange'] <- 'log2_fold_change'
    colnames(res_df)[colnames(res_df) == 'lfcSE'] <- 'lfc_se'
    colnames(res_df)[colnames(res_df) == 'stat'] <- 'stat'
    colnames(res_df)[colnames(res_df) == 'pvalue'] <- 'p_value'
    colnames(res_df)[colnames(res_df) == 'padj'] <- 'padj'

    # Select required columns
    res_df <- res_df[, c('gene_id', 'gene_symbol', 'base_mean', 'log2_fold_change', 'lfc_se', 'stat', 'p_value', 'padj')]

    # Save to file
    filename <- paste0(tolower(cell_line), '_', tolower(cond), '_vs_control_degs.csv')
    write.csv(res_df, file=file.path('/outputs', filename), row.names=FALSE, quote=FALSE)

    results_list[[cond]] <- res_df
  }
  return(results_list)
}

# Run for T98G and HEK293
results_t98g <- run_deseq2('T98G', counts, samples)
results_hek293 <- run_deseq2('HEK293', counts, samples)

# Save shared and D4-specific gene sets
# Shared DEGs: genes significant (padj<0.05) in all treatments per cell line

# Helper to get significant genes per condition
get_sig_genes <- function(res_list) {
  sig_genes <- lapply(res_list, function(df) {
    df %>% filter(!is.na(padj) & padj < 0.05) %>% pull(gene_id)
  })
  Reduce(intersect, sig_genes)
}

# T98G shared DEGs
t98g_shared_genes <- get_sig_genes(results_t98g)

# For shared DEGs, determine regulation direction (up/down) per gene
# Use mean log2 fold change across conditions
shared_df <- data.frame(gene_id = t98g_shared_genes, stringsAsFactors=FALSE)
shared_df$mean_log2fc <- sapply(shared_df$gene_id, function(gid) {
  mean(sapply(results_t98g, function(df) {
    val <- df$log2_fold_change[df$gene_id == gid]
    if(length(val)==0) NA else val
  }), na.rm=TRUE)
})
shared_df$regulation <- ifelse(shared_df$mean_log2fc > 0, 'up_vs_control', 'down_vs_control')

# Map gene_id to gene_symbol if possible (no mapping file available, leave blank)
shared_df$gene_symbol <- NA_character_

# Save shared DEGs by gene_symbol as required (contract expects gene_symbol as row key)
# But gene_symbol missing, so use gene_id as gene_symbol for output
shared_df$gene_symbol <- shared_df$gene_id
shared_out <- shared_df[, c('gene_symbol', 'regulation')]
write.csv(shared_out, file='/outputs/t98g_slow_shared_degs.csv', row.names=FALSE, quote=FALSE)

# D4-specific upregulated genes: significant in D4 vs Ctrl but not in other T98G conditions

d4_res <- results_t98g[['D4']]
d4_sig <- d4_res %>% filter(!is.na(padj) & padj < 0.05 & log2_fold_change > 0)

# Get genes significant in other T98G conditions
other_conditions <- setdiff(names(results_t98g), 'D4')
other_sig_genes <- unique(unlist(lapply(results_t98g[other_conditions], function(df) {
  df %>% filter(!is.na(padj) & padj < 0.05) %>% pull(gene_id)
})))

# D4-specific genes are those in d4_sig not in other_sig
d4_specific_genes <- d4_sig$gene_id[!d4_sig$gene_id %in% other_sig_genes]

# Save D4-specific upregulated genes
write.csv(data.frame(gene_id=d4_specific_genes), file='/outputs/t98g_d4_specific_upregulated_genes.csv', row.names=FALSE, quote=FALSE)

# HEK293 shared DEGs
hek293_shared_genes <- get_sig_genes(results_hek293)
hek_shared_df <- data.frame(gene_id = hek293_shared_genes, stringsAsFactors=FALSE)
hek_shared_df$mean_log2fc <- sapply(hek_shared_df$gene_id, function(gid) {
  mean(sapply(results_hek293, function(df) {
    val <- df$log2_fold_change[df$gene_id == gid]
    if(length(val)==0) NA else val
  }), na.rm=TRUE)
})
hek_shared_df$regulation <- ifelse(hek_shared_df$mean_log2fc > 0, 'up_vs_control', 'down_vs_control')
hek_shared_df$gene_symbol <- hek_shared_df$gene_id
hek_shared_out <- hek_shared_df[, c('gene_symbol', 'regulation')]
write.csv(hek_shared_out, file='/outputs/hek293_shared_degs.csv', row.names=FALSE, quote=FALSE)

# Save individual HEK293 DE results files
# Already saved in run_deseq2

# Save individual T98G DE results files
# Already saved in run_deseq2

# Append to METHODS.md
methods_file <- '/outputs/METHODS.md'
methods_append <- c(
  '\n## Differential Expression Analysis',
  'DESeq2 was run separately per cell line using design formula ~ condition including all retained conditions and biological replicates.',
  'The default DESeq2 Benjamini-Hochberg FDR correction was used with significance threshold padj < 0.05.',
  'Gene IDs were stripped of Ensembl version numbers while preserving _PAR_Y suffixes.',
  'Gene symbols were not assigned due to lack of unambiguous mapping data; gene_id was used as gene_symbol in output gene lists.',
  'Shared DE gene sets were defined as genes significant in all treatments versus control per cell line.',
  'D4-specific genes were defined as genes significantly upregulated in T98G D4 versus control but not significant in other T98G conditions.'
)

write(paste(methods_append, collapse='\n'), file=methods_file, append=TRUE)
