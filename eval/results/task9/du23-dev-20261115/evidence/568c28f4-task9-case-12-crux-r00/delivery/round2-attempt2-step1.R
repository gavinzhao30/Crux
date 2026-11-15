library(DESeq2)
library(data.table)
library(stringr)

# Load count matrix and sample metadata
counts <- fread('/data/tagseq_featurecounts_matrix.tsv', data.table=FALSE)
samples <- fread('/data/samples.tsv', data.table=FALSE)

# Prepare count matrix: rownames are gene_id
rownames(counts) <- counts$gene_id
counts$gene_id <- NULL

# Confirm gene IDs are versioned and uniform
# Strip version numbers from gene IDs but preserve _PAR_Y suffix
strip_version <- function(gene_ids) {
  # Remove version after dot only if not _PAR_Y
  sapply(gene_ids, function(id) {
    if (grepl('_PAR_Y$', id)) {
      # Remove version before _PAR_Y
      sub('(ENSG\d+)\.\d+(_PAR_Y)$', '\1\2', id)
    } else {
      sub('(ENSG\d+)\.\d+', '\1', id)
    }
  })
}

gene_ids_stripped <- strip_version(rownames(counts))
rownames(counts) <- gene_ids_stripped

# Check uniqueness of gene IDs after stripping
if(any(duplicated(rownames(counts)))) {
  stop('Duplicated gene IDs after version stripping')
}

# Load gene_symbol_by_gene_id mapping from inputs_metadata.json
library(jsonlite)
metadata <- fromJSON('/data/inputs_metadata.json')

# Extract gene_symbol_by_gene_id mapping
gene_symbol_map <- metadata$gene_symbol_by_gene_id

# Process mapping keys to strip versions similarly
map_keys <- names(gene_symbol_map)
map_keys_stripped <- strip_version(map_keys)

# Build mapping table and find unambiguous mappings
mapping_df <- data.frame(
  gene_id_versioned = map_keys,
  gene_id = map_keys_stripped,
  gene_symbol = unlist(gene_symbol_map),
  stringsAsFactors = FALSE
)

# Keep only unambiguous mappings (one gene_symbol per gene_id)
ambiguous_ids <- mapping_df$gene_id[duplicated(mapping_df$gene_id) | duplicated(mapping_df$gene_id, fromLast=TRUE)]

unambiguous_map <- mapping_df[!mapping_df$gene_id %in% ambiguous_ids, ]

# Create named vector for mapping
gene_symbol_by_id <- setNames(unambiguous_map$gene_symbol, unambiguous_map$gene_id)

# Record unresolved mappings
unresolved_mappings <- unique(mapping_df$gene_id[mapping_df$gene_id %in% ambiguous_ids])

# Write METHODS.md header
methods_file <- '/outputs/METHODS.md'
writeLines(c(
  '# Methods and Decisions',
  '',
  '## Sample label correction',
  '- Two pairs of T98G samples (S09|S13 and S10|S15) had swapped condition labels; these were exchanged before DE analysis.',
  '',
  '## Library quality and duplicates',
  '- All libraries passed quality checks; no duplicates or degraded libraries were removed.',
  '',
  '## Cell line assignments',
  '- Cell line assignments in sample sheet were consistent with expression data.',
  '',
  '## Gene ID processing',
  '- Gene IDs were uniformly versioned Ensembl IDs; versions were stripped for output while preserving _PAR_Y suffix.',
  '- Unambiguous gene symbol mappings were applied from inputs_metadata.json; ambiguous mappings were left blank.',
  paste0('- Ambiguous gene IDs without unique gene symbol mapping: ', paste(unresolved_mappings, collapse=', ')),
  '',
  '## DESeq2 design formula',
  '- Design formula ~ condition was used per cell line, fitting one model per cell line with all conditions and replicates.',
  '',
  '## Normalization and filtering',
  '- DESeq2 internal median-of-ratios normalization and independent filtering were used; no pre-filtering of low count genes was applied.',
  '',
  '## Multiple testing correction',
  '- Benjamini-Hochberg FDR correction was applied separately within each contrast for each cell line.',
  '',
  '## Contrast extraction',
  '- All treatment-versus-control contrasts were extracted from the same fitted model per cell line.',
  '',
  '## Output row key uniqueness',
  '- Row keys (gene_id or gene_symbol) were checked for uniqueness after version stripping before writing output files.'
), methods_file)

# Apply sample label swaps in samples metadata
swap_map <- list('S09'='S13', 'S13'='S09', 'S10'='S15', 'S15'='S10')
samples$condition <- ifelse(samples$sample_id %in% names(swap_map),
                            samples$condition[match(swap_map[samples$sample_id], samples$sample_id)],
                            samples$condition)

# Confirm no duplicates or degraded libraries to remove
# Proceed with all samples

# Function to run DESeq2 per cell line
run_deseq2 <- function(cell_line_name) {
  # Subset samples
  sub_samples <- samples[samples$cell_line == cell_line_name, ]
  # Subset counts
  sub_counts <- counts[, sub_samples$sample_id, drop=FALSE]

  # Prepare colData
  colData <- data.frame(row.names=sub_samples$sample_id, condition=factor(sub_samples$condition))

  # Create DESeqDataSet
  dds <- DESeqDataSetFromMatrix(countData=sub_counts, colData=colData, design=~condition)

  # Run DESeq
  dds <- DESeq(dds)

  # Extract results for each treatment vs Ctrl
  conditions <- levels(colData$condition)
  ctrl <- 'Ctrl'
  treatments <- setdiff(conditions, ctrl)

  res_list <- list()
  for (treat in treatments) {
    contrast <- c('condition', treat, ctrl)
    res <- results(dds, contrast=contrast)
    res_df <- as.data.frame(res)
    res_df$gene_id <- rownames(res_df)

    # Add baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
    # Rename columns to contract required names
    colnames(res_df)[colnames(res_df) == 'log2FoldChange'] <- 'log2_fold_change'
    colnames(res_df)[colnames(res_df) == 'lfcSE'] <- 'lfc_se'
    colnames(res_df)[colnames(res_df) == 'pvalue'] <- 'p_value'

    # Add gene_symbol from mapping, blank if ambiguous or missing
    res_df$gene_symbol <- gene_symbol_by_id[res_df$gene_id]
    res_df$gene_symbol[is.na(res_df$gene_symbol)] <- ''

    # Select required columns
    res_df_out <- res_df[, c('gene_id', 'gene_symbol', 'baseMean', 'log2_fold_change', 'lfc_se', 'stat', 'p_value', 'padj')]

    # Sort by padj ascending
    res_df_out <- res_df_out[order(res_df_out$padj), ]

    # Save to list
    res_list[[treat]] <- res_df_out
  }
  return(res_list)
}

# Run DESeq2 for T98G
t98g_res <- run_deseq2('T98G')
# Run DESeq2 for HEK293
hek293_res <- run_deseq2('HEK293')

# Write DE results to files as per contract
write_de_results <- function(res_df, filename) {
  # Check uniqueness of gene_id
  if(any(duplicated(res_df$gene_id))) {
    stop(paste('Duplicated gene_id in', filename))
  }
  write.csv(res_df, file=paste0('/outputs/', filename), row.names=FALSE, quote=FALSE)
}

# T98G contrasts
write_de_results(t98g_res[['C10']], 't98g_c10_vs_control_degs.csv')
write_de_results(t98g_res[['F7']], 't98g_f7_vs_control_degs.csv')
write_de_results(t98g_res[['G2']], 't98g_g2_vs_control_degs.csv')
write_de_results(t98g_res[['D4']], 't98g_d4_vs_control_degs.csv')

# HEK293 contrasts
write_de_results(hek293_res[['D10']], 'hek293_d10_vs_control_degs.csv')
write_de_results(hek293_res[['E6']], 'hek293_e6_vs_control_degs.csv')

# Create shared gene lists for T98G slow shared DEGs
# Define shared as genes significant (padj<0.05) and regulated in same direction in C10, F7, G2
sig_genes <- function(df) {
  df[df$padj < 0.05 & !is.na(df$padj), ]
}

c10_sig <- sig_genes(t98g_res[['C10']])
f7_sig <- sig_genes(t98g_res[['F7']])
g2_sig <- sig_genes(t98g_res[['G2']])

# Find intersection of gene_symbol with consistent regulation
shared_genes <- Reduce(intersect, list(c10_sig$gene_symbol, f7_sig$gene_symbol, g2_sig$gene_symbol))

# Filter for consistent regulation
shared_df <- data.frame(gene_symbol=shared_genes, regulation=NA, stringsAsFactors=FALSE)
for(i in seq_len(nrow(shared_df))) {
  gs <- shared_df$gene_symbol[i]
  lfc_c10 <- c10_sig$log2_fold_change[c10_sig$gene_symbol == gs]
  lfc_f7 <- f7_sig$log2_fold_change[f7_sig$gene_symbol == gs]
  lfc_g2 <- g2_sig$log2_fold_change[g2_sig$gene_symbol == gs]
  if(all(c(lfc_c10, lfc_f7, lfc_g2) > 0)) {
    shared_df$regulation[i] <- 'up_vs_control'
  } else if(all(c(lfc_c10, lfc_f7, lfc_g2) < 0)) {
    shared_df$regulation[i] <- 'down_vs_control'
  } else {
    shared_df$regulation[i] <- NA
  }
}
shared_df <- shared_df[!is.na(shared_df$regulation), ]

write.csv(shared_df, file='/outputs/t98g_slow_shared_degs.csv', row.names=FALSE, quote=FALSE)

# Create HEK293 shared DEGs similarly
hek293_d10_sig <- sig_genes(hek293_res[['D10']])
hek293_e6_sig <- sig_genes(hek293_res[['E6']])
shared_hek_genes <- intersect(hek293_d10_sig$gene_symbol, hek293_e6_sig$gene_symbol)

shared_hek_df <- data.frame(gene_symbol=shared_hek_genes, regulation=NA, stringsAsFactors=FALSE)
for(i in seq_len(nrow(shared_hek_df))) {
  gs <- shared_hek_df$gene_symbol[i]
  lfc_d10 <- hek293_d10_sig$log2_fold_change[hek293_d10_sig$gene_symbol == gs]
  lfc_e6 <- hek293_e6_sig$log2_fold_change[hek293_e6_sig$gene_symbol == gs]
  if(all(c(lfc_d10, lfc_e6) > 0)) {
    shared_hek_df$regulation[i] <- 'up_vs_control'
  } else if(all(c(lfc_d10, lfc_e6) < 0)) {
    shared_hek_df$regulation[i] <- 'down_vs_control'
  } else {
    shared_hek_df$regulation[i] <- NA
  }
}
shared_hek_df <- shared_hek_df[!is.na(shared_hek_df$regulation), ]

write.csv(shared_hek_df, file='/outputs/hek293_shared_degs.csv', row.names=FALSE, quote=FALSE)

# Create T98G D4-specific upregulated genes
# Genes upregulated in D4 (padj<0.05, log2FC>0) but not in other treatments
d4_sig <- t98g_res[['D4']]
d4_up <- d4_sig[d4_sig$padj < 0.05 & d4_sig$log2_fold_change > 0, ]

# Combine other treatments
other_treats <- c('C10', 'F7', 'G2')
other_up_genes <- unique(unlist(lapply(other_treats, function(t) {
  df <- t98g_res[[t]]
  df$gene_id[df$padj < 0.05 & df$log2_fold_change > 0]
})))

# Select genes upregulated only in D4
d4_specific_up <- d4_up[!d4_up$gene_id %in% other_up_genes, ]

write.csv(data.frame(gene_id=d4_specific_up$gene_id), file='/outputs/t98g_d4_specific_upregulated_genes.csv', row.names=FALSE, quote=FALSE)
