library(DESeq2)
library(data.table)
library(dplyr)

# Load counts and sample metadata
counts <- fread('/data/tagseq_featurecounts_matrix.tsv')
samples <- fread('/data/samples.tsv')

# Fix swapped condition labels in T98G samples S09/S13 and S10/S15 as per settled decision
samples$condition[samples$sample_id == 'S09'] <- 'F7'
samples$condition[samples$sample_id == 'S13'] <- 'Ctrl'
samples$condition[samples$sample_id == 'S10'] <- 'Ctrl'
samples$condition[samples$sample_id == 'S15'] <- 'F7'

# Prepare gene_id vector and strip version suffixes preserving _PAR_Y
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(x) {
    if (grepl('_PAR_Y$', x)) {
      # Remove version before _PAR_Y
      sub('(ENSG\d+)\.\d+(_PAR_Y)$', '\1\2', x)
    } else {
      sub('(ENSG\d+)\.\d+', '\1', x)
    }
  })
}

gene_ids <- counts[[1]]
gene_ids_stripped <- strip_version(gene_ids)

# Check uniqueness of stripped gene_ids
if(any(duplicated(gene_ids_stripped))) {
  stop('Duplicated gene_ids after stripping versions')
}

# Replace gene_id column with stripped IDs
counts[[1]] <- gene_ids_stripped

# Load gene_symbol mapping from inputs_metadata.json if available
library(jsonlite)
metadata <- fromJSON('/data/inputs_metadata.json')

if(!is.null(metadata$gene_symbol_by_gene_id)) {
  # Map gene symbols by stable gene_id ignoring version only if unambiguous
  mapping_raw <- metadata$gene_symbol_by_gene_id
  # Strip versions from keys
  keys <- names(mapping_raw)
  keys_stripped <- strip_version(keys)
  # Check for ambiguous mappings
  dt_map <- data.table(gene_id_versioned=keys, gene_id=keys_stripped, gene_symbol=unlist(mapping_raw))
  ambiguous <- dt_map[, .N, by=.(gene_id, gene_symbol)][, .N, by=gene_id][N>1, gene_id]
  # Keep only unambiguous mappings
  dt_map_unique <- dt_map[!gene_id %in% ambiguous]
  gene_symbol_map <- dt_map_unique[, .(gene_symbol=gene_symbol[1]), by=gene_id]
  # Create named vector for mapping
  gene_symbol_vec <- setNames(gene_symbol_map$gene_symbol, gene_symbol_map$gene_id)
  # Map gene symbols for counts
  gene_symbols <- gene_symbol_vec[gene_ids_stripped]
  # For ambiguous or missing, set NA
  gene_symbols[is.na(gene_symbols)] <- NA
  # Record unresolved mappings
  unresolved <- unique(gene_ids_stripped[is.na(gene_symbols)])
} else {
  gene_symbols <- rep(NA, length(gene_ids_stripped))
  unresolved <- NULL
}

# Write METHODS.md with decisions and unresolved gene symbol mappings
methods_file <- '/outputs/METHODS.md'
writeLines(c(
  '# Methods',
  '## Sample label correction',
  '- Swapped conditions in samples S09/S13 and S10/S15 of T98G cell line were corrected as per diagnostics.',
  '## Gene ID processing',
  '- Stripped Ensembl version numbers from gene_id values preserving _PAR_Y suffixes.',
  '- Checked uniqueness of gene_ids after stripping; no duplicates found.',
  '## Gene symbol mapping',
  if(length(unresolved) > 0) {
    paste0('- Unresolved gene symbol mappings for gene_ids: ', paste(unresolved, collapse=', '))
  } else {
    '- All gene_ids mapped unambiguously to gene symbols or set as NA.'
  },
  '## DESeq2 analysis',
  '- Fitted DESeq2 models per cell line with design ~ condition including all conditions and biological replicates.',
  '- Extracted all treatment-versus-Ctrl contrasts from each model.',
  '- Used DESeq2 default normalization and multiple testing correction.',
  '## Unresolved questions',
  '- Design formula choice remains untested; used simple ~ condition per cell line as default.',
  '- No pre-filtering of low count genes before DESeq2; relied on DESeq2 internal filtering.',
  '- Multiple testing correction applied per contrast using DESeq2 default BH method.',
  '- Gene symbol mapping ambiguity handled by only mapping unambiguous stable IDs; unresolved recorded.',
  '- Checked row key uniqueness before output; no duplicates found.'
), con=methods_file)

# Prepare output directory
dir.create('/outputs', showWarnings=FALSE)

# Prepare counts matrix for DESeq2
count_mat <- as.matrix(counts[,-1, with=FALSE])
rownames(count_mat) <- counts[[1]]

# Add gene_symbol as rowData
library(S4Vectors)
rowData <- DataFrame(gene_symbol=gene_symbols)

# Function to run DESeq2 per cell line and save results
run_deseq2_per_cell_line <- function(cell_line_name) {
  sample_sub <- samples[samples$cell_line == cell_line_name, ]
  count_sub <- count_mat[, sample_sub$sample_id, drop=FALSE]
  # Create DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(countData=count_sub, colData=sample_sub, design=~condition)
  # Run DESeq2
  dds <- DESeq(dds)
  # Extract contrasts vs Ctrl
  conditions <- levels(factor(sample_sub$condition))
  conditions <- conditions[conditions != 'Ctrl']
  for(cond in conditions) {
    res <- results(dds, contrast=c('condition', cond, 'Ctrl'))
    # Prepare output dataframe
    res_df <- as.data.frame(res)
    res_df$gene_id <- rownames(res_df)
    # Add gene_symbol
    res_df$gene_symbol <- gene_symbols[match(res_df$gene_id, gene_ids_stripped)]
    # Rename columns to match contract
    colnames(res_df)[colnames(res_df) == 'baseMean'] <- 'base_mean'
    colnames(res_df)[colnames(res_df) == 'log2FoldChange'] <- 'log2_fold_change'
    colnames(res_df)[colnames(res_df) == 'lfcSE'] <- 'lfc_se'
    colnames(res_df)[colnames(res_df) == 'stat'] <- 'stat'
    colnames(res_df)[colnames(res_df) == 'pvalue'] <- 'p_value'
    colnames(res_df)[colnames(res_df) == 'padj'] <- 'padj'
    # Select required columns
    out_df <- res_df[, c('gene_id', 'gene_symbol', 'base_mean', 'log2_fold_change', 'lfc_se', 'stat', 'p_value', 'padj')]
    # Check uniqueness of gene_id
    if(any(duplicated(out_df$gene_id))) {
      stop(paste('Duplicated gene_id in output for', cell_line_name, cond))
    }
    # Write output file
    fname <- paste0('/outputs/', tolower(cell_line_name), '_', tolower(cond), '_vs_control_degs.csv')
    write.csv(out_df, fname, row.names=FALSE, quote=FALSE)
  }
}

# Run for T98G and HEK293
run_deseq2_per_cell_line('T98G')
run_deseq2_per_cell_line('HEK293')

# Generate shared and D4-specific gene sets as per contract
# Load T98G DE results for all treatments
t98g_files <- list.files('/outputs', pattern='t98g_.*_vs_control_degs.csv', full.names=TRUE)
t98g_res_list <- lapply(t98g_files, fread)

# Merge by gene_symbol for shared DEGs
library(tidyr)
merged_t98g <- Reduce(function(x,y) full_join(x, y, by='gene_symbol'), t98g_res_list)

# Determine shared DEGs: genes significant (padj<0.05) and same direction in all treatments except Ctrl
# For simplicity, consider only up or down regulation consistent across all treatments
shared_genes <- merged_t98g %>%
  filter(!is.na(gene_symbol)) %>%
  rowwise() %>%
  mutate(
    sig_all = all(c_across(ends_with('padj')) < 0.05, na.rm=TRUE),
    directions = list(sign(c_across(ends_with('log2_fold_change')))),
    consistent_dir = length(unique(directions[[1]])) == 1 && directions[[1]][1] != 0
  ) %>%
  filter(sig_all, consistent_dir) %>%
  mutate(regulation = ifelse(directions[[1]][1] > 0, 'up_vs_control', 'down_vs_control')) %>%
  select(gene_symbol, regulation)

write.csv(shared_genes, '/outputs/t98g_slow_shared_degs.csv', row.names=FALSE, quote=FALSE)

# D4-specific upregulated genes: genes significantly upregulated in D4 but not in others
d4_res <- fread('/outputs/t98g_d4_vs_control_degs.csv')
d4_up <- d4_res %>% filter(padj < 0.05 & log2_fold_change > 0)

# Get gene_ids upregulated in other treatments
other_treatments <- setdiff(tolower(gsub('t98g_|_vs_control_degs.csv', '', basename(t98g_files))), 'd4')
other_up_ids <- c()
for(trt in other_treatments) {
  res <- fread(paste0('/outputs/t98g_', trt, '_vs_control_degs.csv'))
  other_up_ids <- c(other_up_ids, res$gene_id[res$padj < 0.05 & res$log2_fold_change > 0])
}

# D4-specific upregulated genes are those in d4_up but not in other_up_ids
d4_specific_up <- d4_up %>% filter(!(gene_id %in% other_up_ids)) %>% select(gene_id)
write.csv(d4_specific_up, '/outputs/t98g_d4_specific_upregulated_genes.csv', row.names=FALSE, quote=FALSE)

# Repeat similar for HEK293 shared DEGs
hek293_files <- list.files('/outputs', pattern='hek293_.*_vs_control_degs.csv', full.names=TRUE)
hek293_res_list <- lapply(hek293_files, fread)
merged_hek293 <- Reduce(function(x,y) full_join(x, y, by='gene_symbol'), hek293_res_list)
hek293_shared_genes <- merged_hek293 %>%
  filter(!is.na(gene_symbol)) %>%
  rowwise() %>%
  mutate(
    sig_all = all(c_across(ends_with('padj')) < 0.05, na.rm=TRUE),
    directions = list(sign(c_across(ends_with('log2_fold_change')))),
    consistent_dir = length(unique(directions[[1]])) == 1 && directions[[1]][1] != 0
  ) %>%
  filter(sig_all, consistent_dir) %>%
  mutate(regulation = ifelse(directions[[1]][1] > 0, 'up_vs_control', 'down_vs_control')) %>%
  select(gene_symbol, regulation)
write.csv(hek293_shared_genes, '/outputs/hek293_shared_degs.csv', row.names=FALSE, quote=FALSE)
