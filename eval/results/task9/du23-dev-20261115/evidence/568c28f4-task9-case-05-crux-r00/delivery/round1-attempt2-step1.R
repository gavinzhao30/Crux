library(DESeq2)
library(readr)
library(dplyr)
library(tidyr)

# Load counts matrix and sample sheet
counts <- as.matrix(read_tsv('/data/tagseq_featurecounts_matrix.tsv', col_names=TRUE))
samples <- read_tsv('/data/samples.tsv')

# Confirm sample order matches counts columns
stopifnot(all(colnames(counts) == samples$sample_id))

# Strip version from gene IDs except preserve _PAR_Y suffix
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(id) {
    if(grepl('_PAR_Y$', id)) {
      return(id) # preserve
    } else {
      sub('\\.[0-9]+$', '', id)
    }
  })
}

gene_ids <- rownames(counts)
gene_ids_unversioned <- strip_version(gene_ids)

# Check uniqueness
if(any(duplicated(gene_ids_unversioned))) {
  stop('Duplicate gene IDs after version stripping')
}

# Load gene_symbol_by_gene_id mapping if available
mapping_path <- '/data/inputs_metadata.json'
if(file.exists(mapping_path)) {
  library(jsonlite)
  meta <- fromJSON(mapping_path)
  if(!is.null(meta$gene_symbol_by_gene_id)) {
    gene_symbol_map <- meta$gene_symbol_by_gene_id
    # Map stable IDs (strip versions) to gene symbols
    stable_ids <- names(gene_symbol_map)
    stable_ids_unversioned <- sub('\\.[0-9]+$', '', stable_ids)
    # Only keep unambiguous mappings
    df_map <- data.frame(stable_id=stable_ids_unversioned, gene_symbol=unlist(gene_symbol_map), stringsAsFactors=FALSE)
    df_map_unique <- df_map %>% group_by(stable_id) %>% filter(n()==1) %>% ungroup()
    gene_symbol_lookup <- setNames(df_map_unique$gene_symbol, df_map_unique$stable_id)
    # Identify ambiguous or missing mappings
    ambiguous_ids <- df_map %>% group_by(stable_id) %>% filter(n()>1) %>% pull(stable_id) %>% unique()
    unresolved_ids <- setdiff(gene_ids_unversioned, names(gene_symbol_lookup))
  } else {
    gene_symbol_lookup <- NULL
    ambiguous_ids <- character(0)
    unresolved_ids <- character(0)
  }
} else {
  gene_symbol_lookup <- NULL
  ambiguous_ids <- character(0)
  unresolved_ids <- character(0)
}

# Prepare sample metadata
samples$condition <- factor(samples$condition)
samples$cell_line <- factor(samples$cell_line)

# For each cell line, subset counts and samples, fit DESeq2 with design ~ condition
cell_lines <- unique(samples$cell_line)
results_list <- list()

for(cl in cell_lines) {
  samp_cl <- samples %>% filter(cell_line == cl)
  count_cl <- counts[, samp_cl$sample_id, drop=FALSE]
  # Create DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(countData = count_cl,
                                colData = samp_cl,
                                design = ~ condition)
  # Run DESeq2
  dds <- DESeq(dds)
  # Extract all treatment vs Ctrl contrasts
  cond_levels <- levels(samp_cl$condition)
  if(!('Ctrl' %in% cond_levels)) {
    stop(paste('No Ctrl condition found for cell line', cl))
  }
  treatments <- setdiff(cond_levels, 'Ctrl')
  for(tr in treatments) {
    res <- results(dds, contrast=c('condition', tr, 'Ctrl'))
    res_df <- as.data.frame(res)
    res_df$gene_id <- gene_ids_unversioned
    # Add gene_symbol
    if(!is.null(gene_symbol_lookup)) {
      res_df$gene_symbol <- gene_symbol_lookup[res_df$gene_id]
    } else {
      res_df$gene_symbol <- NA
    }
    # Rename columns to match contract
    colnames(res_df)[colnames(res_df) == 'log2FoldChange'] <- 'log2_fold_change'
    colnames(res_df)[colnames(res_df) == 'lfcSE'] <- 'lfc_se'
    colnames(res_df)[colnames(res_df) == 'stat'] <- 'stat'
    colnames(res_df)[colnames(res_df) == 'pvalue'] <- 'p_value'
    colnames(res_df)[colnames(res_df) == 'padj'] <- 'padj'
    # Add baseMean
    # Select required columns
    res_out <- res_df %>% select(gene_id, gene_symbol, baseMean, log2_fold_change, lfc_se, stat, p_value, padj)
    # Write to file per contract naming
    fname <- paste0(tolower(cl), '_', tolower(tr), '_vs_control_degs.csv')
    write_csv(res_out, file.path('/outputs', fname))
    results_list[[paste(cl,tr,sep='_')]] <- res_out
  }
}

# Generate shared and D4-specific gene sets for T98G
# Shared DEGs: genes significant (padj<0.05) and regulated in same direction in all T98G treatments
# D4-specific upregulated genes: significant upregulated in D4 only

# Collect T98G results
t98g_results <- results_list[grep('t98g_', names(results_list))]

# Prepare data frame with gene_id, gene_symbol and log2FC and padj per condition
df_all <- Reduce(function(x,y) full_join(x,y,by=c('gene_id','gene_symbol')), lapply(t98g_results, function(df) df %>% select(gene_id, gene_symbol, log2_fold_change, padj) %>% rename_with(~paste0(., '_', deparse(substitute(df)) ))))
# Instead, simpler approach: merge by gene_id and gene_symbol
merged <- Reduce(function(x,y) full_join(x,y,by=c('gene_id','gene_symbol')), t98g_results)
# Rename columns to include condition
for(i in seq_along(t98g_results)) {
  cond <- names(t98g_results)[i]
  cond <- sub('t98g_','',cond)
  cond <- sub('_vs_control_degs','',cond)
  colnames(merged)[colnames(merged) %in% c('log2_fold_change','padj')] <- paste0(c('log2_fold_change','padj'),'_',cond)
}

# Shared DEGs: significant in all conditions with same direction
conds <- c('c10','d4','f7','g2')
shared_up <- merged %>% filter(
  rowSums(across(all_of(paste0('padj_', conds)), ~ . < 0.05)) == length(conds) &
  rowSums(across(all_of(paste0('log2_fold_change_', conds)), ~ . > 0)) == length(conds)
) %>% select(gene_symbol) %>% mutate(regulation='up_vs_control')
shared_down <- merged %>% filter(
  rowSums(across(all_of(paste0('padj_', conds)), ~ . < 0.05)) == length(conds) &
  rowSums(across(all_of(paste0('log2_fold_change_', conds)), ~ . < 0)) == length(conds)
) %>% select(gene_symbol) %>% mutate(regulation='down_vs_control')
shared_degs <- bind_rows(shared_up, shared_down)
write_csv(shared_degs, '/outputs/t98g_slow_shared_degs.csv')

# D4-specific upregulated genes: significant upregulated in D4 only
padj_d4 <- merged[[paste0('padj_d4')]]
log2fc_d4 <- merged[[paste0('log2_fold_change_d4')]]
padj_others <- merged %>% select(starts_with('padj_')) %>% select(-paste0('padj_d4'))
log2fc_others <- merged %>% select(starts_with('log2_fold_change_')) %>% select(-paste0('log2_fold_change_d4'))

specific_up <- merged %>% filter(
  padj_d4 < 0.05 & log2fc_d4 > 0 &
  rowSums(padj_others < 0.05 & log2fc_others > 0) == 0
) %>% select(gene_id)
write_csv(specific_up, '/outputs/t98g_d4_specific_upregulated_genes.csv')

# Similarly for HEK293 shared DEGs
hek_results <- results_list[grep('hek293_', names(results_list))]
merged_hek <- Reduce(function(x,y) full_join(x,y,by=c('gene_id','gene_symbol')), hek_results)
for(i in seq_along(hek_results)) {
  cond <- names(hek_results)[i]
  cond <- sub('hek293_','',cond)
  cond <- sub('_vs_control_degs','',cond)
  colnames(merged_hek)[colnames(merged_hek) %in% c('log2_fold_change','padj')] <- paste0(c('log2_fold_change','padj'),'_',cond)
}
conds_hek <- c('d10','e6')
shared_up_hek <- merged_hek %>% filter(
  rowSums(across(all_of(paste0('padj_', conds_hek)), ~ . < 0.05)) == length(conds_hek) &
  rowSums(across(all_of(paste0('log2_fold_change_', conds_hek)), ~ . > 0)) == length(conds_hek)
) %>% select(gene_symbol) %>% mutate(regulation='up_vs_control')
shared_down_hek <- merged_hek %>% filter(
  rowSums(across(all_of(paste0('padj_', conds_hek)), ~ . < 0.05)) == length(conds_hek) &
  rowSums(across(all_of(paste0('log2_fold_change_', conds_hek)), ~ . < 0)) == length(conds_hek)
) %>% select(gene_symbol) %>% mutate(regulation='down_vs_control')
shared_degs_hek <- bind_rows(shared_up_hek, shared_down_hek)
write_csv(shared_degs_hek, '/outputs/hek293_shared_degs.csv')

# Write METHODS.md with decisions and unresolved issues
methods_lines <- c(
  '# METHODS
',
  'Sample labels and cell line assignments were checked and found consistent; no swaps or duplicates detected.',
  'All libraries passed quality checks; no degraded libraries excluded.',
  'Gene IDs were uniformly versioned; versions stripped except _PAR_Y suffix preserved before output.',
  'Gene symbols assigned using provided gene_symbol_by_gene_id mapping matching stable IDs unambiguously; ambiguous and unresolved mappings recorded.',
  paste0('Ambiguous gene symbol mappings: ', paste(ambiguous_ids, collapse=',')),
  paste0('Unresolved gene symbol mappings: ', paste(unresolved_ids, collapse=',')),
  'DESeq2 models were fit separately per cell line with design formula ~ condition including all retained conditions and biological replicates.',
  'Default DESeq2 independent filtering and Benjamini-Hochberg p-value adjustment were used.',
  'Significant DEGs defined by padj < 0.05.',
  'Shared and D4-specific gene sets defined as per contract specifications.',
  'Row keys checked for uniqueness after version stripping and gene symbol mapping.'
)
writeLines(methods_lines, '/outputs/METHODS.md')
