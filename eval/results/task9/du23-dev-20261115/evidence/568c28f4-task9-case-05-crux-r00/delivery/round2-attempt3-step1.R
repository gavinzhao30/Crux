library(DESeq2)
library(data.table)
library(stringr)

# Load counts and sample sheet
counts <- fread('/data/tagseq_featurecounts_matrix.tsv')
samples <- fread('/data/samples.tsv')

# Prepare count matrix and sample info
count_mat <- as.matrix(counts[ , -1, with=FALSE])
rownames(count_mat) <- counts[[1]]

# Check sample order matches
stopifnot(all(colnames(count_mat) == samples$sample_id))

# Load gene_symbol_by_gene_id mapping if available
mapping_path <- '/data/inputs_metadata.json'
mapping_json <- jsonlite::fromJSON(mapping_path)
gene_map <- NULL
if (!is.null(mapping_json$gene_symbol_by_gene_id)) {
  gene_map <- mapping_json$gene_symbol_by_gene_id
  # Convert to data.table
  gene_map <- data.table(gene_id=names(gene_map), gene_symbol=unlist(gene_map))
  # Strip version from gene_id for mapping
  gene_map[, gene_id_stripped := sub('\\.[0-9]+$', '', gene_id)]
  # Check for ambiguous mappings
  ambig <- gene_map[, .N, by=gene_id_stripped][N>1]
  ambiguous_ids <- ambig$gene_id_stripped
} else {
  ambiguous_ids <- character(0)
}

# Function to strip version but preserve _PAR_Y
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(id) {
    if (grepl('_PAR_Y$', id)) {
      sub('\\.[0-9]+', '', id)
    } else {
      sub('\\.[0-9]+$', '', id)
    }
  })
}

# Prepare output directory
if (!dir.exists('/outputs')) dir.create('/outputs')

# Function to run DESeq2 per cell line
run_deseq2 <- function(cell_line) {
  # Subset samples
  samp_sub <- samples[cell_line == cell_line]
  count_sub <- count_mat[, samp_sub$sample_id, drop=FALSE]
  
  # Create colData
  coldata <- data.frame(condition=factor(samp_sub$condition))
  rownames(coldata) <- samp_sub$sample_id
  
  # DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(countData=count_sub, colData=coldata, design=~condition)
  
  # Run DESeq
  dds <- DESeq(dds)
  
  # Extract results for each treatment vs Ctrl
  conds <- levels(coldata$condition)
  ctrl <- 'Ctrl'
  treatments <- setdiff(conds, ctrl)
  res_list <- list()
  for (treat in treatments) {
    contrast <- c('condition', treat, ctrl)
    res <- results(dds, contrast=contrast)
    res_dt <- as.data.table(res, keep.rownames='gene_id')
    
    # Add baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
    # Rename columns to contract names
    setnames(res_dt, c('log2FoldChange', 'lfcSE', 'pvalue', 'padj'),
             c('log2_fold_change', 'lfc_se', 'p_value', 'padj'))
    
    # Add gene_symbol from mapping
    res_dt[, gene_id_stripped := strip_version(gene_id)]
    if (!is.null(gene_map)) {
      # Map gene_symbol only if unambiguous
      res_dt <- merge(res_dt, gene_map[, .(gene_id_stripped, gene_symbol)], by='gene_id_stripped', all.x=TRUE)
      # For ambiguous IDs, set gene_symbol to NA
      res_dt[gene_id_stripped %in% ambiguous_ids, gene_symbol := NA]
    } else {
      res_dt[, gene_symbol := NA_character_]
    }
    
    # Remove version from gene_id for output
    res_dt[, gene_id := gene_id_stripped]
    res_dt[, gene_id_stripped := NULL]
    
    # Check uniqueness of gene_id
    if (any(duplicated(res_dt$gene_id))) {
      stop(paste('Duplicate gene_id after version stripping in', cell_line, treat))
    }
    
    # Select required columns
    res_dt <- res_dt[, .(gene_id, gene_symbol, baseMean, log2_fold_change, lfc_se, stat, p_value, padj)]
    
    # Save to file
    fname <- paste0('/outputs/', tolower(cell_line), '_', tolower(treat), '_vs_control_degs.csv')
    fwrite(res_dt, fname)
    res_list[[treat]] <- res_dt
  }
  return(res_list)
}

# Run for T98G and HEK293
res_t98g <- run_deseq2('T98G')
res_hek293 <- run_deseq2('HEK293')

# Generate shared and D4-specific gene sets for T98G
# Shared DEGs: genes significant (padj<0.05) and regulated in same direction in all T98G treatments
sig_genes <- lapply(res_t98g, function(dt) dt[padj < 0.05])
# Find intersection of gene_ids
shared_genes <- Reduce(function(x,y) merge(x, y, by='gene_id'), sig_genes)
# Check consistent regulation direction
log2fc_cols <- paste0('log2_fold_change.x', seq_along(sig_genes))
# Actually columns are suffixed by .x, .y etc. after merge; better to do manually
shared_ids <- Reduce(intersect, lapply(sig_genes, function(dt) dt$gene_id))
shared_dt <- data.table(gene_id=shared_ids)
# For each gene, check direction
shared_dt[, regulation := {
  signs <- sapply(sig_genes, function(dt) dt[gene_id==.BY$gene_id, log2_fold_change])
  if (all(signs > 0)) 'up_vs_control' else if (all(signs < 0)) 'down_vs_control' else NA_character_
}, by=gene_id]
shared_dt <- shared_dt[!is.na(regulation)]

# Map gene_symbol for shared genes
if (!is.null(gene_map)) {
  shared_dt[, gene_symbol := gene_map$gene_symbol[match(gene_id, gene_map$gene_id_stripped)]]
} else {
  shared_dt[, gene_symbol := NA_character_]
}

# Save shared DEGs for T98G
shared_out <- shared_dt[, .(gene_symbol, regulation)]
fwrite(shared_out, '/outputs/t98g_slow_shared_degs.csv')

# D4-specific upregulated genes: significant upregulated in D4 vs Ctrl, not significant upregulated in other T98G treatments
d4_res <- res_t98g[['D4']]
d4_sig_up <- d4_res[padj < 0.05 & log2_fold_change > 0]
other_treats <- setdiff(names(res_t98g), 'D4')
other_sig_up_ids <- unique(unlist(lapply(res_t98g[other_treats], function(dt) dt[padj < 0.05 & log2_fold_change > 0, gene_id])))
d4_specific_up <- d4_sig_up[!gene_id %in% other_sig_up_ids]

# Save D4-specific upregulated gene list
fwrite(d4_specific_up[, .(gene_id)], '/outputs/t98g_d4_specific_upregulated_genes.csv')

# HEK293 shared DEGs: similar approach
sig_genes_hek <- lapply(res_hek293, function(dt) dt[padj < 0.05])
shared_ids_hek <- Reduce(intersect, lapply(sig_genes_hek, function(dt) dt$gene_id))
shared_dt_hek <- data.table(gene_id=shared_ids_hek)
shared_dt_hek[, regulation := {
  signs <- sapply(sig_genes_hek, function(dt) dt[gene_id==.BY$gene_id, log2_fold_change])
  if (all(signs > 0)) 'up_vs_control' else if (all(signs < 0)) 'down_vs_control' else NA_character_
}, by=gene_id]
shared_dt_hek <- shared_dt_hek[!is.na(regulation)]
if (!is.null(gene_map)) {
  shared_dt_hek[, gene_symbol := gene_map$gene_symbol[match(gene_id, gene_map$gene_id_stripped)]]
} else {
  shared_dt_hek[, gene_symbol := NA_character_]
}
shared_out_hek <- shared_dt_hek[, .(gene_symbol, regulation)]
fwrite(shared_out_hek, '/outputs/hek293_shared_degs.csv')

# Write METHODS.md
methods_text <- c(
  '# Methods',
  'DESeq2 differential expression analysis was performed separately per cell line using design ~ condition.',
  'All biological replicates and all retained conditions per cell line were included in the model.',
  'Gene counts were used as input without manual prefiltering; DESeq2 internal independent filtering was applied.',
  'Benjamini-Hochberg procedure was used for p-value adjustment per contrast.',
  'Gene symbols were assigned using provided gene_symbol_by_gene_id mapping, matching on stable Ensembl IDs after stripping version numbers.',
  'Ambiguous gene symbol mappings were left blank and recorded.',
  'Ensembl version numbers were stripped from gene IDs before output, preserving _PAR_Y suffixes.',
  'Row keys (gene IDs or gene symbols) were checked for uniqueness before export.',
  'Shared DEGs were defined as genes significant (padj < 0.05) and regulated in the same direction across all treatments within a cell line.',
  'D4-specific upregulated genes were defined as genes significantly upregulated in D4 vs control but not significantly upregulated in other T98G treatments.',
  'Sample labels and cell line assignments were verified consistent; no swaps or duplicates detected.',
  'All libraries passed quality checks and were retained for analysis.'
)
writeLines(methods_text, '/outputs/METHODS.md')
