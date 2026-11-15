library(DESeq2)
library(data.table)
library(stringr)

# Load data
counts <- fread('/work/counts.tsv', data.table=FALSE)
samples <- fread('/work/samples.tsv', data.table=FALSE)

# Prepare counts matrix
rownames(counts) <- counts[,1]
counts <- counts[,-1]

# Function to strip Ensembl version suffix but preserve _PAR_Y
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(id) {
    if (grepl('_PAR_Y$', id)) {
      # preserve _PAR_Y suffix, strip version before it
      sub('(ENSG\d+)(\.\d+)(_PAR_Y)$', '\1\3', id)
    } else {
      sub('(ENSG\d+)\.\d+$', '\1', id)
    }
  })
}

# Strip versions from gene IDs
gene_ids_stripped <- strip_version(rownames(counts))

# Check uniqueness
if (any(duplicated(gene_ids_stripped))) {
  stop('Non-unique gene IDs after version stripping')
}
rownames(counts) <- gene_ids_stripped

# Prepare gene symbol mapping if available
gene_symbol_map <- NULL
if (file.exists('/data/inputs_metadata.json')) {
  library(jsonlite)
  metadata <- fromJSON('/data/inputs_metadata.json')
  if (!is.null(metadata$gene_symbol_by_gene_id)) {
    # Strip versions in keys
    keys <- names(metadata$gene_symbol_by_gene_id)
    keys_stripped <- strip_version(keys)
    # Keep only unambiguous mappings
    dt_map <- data.table(gene_id=keys_stripped, gene_symbol=unlist(metadata$gene_symbol_by_gene_id))
    dt_map <- dt_map[, .(unique_symbols=unique(gene_symbol)), by=gene_id]
    ambiguous <- dt_map[, .N, by=gene_id][N>1, gene_id]
    if (length(ambiguous) > 0) {
      # Remove ambiguous
      dt_map <- dt_map[!gene_id %in% ambiguous]
    }
    gene_symbol_map <- setNames(dt_map$unique_symbols, dt_map$gene_id)
  }
}

# Add gene_symbol column function
get_gene_symbol <- function(gene_ids) {
  if (is.null(gene_symbol_map)) {
    rep('', length(gene_ids))
  } else {
    sapply(gene_ids, function(id) {
      if (id %in% names(gene_symbol_map)) {
        gene_symbol_map[[id]]
      } else {
        ''
      }
    })
  }
}

# Split samples by cell line
cell_lines <- unique(samples$cell_line)

# Prepare output containers
res_list <- list()

for (cl in cell_lines) {
  cat('Processing cell line:', cl, '\n')
  samp_cl <- samples[samples$cell_line == cl, ]
  count_cl <- counts[, samp_cl$sample_id, drop=FALSE]
  
  # Prepare colData
  coldata <- data.frame(condition=factor(samp_cl$condition))
  # Set control as reference
  if ('Ctrl' %in% levels(coldata$condition)) {
    coldata$condition <- relevel(coldata$condition, ref='Ctrl')
  } else {
    # If no Ctrl, use first level
    coldata$condition <- relevel(coldata$condition, ref=levels(coldata$condition)[1])
  }
  rownames(coldata) <- samp_cl$sample_id
  
  # Create DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(countData=count_cl, colData=coldata, design=~condition)
  
  # Run DESeq
  dds <- DESeq(dds)
  
  # Extract results for each treatment vs Ctrl
  cond_levels <- levels(coldata$condition)
  treatments <- setdiff(cond_levels, 'Ctrl')
  for (treat in treatments) {
    contrast <- c('condition', treat, 'Ctrl')
    res <- results(dds, contrast=contrast)
    res_df <- as.data.frame(res)
    res_df$gene_id <- rownames(res_df)
    res_df$gene_symbol <- get_gene_symbol(res_df$gene_id)
    
    # Rename columns to contract names
    # baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
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
    
    # Save DE results
    fname <- sprintf('/outputs/%s_%s_vs_control_degs.csv', tolower(cl), tolower(treat))
    # Contract expects gene_id as row key
    rownames(out_df) <- out_df$gene_id
    write.csv(out_df, file=fname, row.names=FALSE)
    
    # Store for shared and D4-specific gene sets if T98G and relevant
    if (cl == 'T98G') {
      res_list[[paste0(treat)]] <- out_df
    }
  }
}

# Define shared and D4-specific gene sets for T98G
# Shared: genes significant (padj<0.05) in all treatments (C10, F7, G2) with consistent direction
# D4-specific upregulated: significant upregulated only in D4

# Treatments for shared: C10, F7, G2
shared_treats <- c('c10', 'f7', 'g2')
if (all(shared_treats %in% names(res_list))) {
  shared_dfs <- lapply(shared_treats, function(x) res_list[[x]])
  # Filter significant
  sig_genes <- lapply(shared_dfs, function(df) df[df$padj < 0.05, ])
  # Intersect gene_ids
  common_genes <- Reduce(intersect, lapply(sig_genes, function(df) df$gene_id))
  # Check consistent direction
  consistent_genes <- common_genes[sapply(common_genes, function(g) {
    signs <- sapply(sig_genes, function(df) df$log2_fold_change[df$gene_id == g])
    all(signs > 0) || all(signs < 0)
  })]
  
  # Prepare shared gene list with gene_symbol and regulation
  # Use gene_symbol from first df
  gene_symbols <- sig_genes[[1]]$gene_symbol[match(consistent_genes, sig_genes[[1]]$gene_id)]
  regulation <- sapply(consistent_genes, function(g) {
    val <- sig_genes[[1]]$log2_fold_change[sig_genes[[1]]$gene_id == g]
    ifelse(val > 0, 'up_vs_control', 'down_vs_control')
  })
  shared_df <- data.frame(gene_symbol=gene_symbols, regulation=regulation, stringsAsFactors=FALSE)
  
  write.csv(shared_df, file='/outputs/t98g_slow_shared_degs.csv', row.names=FALSE)
}

# D4-specific upregulated genes
if ('d4' %in% names(res_list)) {
  d4_df <- res_list[['d4']]
  # Significant upregulated genes
  d4_up <- d4_df[(d4_df$padj < 0.05) & (d4_df$log2_fold_change > 0), ]
  # Check not significant in other treatments
  other_treats <- setdiff(names(res_list), 'd4')
  other_sig <- sapply(other_treats, function(t) {
    df <- res_list[[t]]
    df$gene_id %in% d4_up$gene_id & df$padj < 0.05
  })
  exclusive_genes <- d4_up$gene_id[!rowSums(other_sig) > 0]
  exclusive_df <- d4_up[d4_up$gene_id %in% exclusive_genes, c('gene_id')]
  write.csv(exclusive_df, file='/outputs/t98g_d4_specific_upregulated_genes.csv', row.names=FALSE)
}

# HEK293 shared DEGs similarly
hek_treats <- c('d10', 'e6')
if (all(hek_treats %in% names(res_list))) {
  hek_dfs <- lapply(hek_treats, function(x) res_list[[x]])
  sig_genes <- lapply(hek_dfs, function(df) df[df$padj < 0.05, ])
  common_genes <- Reduce(intersect, lapply(sig_genes, function(df) df$gene_id))
  consistent_genes <- common_genes[sapply(common_genes, function(g) {
    signs <- sapply(sig_genes, function(df) df$log2_fold_change[df$gene_id == g])
    all(signs > 0) || all(signs < 0)
  })]
  gene_symbols <- sig_genes[[1]]$gene_symbol[match(consistent_genes, sig_genes[[1]]$gene_id)]
  regulation <- sapply(consistent_genes, function(g) {
    val <- sig_genes[[1]]$log2_fold_change[sig_genes[[1]]$gene_id == g]
    ifelse(val > 0, 'up_vs_control', 'down_vs_control')
  })
  hek_shared_df <- data.frame(gene_symbol=gene_symbols, regulation=regulation, stringsAsFactors=FALSE)
  write.csv(hek_shared_df, file='/outputs/hek293_shared_degs.csv', row.names=FALSE)
}

# Save individual HEK293 DE results
for (cl in cell_lines) {
  if (cl == 'HEK293') {
    samp_cl <- samples[samples$cell_line == cl, ]
    count_cl <- counts[, samp_cl$sample_id, drop=FALSE]
    coldata <- data.frame(condition=factor(samp_cl$condition))
    if ('Ctrl' %in% levels(coldata$condition)) {
      coldata$condition <- relevel(coldata$condition, ref='Ctrl')
    } else {
      coldata$condition <- relevel(coldata$condition, ref=levels(coldata$condition)[1])
    }
    rownames(coldata) <- samp_cl$sample_id
    dds <- DESeqDataSetFromMatrix(countData=count_cl, colData=coldata, design=~condition)
    dds <- DESeq(dds)
    cond_levels <- levels(coldata$condition)
    treatments <- setdiff(cond_levels, 'Ctrl')
    for (treat in treatments) {
      contrast <- c('condition', treat, 'Ctrl')
      res <- results(dds, contrast=contrast)
      res_df <- as.data.frame(res)
      res_df$gene_id <- rownames(res_df)
      res_df$gene_symbol <- get_gene_symbol(res_df$gene_id)
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
      fname <- sprintf('/outputs/%s_%s_vs_control_degs.csv', tolower(cl), tolower(treat))
      write.csv(out_df, file=fname, row.names=FALSE)
    }
  }
}
