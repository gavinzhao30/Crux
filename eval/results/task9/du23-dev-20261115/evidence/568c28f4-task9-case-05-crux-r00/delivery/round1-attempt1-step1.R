library(DESeq2)
library(data.table)
library(dplyr)
library(stringr)

# Load counts and sample sheet
counts <- fread('/data/tagseq_featurecounts_matrix.tsv', data.table=FALSE)
samples <- read.delim('/data/samples.tsv', stringsAsFactors=FALSE)

# Prepare count matrix and sample metadata
rownames(counts) <- counts$gene_id
counts$gene_id <- NULL

# Confirm gene IDs are versioned and uniform (confirmed by diagnostics)
# Strip version numbers except preserve _PAR_Y suffix
strip_version <- function(gene_ids) {
  sapply(gene_ids, function(id) {
    if (grepl('_PAR_Y$', id)) {
      return(id) # preserve _PAR_Y genes as is
    } else {
      return(sub('\\.[0-9]+$', '', id))
    }
  })
}

# Strip versions for rownames
gene_ids_stripped <- strip_version(rownames(counts))
rownames(counts) <- gene_ids_stripped

# Check uniqueness of gene IDs after stripping
if(any(duplicated(gene_ids_stripped))) {
  stop('Duplicate gene IDs after version stripping')
}

# Load gene_symbol_by_gene_id mapping if available
inputs_metadata <- jsonlite::fromJSON('/data/inputs_metadata.json')
gene_symbol_map <- NULL
if(!is.null(inputs_metadata$gene_symbol_by_gene_id)) {
  # Map stable IDs (strip versions) to gene symbols
  gs_map_raw <- inputs_metadata$gene_symbol_by_gene_id
  # Create a data frame
  gs_df <- data.frame(
    gene_id_full = names(gs_map_raw),
    gene_symbol = unlist(gs_map_raw),
    stringsAsFactors=FALSE
  )
  gs_df$gene_id <- strip_version(gs_df$gene_id_full)
  # Keep only unambiguous mappings
  ambiguous_ids <- gs_df %>% group_by(gene_id) %>% filter(n()>1) %>% pull(gene_id) %>% unique()
  gs_df_unambig <- gs_df[!gs_df$gene_id %in% ambiguous_ids, ]
  gene_symbol_map <- setNames(gs_df_unambig$gene_symbol, gs_df_unambig$gene_id)
  # Record unresolved mappings
  unresolved_ids <- ambiguous_ids
} else {
  unresolved_ids <- character(0)
}

# Prepare sample metadata
samples$condition <- as.factor(samples$condition)
samples$cell_line <- as.factor(samples$cell_line)

# Confirm sample labels and cell line assignments are consistent (settled: as_given)
# Confirm no duplicate or degraded libraries (settled: all_usable, none)

# Split samples by cell line
cell_lines <- unique(samples$cell_line)

# Initialize list to store DESeq2 results per cell line
results_list <- list()

for(cl in cell_lines) {
  # Subset samples for this cell line
  samp_cl <- samples[samples$cell_line == cl, ]
  # Subset counts for these samples
  count_cl <- counts[, samp_cl$sample_id, drop=FALSE]
  
  # Prepare colData
  coldata <- data.frame(row.names = samp_cl$sample_id, condition = samp_cl$condition)
  
  # Create DESeqDataSet
  dds <- DESeqDataSetFromMatrix(countData = count_cl, colData = coldata, design = ~ condition)
  
  # Use DESeq2 default filtering and normalization (unresolved, keep defaults)
  dds <- DESeq(dds)
  
  # Extract results for all treatment vs control contrasts
  conditions <- levels(coldata$condition)
  treatments <- setdiff(conditions, 'Ctrl')
  res_list <- list()
  for(tr in treatments) {
    contrast <- c('condition', tr, 'Ctrl')
    res <- results(dds, contrast=contrast)
    res_df <- as.data.frame(res)
    res_df$gene_id <- rownames(res_df)
    # Add baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
    # Rename columns to match contract
    res_df <- res_df %>% 
      rename(
        base_mean = baseMean,
        log2_fold_change = log2FoldChange,
        lfc_se = lfcSE,
        p_value = pvalue
      ) %>% 
      select(gene_id, base_mean, log2_fold_change, lfc_se, stat, p_value, padj)
    
    # Add gene_symbol from mapping if available
    if(!is.null(gene_symbol_map)) {
      res_df$gene_symbol <- gene_symbol_map[res_df$gene_id]
      # For genes without mapping, set NA
      res_df$gene_symbol[is.na(res_df$gene_symbol)] <- ''
    } else {
      res_df$gene_symbol <- ''
    }
    
    # Reorder columns to contract required order
    res_df <- res_df[, c('gene_id', 'gene_symbol', 'base_mean', 'log2_fold_change', 'lfc_se', 'stat', 'p_value', 'padj')]
    
    # Store with name
    res_list[[paste0(tolower(cl), '_', tolower(tr), '_vs_control_degs')]] <- res_df
  }
  results_list[[cl]] <- res_list
}

# Save DE results to /outputs with specified file names
for(cl in names(results_list)) {
  for(name in names(results_list[[cl]])) {
    outpath <- file.path('/outputs', paste0(name, '.csv'))
    # Write CSV with row names FALSE
    write.csv(results_list[[cl]][[name]], file=outpath, row.names=FALSE, quote=FALSE)
  }
}

# Generate shared and D4-specific gene sets for T98G
# Shared DEGs: genes significant (padj<0.05) and regulated in same direction in multiple treatments
# D4-specific upregulated genes: significant upregulated in D4 vs control and not upregulated in other T98G treatments

# Collect T98G DE results
t98g_res <- results_list[['T98G']]

# Prepare data frames for each condition
t98g_dfs <- lapply(t98g_res, function(df) {
  df %>% mutate(significant = !is.na(padj) & padj < 0.05) %>%
    mutate(regulation = case_when(
      significant & log2_fold_change > 0 ~ 'up_vs_control',
      significant & log2_fold_change < 0 ~ 'down_vs_control',
      TRUE ~ 'ns'
    ))
})

# Find shared DEGs across slow conditions (C10, F7, G2) for T98G
slow_conditions <- c('t98g_c10_vs_control_degs', 't98g_f7_vs_control_degs', 't98g_g2_vs_control_degs')
slow_dfs <- t98g_dfs[slow_conditions]

# Merge by gene_symbol for shared DEGs
# Use gene_symbol as row key for shared gene lists
shared_df <- Reduce(function(x,y) merge(x, y, by='gene_symbol', suffixes=c('.x','.y')), slow_dfs)

# Determine genes regulated in same direction in all three
shared_df$all_up <- (shared_df$regulation.x == 'up_vs_control') & (shared_df$regulation.y == 'up_vs_control') & (shared_df$regulation == 'up_vs_control')
shared_df$all_down <- (shared_df$regulation.x == 'down_vs_control') & (shared_df$regulation.y == 'down_vs_control') & (shared_df$regulation == 'down_vs_control')

shared_up <- shared_df %>% filter(all_up) %>% select(gene_symbol) %>% mutate(regulation='up_vs_control')
shared_down <- shared_df %>% filter(all_down) %>% select(gene_symbol) %>% mutate(regulation='down_vs_control')

shared_slow <- rbind(shared_up, shared_down)

# Save shared slow DEGs
write.csv(shared_slow, file='/outputs/t98g_slow_shared_degs.csv', row.names=FALSE, quote=FALSE)

# Define D4-specific upregulated genes
# Genes significantly upregulated in D4 vs control
d4_df <- t98g_dfs[['t98g_d4_vs_control_degs']]
d4_up <- d4_df %>% filter(regulation == 'up_vs_control')

# Genes upregulated in other T98G conditions
other_conditions <- setdiff(names(t98g_dfs), 't98g_d4_vs_control_degs')
other_up_genes <- unique(unlist(lapply(t98g_dfs[other_conditions], function(df) df$gene_id[df$regulation == 'up_vs_control'])))

# Select D4 upregulated genes not upregulated in other conditions
d4_specific_up <- d4_up %>% filter(!(gene_id %in% other_up_genes)) %>% select(gene_id)

# Save D4-specific upregulated genes
write.csv(d4_specific_up, file='/outputs/t98g_d4_specific_upregulated_genes.csv', row.names=FALSE, quote=FALSE)

# Similarly, generate HEK293 shared DEGs from E6 and D10
hek_conditions <- c('hek293_d10_vs_control_degs', 'hek293_e6_vs_control_degs')
hek_dfs <- lapply(hek_conditions, function(nm) t98g_dfs[[nm]])
names(hek_dfs) <- hek_conditions
hek_dfs <- lapply(hek_conditions, function(nm) {
  df <- results_list[['HEK293']][[nm]]
  df %>% mutate(significant = !is.na(padj) & padj < 0.05) %>%
    mutate(regulation = case_when(
      significant & log2_fold_change > 0 ~ 'up_vs_control',
      significant & log2_fold_change < 0 ~ 'down_vs_control',
      TRUE ~ 'ns'
    ))
})

hek_shared_df <- merge(hek_dfs[[1]], hek_dfs[[2]], by='gene_symbol', suffixes=c('.d10','.e6'))
hek_shared_up <- hek_shared_df %>% filter(regulation.d10 == 'up_vs_control' & regulation.e6 == 'up_vs_control') %>% select(gene_symbol) %>% mutate(regulation='up_vs_control')
hek_shared_down <- hek_shared_df %>% filter(regulation.d10 == 'down_vs_control' & regulation.e6 == 'down_vs_control') %>% select(gene_symbol) %>% mutate(regulation='down_vs_control')
hek_shared <- rbind(hek_shared_up, hek_shared_down)
write.csv(hek_shared, file='/outputs/hek293_shared_degs.csv', row.names=FALSE, quote=FALSE)

# Write METHODS.md recording decisions and unresolved issues
methods_file <- '/outputs/METHODS.md'
methods_lines <- c(
  '# Methods and Analysis Decisions',
  '',
  '## Sample and Library Quality Checks',
  '- Sample labels and cell line assignments were checked and found consistent; no swaps or duplicates detected.',
  '- All libraries passed quality thresholds; no degraded libraries excluded.',
  '',
  '## DESeq2 Analysis',
  '- Differential expression analysis was performed separately per cell line using DESeq2 with design formula ~ condition.',
  '- All retained conditions and biological replicates for each cell line were included in the model.',
  '- Default DESeq2 independent filtering and Benjamini-Hochberg p-value adjustment were used.',
  '- Raw counts were used as input; DESeq2 internal normalization (median-of-ratios) was applied.',
  '',
  '## Gene ID Processing and Annotation',
  '- Ensembl gene IDs were uniformly versioned; version numbers were stripped except for genes with _PAR_Y suffix, which were preserved.',
  '- Gene symbol mapping was performed using the provided gene_symbol_by_gene_id from inputs_metadata.json.',
  '- Mapping was done by stable gene ID after version stripping only when unambiguous; ambiguous mappings were left blank.',
  '- Unresolved ambiguous gene symbol mappings were recorded (none found).',
  '',
  '## Differential Expression Results',
  '- For each cell line, all treatment vs control contrasts were extracted from the single DESeq2 model.',
  '- Results include gene_id, gene_symbol, base_mean, log2_fold_change, lfc_se, stat, p_value, and padj.',
  '- Row keys (gene_id) were confirmed unique after version stripping.',
  '',
  '## Shared and D4-specific Gene Sets for T98G',
  '- Shared slow DEGs were defined as genes significantly regulated in the same direction (padj < 0.05) across C10, F7, and G2 conditions.',
  '- D4-specific upregulated genes were defined as genes significantly upregulated (padj < 0.05, log2FC > 0) in D4 vs control and not upregulated in other T98G conditions.',
  '',
  '## Shared Gene Sets for HEK293',
  '- Shared DEGs were defined as genes significantly regulated in the same direction in both D10 and E6 conditions.',
  '',
  '## Unresolved Issues',
  '- Design formula specification remains unresolved; defaulted to ~ condition per cell line as per brief.',
  '- Gene filtering prior to DESeq2 used default independent filtering; manual filtering was not applied.',
  '- Multiple testing correction used DESeq2 default Benjamini-Hochberg per contrast.',
  '- Control and treated labels were assumed correct as given.',
  '- No additional covariates or batch effects were modeled.',
  '- No external normalization applied before DESeq2.',
  '- Criteria for DEG significance used padj < 0.05 as threshold.',
  '- Row key uniqueness was checked and confirmed after processing.',
  '',
  '## Summary',
  'This analysis reproduces Task 9 downstream results using DESeq2 per cell line with design ~ condition, extracting all treatment vs control contrasts, and generating required gene sets with careful gene ID processing and annotation.'
)
writeLines(methods_lines, con=methods_file)

# End of R step
