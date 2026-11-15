dir.create('/outputs', showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(DESeq2)
})

# ---------- Inputs ----------
counts_path <- '/data/tagseq_featurecounts_matrix.tsv'
samples_path <- '/data/samples.tsv'

counts_raw <- read.delim(counts_path, check.names = FALSE, stringsAsFactors = FALSE)
samples <- read.delim(samples_path, check.names = FALSE, stringsAsFactors = FALSE)

# Identify gene-id column and optional symbol column robustly.
gene_col <- intersect(c('gene_id','Geneid','GeneID','gene','Gene','ensembl_gene_id'), names(counts_raw))[1]
if (is.na(gene_col)) gene_col <- names(counts_raw)[1]
symbol_col <- intersect(c('gene_symbol','gene_name','symbol','Symbol','external_gene_name'), names(counts_raw))[1]

sample_col <- intersect(c('sample_id','sample','library','SampleID'), names(samples))[1]
cell_col <- intersect(c('cell_line','cellline','CellLine'), names(samples))[1]
cond_col <- intersect(c('condition','Condition','treatment','Treatment'), names(samples))[1]
if (any(is.na(c(sample_col, cell_col, cond_col)))) stop('samples.tsv lacks required sample_id/cell_line/condition columns')

sample_ids <- samples[[sample_col]]
missing_samples <- setdiff(sample_ids, names(counts_raw))
if (length(missing_samples) > 0) stop(paste('Count matrix missing samples:', paste(missing_samples, collapse=', ')))

# Gene IDs: settled diagnostic found a uniform versioned Ensembl namespace with no stripping collisions.
# Use IDs as given for DE row keys; also keep unversioned IDs internally for possible fallback symbols.
gene_id <- as.character(counts_raw[[gene_col]])
stable_id <- sub('\\.[0-9]+$', '', gene_id)

# Fill gene_symbol from supplied annotation if present; otherwise use stable Ensembl IDs as deterministic symbols.
if (!is.na(symbol_col)) {
  gene_symbol <- as.character(counts_raw[[symbol_col]])
  gene_symbol[is.na(gene_symbol) | gene_symbol == ''] <- stable_id[is.na(gene_symbol) | gene_symbol == '']
} else {
  gene_symbol <- stable_id
}

counts <- as.matrix(counts_raw[, sample_ids, drop = FALSE])
storage.mode(counts) <- 'integer'
rownames(counts) <- gene_id

# Collapse duplicated row names if any, preserving a representative symbol. The provided diagnostics reported no
# strip-version collisions; this is only a safety guard for exact duplicate row names.
if (any(duplicated(rownames(counts)))) {
  dfc <- data.frame(gene_id = rownames(counts), counts, check.names = FALSE)
  agg <- aggregate(. ~ gene_id, data = dfc, FUN = sum)
  rownames(agg) <- agg$gene_id
  counts <- as.matrix(agg[, sample_ids, drop = FALSE])
  storage.mode(counts) <- 'integer'
  sym_map <- gene_symbol[match(rownames(counts), gene_id)]
  gene_id <- rownames(counts)
  stable_id <- sub('\\.[0-9]+$', '', gene_id)
  gene_symbol <- sym_map
  gene_symbol[is.na(gene_symbol) | gene_symbol == ''] <- stable_id[is.na(gene_symbol) | gene_symbol == '']
}

anno <- data.frame(gene_id = rownames(counts), gene_symbol = gene_symbol, stringsAsFactors = FALSE)

# Decisions applied:
# - sample labels/cell lines as given; no duplicate/degraded libraries removed.
# - unresolved design question: top option was not actionable ('None of these options'), so apply method guidance:
#   cell-line-specific biological-replicate DESeq2 analyses with design ~ condition.
# - unresolved filtering: use the top actionable option, keep genes with total count >= 10 in the subset.
# - unresolved multiplicity/normalization: DESeq2 defaults (BH adjusted p-values and median-of-ratios size factors).
# - unresolved direction inversion: top option says labels are as given.

samples$cell_line <- samples[[cell_col]]
samples$condition <- samples[[cond_col]]
samples$sample_id <- samples[[sample_col]]

make_de <- function(cell_line, treatment, outfile) {
  ss <- samples[samples$cell_line == cell_line & samples$condition %in% c('Ctrl', treatment), , drop = FALSE]
  ss$condition <- factor(ss$condition, levels = c('Ctrl', treatment))
  rownames(ss) <- ss$sample_id
  cts <- counts[, ss$sample_id, drop = FALSE]

  keep <- rowSums(cts) >= 10
  cts_f <- cts[keep, , drop = FALSE]

  dds <- DESeqDataSetFromMatrix(countData = cts_f, colData = ss, design = ~ condition)
  dds <- DESeq(dds, quiet = TRUE)
  res <- results(dds, contrast = c('condition', treatment, 'Ctrl'))
  resdf <- as.data.frame(res)
  resdf$gene_id <- rownames(resdf)
  resdf <- merge(anno, resdf, by = 'gene_id', all.y = TRUE, sort = FALSE)
  resdf <- resdf[, c('gene_id','gene_symbol','baseMean','log2FoldChange','lfcSE','stat','pvalue','padj')]
  names(resdf) <- c('gene_id','gene_symbol','base_mean','log2_fold_change','lfc_se','stat','p_value','padj')
  resdf <- resdf[order(resdf$padj, resdf$p_value, decreasing = FALSE, na.last = TRUE), ]
  write.csv(resdf, file.path('/outputs', outfile), row.names = FALSE, quote = TRUE, na = '')
  resdf
}

# ---------- Differential-expression tables ----------
de <- list()
de$t98g_c10 <- make_de('T98G', 'C10', 't98g_c10_vs_control_degs.csv')
de$t98g_f7  <- make_de('T98G', 'F7',  't98g_f7_vs_control_degs.csv')
de$t98g_g2  <- make_de('T98G', 'G2',  't98g_g2_vs_control_degs.csv')
de$t98g_d4  <- make_de('T98G', 'D4',  't98g_d4_vs_control_degs.csv')
de$hek_d10  <- make_de('HEK293', 'D10', 'hek293_d10_vs_control_degs.csv')
de$hek_e6   <- make_de('HEK293', 'E6',  'hek293_e6_vs_control_degs.csv')

sig_ids <- function(x) x$gene_id[!is.na(x$padj) & x$padj < 0.05]
sig_up_ids <- function(x) x$gene_id[!is.na(x$padj) & x$padj < 0.05 & !is.na(x$log2_fold_change) & x$log2_fold_change > 0]
sig_down_ids <- function(x) x$gene_id[!is.na(x$padj) & x$padj < 0.05 & !is.na(x$log2_fold_change) & x$log2_fold_change < 0]

regulation_for <- function(ids, tables) {
  out <- character(length(ids))
  names(out) <- ids
  for (gid in ids) {
    lfcs <- vapply(tables, function(tab) tab$log2_fold_change[match(gid, tab$gene_id)], numeric(1))
    lfcs <- lfcs[!is.na(lfcs)]
    out[gid] <- if (length(lfcs) > 0 && mean(lfcs) >= 0) 'upregulated' else 'downregulated'
  }
  out
}

# ---------- Shared DEG gene lists ----------
# Unresolved shared-set definition: top option was 'None'; use the top actionable contract-compatible option:
# significant in all relevant treatments, with consensus regulation by mean log2 fold-change sign.
t98g_shared_ids <- Reduce(intersect, list(sig_ids(de$t98g_c10), sig_ids(de$t98g_f7), sig_ids(de$t98g_g2)))
t98g_reg <- regulation_for(t98g_shared_ids, list(de$t98g_c10, de$t98g_f7, de$t98g_g2))
t98g_shared <- data.frame(
  gene_symbol = anno$gene_symbol[match(t98g_shared_ids, anno$gene_id)],
  regulation = unname(t98g_reg),
  stringsAsFactors = FALSE
)
t98g_shared <- t98g_shared[!is.na(t98g_shared$gene_symbol) & t98g_shared$gene_symbol != '', , drop = FALSE]
t98g_shared <- t98g_shared[!duplicated(t98g_shared$gene_symbol), , drop = FALSE]
t98g_shared <- t98g_shared[order(t98g_shared$regulation, t98g_shared$gene_symbol), , drop = FALSE]
write.csv(t98g_shared, '/outputs/t98g_slow_shared_degs.csv', row.names = FALSE, quote = TRUE, na = '')

hek_shared_ids <- intersect(sig_ids(de$hek_d10), sig_ids(de$hek_e6))
hek_reg <- regulation_for(hek_shared_ids, list(de$hek_d10, de$hek_e6))
hek_shared <- data.frame(
  gene_symbol = anno$gene_symbol[match(hek_shared_ids, anno$gene_id)],
  regulation = unname(hek_reg),
  stringsAsFactors = FALSE
)
hek_shared <- hek_shared[!is.na(hek_shared$gene_symbol) & hek_shared$gene_symbol != '', , drop = FALSE]
hek_shared <- hek_shared[!duplicated(hek_shared$gene_symbol), , drop = FALSE]
hek_shared <- hek_shared[order(hek_shared$regulation, hek_shared$gene_symbol), , drop = FALSE]
write.csv(hek_shared, '/outputs/hek293_shared_degs.csv', row.names = FALSE, quote = TRUE, na = '')

# ---------- T98G D4-specific upregulated genes ----------
# D4-specific upregulated: significant and positive in D4, not significant in C10/F7/G2.
other_t98g_sig <- union(union(sig_ids(de$t98g_c10), sig_ids(de$t98g_f7)), sig_ids(de$t98g_g2))
d4_specific_up <- setdiff(sig_up_ids(de$t98g_d4), other_t98g_sig)
d4_specific_up <- sort(d4_specific_up)
write.csv(data.frame(gene_id = d4_specific_up, stringsAsFactors = FALSE),
          '/outputs/t98g_d4_specific_upregulated_genes.csv', row.names = FALSE, quote = TRUE, na = '')

# ---------- Methods and provenance ----------
methods <- c(
  '# METHODS',
  '',
  '## Inputs',
  '- Read `/data/tagseq_featurecounts_matrix.tsv` and `/data/samples.tsv`.',
  paste0('- Count matrix dimensions after reading contracted sample columns: ', nrow(counts), ' genes by ', ncol(counts), ' libraries.'),
  '- Biological replicates in the sample sheet were used for all comparisons.',
  '',
  '## Differential expression',
  '- For each cell line and treatment, the analysis was subset to that cell line and to Ctrl plus the treatment of interest.',
  '- DESeq2 was run with design `~ condition`; contrasts were treatment versus Ctrl.',
  '- Genes with total raw count >= 10 within each two-condition subset were retained before creating the DESeq2 object.',
  '- DESeq2 default median-of-ratios size-factor normalization, dispersion estimation, Wald testing, independent filtering, and Benjamini-Hochberg adjusted p-values were used.',
  '- Output DE columns are `gene_id`, `gene_symbol`, `base_mean`, `log2_fold_change`, `lfc_se`, `stat`, `p_value`, and `padj`.',
  '- Ensembl gene IDs were reported as in the input matrix. Gene symbols were taken from an annotation column if present; otherwise the unversioned Ensembl ID was used as a deterministic fallback.',
  '',
  '## Gene sets',
  '- `t98g_slow_shared_degs.csv`: genes significant at padj < 0.05 in all three slow T98G treatment comparisons C10, F7, and G2 versus Ctrl. Regulation is the sign of the mean log2 fold-change across those comparisons.',
  '- `hek293_shared_degs.csv`: genes significant at padj < 0.05 in both HEK293 treatment comparisons D10 and E6 versus Ctrl. Regulation is the sign of the mean log2 fold-change across those comparisons.',
  '- `t98g_d4_specific_upregulated_genes.csv`: genes significant at padj < 0.05 with positive log2 fold-change in T98G D4 versus Ctrl and not significant at padj < 0.05 in T98G C10, F7, or G2 versus Ctrl.',
  '',
  '## Settled decisions applied',
  '- Sample labels and condition assignments were kept as given. Nearest-neighbour diagnostics found every library nearest to its biological replicate; swap search outcome was `as_given`, with no flagged samples and negative swap gain.',
  '- All libraries were retained. Duplicate diagnostics found no identical or near-duplicate pair (maximum log2-CPM correlation 0.994158, below the duplicate threshold).',
  '- All libraries were retained for quality. Depth/detected-gene diagnostics flagged no samples; noise diagnostics flagged no samples.',
  '- Cell-line labels were kept as given. Centroid and PC1 diagnostics reported no mismatched samples.',
  '- Gene IDs were treated as one uniform versioned namespace. Strip-version diagnostic found all IDs versioned and no collisions; versions were not stripped from DE `gene_id` outputs.',
  '',
  '## Unresolved decisions and chosen options',
  '- Design formula question was unresolved and its top option was non-actionable (`None of these options`). Following the task method guidance, cell lines were analyzed separately with design `~ condition`.',
  '- Filtering threshold question was unresolved; the top actionable option, total counts >= 10, was used.',
  '- Multiple-testing correction question was unresolved; DESeq2 default Benjamini-Hochberg adjusted p-values were used.',
  '- Normalization question was unresolved and top options were non-actionable; DESeq2 default median-of-ratios normalization was used.',
  '- Gene-identifier and annotation questions were unresolved; gene IDs were kept as supplied for row keys, and available in-matrix annotation was preferred for symbols with stable Ensembl IDs as fallback.',
  '- Shared and D4-specific set definition was unresolved and top options were non-actionable; the top actionable significance-overlap definition was used, adapted to the contracted T98G slow shared set (C10/F7/G2) and D4-specific output.',
  '- Control-versus-treated inversion question was unresolved; the top option (`Control and treated groups are labelled the right way round`) was applied. Direction-signature diagnostic was consistent and no marker evidence supported inversion.',
  '',
  '## Rejected alternatives',
  '- Did not exchange labels, drop duplicate libraries, drop degraded libraries, or reassign cell lines because settled diagnostics rejected those alternatives.',
  '- Did not use Bonferroni/no correction, external normalization, full-dataset cell-line adjustment, or fold-change-only gene set definitions.'
)
writeLines(methods, '/outputs/METHODS.md')

# Basic contract sanity checks.
required_files <- c(
  't98g_c10_vs_control_degs.csv','t98g_f7_vs_control_degs.csv','t98g_g2_vs_control_degs.csv',
  't98g_slow_shared_degs.csv','hek293_d10_vs_control_degs.csv','hek293_e6_vs_control_degs.csv',
  'hek293_shared_degs.csv','t98g_d4_vs_control_degs.csv','t98g_d4_specific_upregulated_genes.csv','METHODS.md'
)
stopifnot(all(file.exists(file.path('/outputs', required_files))))
