suppressPackageStartupMessages({
  library(DESeq2)
})

outdir <- "/outputs"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

samples_path <- "/data/samples.tsv"
counts_path <- "/data/tagseq_featurecounts_matrix.tsv"

samples <- read.delim(samples_path, check.names = FALSE, stringsAsFactors = FALSE)
if (!"sample_id" %in% names(samples)) {
  if ("sample" %in% names(samples)) names(samples)[names(samples) == "sample"] <- "sample_id"
  if ("Sample" %in% names(samples)) names(samples)[names(samples) == "Sample"] <- "sample_id"
}
stopifnot(all(c("sample_id", "cell_line", "condition") %in% names(samples)))
samples$sample_id <- as.character(samples$sample_id)
samples$cell_line <- as.character(samples$cell_line)
samples$condition <- as.character(samples$condition)

raw <- read.delim(counts_path, check.names = FALSE, stringsAsFactors = FALSE)
sample_ids <- samples$sample_id
if (!all(sample_ids %in% colnames(raw))) {
  raw2 <- read.delim(counts_path, check.names = FALSE, stringsAsFactors = FALSE, row.names = 1)
  if (all(sample_ids %in% colnames(raw2))) raw <- data.frame(gene_id = rownames(raw2), raw2, check.names = FALSE)
}
missing_samples <- setdiff(sample_ids, colnames(raw))
if (length(missing_samples) > 0) stop("Missing count columns: ", paste(missing_samples, collapse = ", "))

non_sample_cols <- setdiff(colnames(raw), sample_ids)
id_candidates <- intersect(c("gene_id", "Geneid", "geneid", "GeneID", "ensembl_gene_id", "ENSEMBL"), non_sample_cols)
if (length(id_candidates) > 0) {
  id_col <- id_candidates[1]
} else if (length(non_sample_cols) > 0) {
  id_col <- non_sample_cols[1]
} else {
  stop("No gene identifier column found in count matrix")
}

gene_id_versioned <- as.character(raw[[id_col]])
gene_id <- sub("\\.[0-9]+$", "", gene_id_versioned)
if (anyDuplicated(gene_id) > 0) stop("Unversioned gene IDs are not unique after stripping versions")

symbol_candidates <- intersect(c("gene_symbol", "gene_name", "symbol", "Symbol", "external_gene_name", "hgnc_symbol"), setdiff(non_sample_cols, id_col))
gene_symbol <- rep(NA_character_, length(gene_id))
if (length(symbol_candidates) > 0) {
  gene_symbol <- as.character(raw[[symbol_candidates[1]]])
  gene_symbol[gene_symbol == ""] <- NA_character_
}

# Fill gene symbols from a local Bioconductor annotation package when it is available.
if (any(is.na(gene_symbol))) {
  if (requireNamespace("AnnotationDbi", quietly = TRUE) && requireNamespace("org.Hs.eg.db", quietly = TRUE)) {
    suppressPackageStartupMessages({
      library(AnnotationDbi)
      library(org.Hs.eg.db)
    })
    mapped <- AnnotationDbi::mapIds(org.Hs.eg.db, keys = unique(gene_id), keytype = "ENSEMBL", column = "SYMBOL", multiVals = "first")
    fill <- unname(mapped[gene_id])
    gene_symbol[is.na(gene_symbol) & !is.na(fill)] <- fill[is.na(gene_symbol) & !is.na(fill)]
  }
}
# Offline fallback: keep the stable Ensembl ID in gene_symbol if no symbol is locally available.
gene_symbol[is.na(gene_symbol) | gene_symbol == ""] <- gene_id[is.na(gene_symbol) | gene_symbol == ""]

count_mat <- as.matrix(raw[, sample_ids, drop = FALSE])
storage.mode(count_mat) <- "numeric"
if (any(is.na(count_mat))) stop("NA values found in count matrix")
count_mat <- round(count_mat)
rownames(count_mat) <- gene_id_versioned

symbol_by_versioned <- setNames(gene_symbol, gene_id_versioned)
stable_by_versioned <- setNames(gene_id, gene_id_versioned)

run_contrast <- function(cell_line, treatment, outfile) {
  ss <- samples[samples$cell_line == cell_line & samples$condition %in% c("Ctrl", treatment), , drop = FALSE]
  ss <- ss[match(intersect(sample_ids, ss$sample_id), ss$sample_id), , drop = FALSE]
  rownames(ss) <- ss$sample_id
  ss$condition <- factor(ss$condition, levels = c("Ctrl", treatment))
  cts <- count_mat[, rownames(ss), drop = FALSE]

  # Unresolved filtering question: apply the top actionable threshold, standard row-sum >= 10 counts.
  keep <- rowSums(cts) >= 10
  cts <- cts[keep, , drop = FALSE]

  dds <- DESeqDataSetFromMatrix(countData = cts, colData = ss, design = ~ condition)
  dds <- DESeq(dds, quiet = TRUE)
  res <- results(dds, contrast = c("condition", treatment, "Ctrl"), alpha = 0.05)
  rdf <- as.data.frame(res)
  rdf$gene_id <- stable_by_versioned[rownames(rdf)]
  rdf$gene_symbol <- symbol_by_versioned[rownames(rdf)]

  out <- data.frame(
    gene_id = rdf$gene_id,
    gene_symbol = rdf$gene_symbol,
    base_mean = rdf$baseMean,
    log2_fold_change = rdf$log2FoldChange,
    lfc_se = rdf$lfcSE,
    stat = rdf$stat,
    p_value = rdf$pvalue,
    padj = rdf$padj,
    stringsAsFactors = FALSE
  )
  out <- out[!is.na(out$padj) & out$padj < 0.05, , drop = FALSE]
  out <- out[order(out$padj, out$gene_id), , drop = FALSE]
  rownames(out) <- NULL
  write.csv(out, file.path(outdir, outfile), row.names = FALSE, quote = TRUE, na = "")

  full <- data.frame(
    gene_id = rdf$gene_id,
    gene_symbol = rdf$gene_symbol,
    base_mean = rdf$baseMean,
    log2_fold_change = rdf$log2FoldChange,
    lfc_se = rdf$lfcSE,
    stat = rdf$stat,
    p_value = rdf$pvalue,
    padj = rdf$padj,
    stringsAsFactors = FALSE
  )
  full <- full[order(full$gene_id), , drop = FALSE]
  rownames(full) <- full$gene_id
  list(full = full, sig = out, n_tested = nrow(full), n_sig = nrow(out), samples = rownames(ss))
}

res <- list()
res[["T98G_C10"]] <- run_contrast("T98G", "C10", "t98g_c10_vs_control_degs.csv")
res[["T98G_F7"]]  <- run_contrast("T98G", "F7",  "t98g_f7_vs_control_degs.csv")
res[["T98G_G2"]]  <- run_contrast("T98G", "G2",  "t98g_g2_vs_control_degs.csv")
res[["T98G_D4"]]  <- run_contrast("T98G", "D4",  "t98g_d4_vs_control_degs.csv")
res[["HEK293_D10"]] <- run_contrast("HEK293", "D10", "hek293_d10_vs_control_degs.csv")
res[["HEK293_E6"]]  <- run_contrast("HEK293", "E6",  "hek293_e6_vs_control_degs.csv")

is_sig <- function(tbl) !is.na(tbl$padj) & tbl$padj < 0.05

shared_list <- function(keys, outfile) {
  gene_sets <- lapply(keys, function(k) rownames(res[[k]]$full)[is_sig(res[[k]]$full)])
  common <- Reduce(intersect, gene_sets)
  rows <- list()
  for (gid in common) {
    lfcs <- vapply(keys, function(k) res[[k]]$full[gid, "log2_fold_change"], numeric(1))
    if (all(lfcs > 0, na.rm = FALSE)) {
      rows[[length(rows) + 1]] <- data.frame(gene_symbol = res[[keys[1]]]$full[gid, "gene_symbol"], regulation = "upregulated", stringsAsFactors = FALSE)
    } else if (all(lfcs < 0, na.rm = FALSE)) {
      rows[[length(rows) + 1]] <- data.frame(gene_symbol = res[[keys[1]]]$full[gid, "gene_symbol"], regulation = "downregulated", stringsAsFactors = FALSE)
    }
  }
  if (length(rows) == 0) {
    out <- data.frame(gene_symbol = character(), regulation = character(), stringsAsFactors = FALSE)
  } else {
    out <- do.call(rbind, rows)
    out <- out[!is.na(out$gene_symbol) & out$gene_symbol != "", , drop = FALSE]
    out <- out[!duplicated(out$gene_symbol), , drop = FALSE]
    out <- out[order(out$gene_symbol), , drop = FALSE]
  }
  rownames(out) <- NULL
  write.csv(out, file.path(outdir, outfile), row.names = FALSE, quote = TRUE)
  out
}

t98g_shared <- shared_list(c("T98G_C10", "T98G_F7", "T98G_G2"), "t98g_slow_shared_degs.csv")
hek_shared <- shared_list(c("HEK293_D10", "HEK293_E6"), "hek293_shared_degs.csv")

# D4-specific upregulated: significant and positive in D4; not significant in any other T98G treatment.
d4_tbl <- res[["T98G_D4"]]$full
other_t98g <- c("T98G_C10", "T98G_F7", "T98G_G2")
d4_up <- rownames(d4_tbl)[is_sig(d4_tbl) & !is.na(d4_tbl$log2_fold_change) & d4_tbl$log2_fold_change > 0]
other_sig <- unique(unlist(lapply(other_t98g, function(k) rownames(res[[k]]$full)[is_sig(res[[k]]$full)]), use.names = FALSE))
d4_specific <- sort(setdiff(d4_up, other_sig))
d4_specific_out <- data.frame(gene_id = d4_specific, stringsAsFactors = FALSE)
write.csv(d4_specific_out, file.path(outdir, "t98g_d4_specific_upregulated_genes.csv"), row.names = FALSE, quote = TRUE)

# Confirm contracted files have unversioned gene_id values where applicable.
for (fn in c("t98g_c10_vs_control_degs.csv", "t98g_f7_vs_control_degs.csv", "t98g_g2_vs_control_degs.csv", "t98g_d4_vs_control_degs.csv", "hek293_d10_vs_control_degs.csv", "hek293_e6_vs_control_degs.csv", "t98g_d4_specific_upregulated_genes.csv")) {
  x <- read.csv(file.path(outdir, fn), stringsAsFactors = FALSE, check.names = FALSE)
  if ("gene_id" %in% names(x) && any(grepl("\\.[0-9]+$", x$gene_id))) stop("Versioned gene_id found in ", fn)
}

methods <- c(
  "# METHODS",
  "",
  "## Inputs",
  paste0("Counts: ", counts_path),
  paste0("Samples: ", samples_path),
  paste0("Samples analysed: ", paste(sample_ids, collapse = ", ")),
  "",
  "## Differential expression analysis",
  "For each requested cell line/treatment comparison, samples were subset to that cell line and to Ctrl plus the treatment. Biological replicate columns in the sample sheet were used as independent replicates. DESeq2 was run with design `~ condition`, Ctrl as the reference level, and the contrast treatment versus Ctrl. Default DESeq2 median-of-ratios size-factor normalization, dispersion estimation, Wald testing, independent filtering, and Benjamini-Hochberg adjusted p-values were used.",
  "The unresolved filtering question's top actionable threshold was applied as a prefilter: genes with row-sum counts < 10 in the samples for a comparison were removed before DESeq2. Contracted DE tables contain genes with DESeq2 padj < 0.05. Log2 fold change is positive for higher expression in treatment than Ctrl.",
  "Gene IDs were stripped from versioned Ensembl IDs to stable unversioned Ensembl IDs for every output. This specifically fixes the prior critic finding that version suffixes were present in DE tables and in the D4-specific gene list. The settled namespace diagnostic found all IDs were uniformly versioned and reported zero collisions after stripping, so stripping only at reporting is safe.",
  "Gene symbols were taken from an input symbol/name column when present; otherwise a local org.Hs.eg.db mapping was used when installed. Remaining unmapped symbols were filled with the unversioned Ensembl ID to keep required columns complete without network access.",
  "",
  "## Gene-set definitions",
  "T98G slow shared DEGs were defined as genes significant at padj < 0.05 in all three slow T98G treatments C10, F7, and G2 versus Ctrl, retaining only genes with concordant log2 fold-change direction across all three. HEK293 shared DEGs were defined analogously for D10 and E6. The regulation label is `upregulated` when all relevant log2 fold changes are positive and `downregulated` when all are negative.",
  "T98G D4-specific upregulated genes were defined as genes with padj < 0.05 and positive log2 fold change in D4 versus Ctrl and not significant at padj < 0.05 in C10, F7, or G2 versus Ctrl. The contracted list is reported as unversioned Ensembl gene_id values.",
  "",
  "## Decisions and diagnostics",
  "Settled sample-label decision: accepted labels as given. Nearest-neighbour Spearman diagnostics had every library nearest to its same-condition replicate, no misplaced samples, no resolving swaps, and swap-search preferred the as-given labels with negative swap gain. The alternative two-library swap was rejected.",
  "Settled duplicate-library decision: every library was distinct. No pairs were identical on >=99% detected genes; maximum identical fraction was 0.087827. No log2-CPM correlations reached 0.9999; maximum was 0.994158. The duplicate-library exclusion option was rejected.",
  "Settled library-quality decision: every library was usable. Depth/detected-gene diagnostics flagged zero samples; minimum depth ratio was 0.5235 and minimum detected-gene ratio was 0.868586. Noise diagnostics flagged zero samples; maximum noise ratio was 1.297584. The degraded-library exclusion option was rejected.",
  "Settled cell-line decision: the cell_line column was used as given. Leave-one-out cell-line centroid diagnostics found zero mismatches with minimum margin 1.577782, and unsupervised PC1 split found zero mismatches and strong separation. Cell-line reassignment was rejected.",
  "Settled gene-ID namespace decision: IDs were in one namespace. Stripping version suffixes changed all 60708 IDs, produced 60708 stable IDs, and produced zero collisions. Mixed-version handling was rejected.",
  "Unresolved design-formula decision: the options did not settle a single global formula. The task explicitly requested per-cell-line treatment-versus-Ctrl analyses, so each cell line/comparison was analysed with DESeq2 `~ condition` using biological replicates.",
  "Unresolved multiple-testing decision: DESeq2's default Benjamini-Hochberg padj was used, matching the top actionable option and DESeq2 defaults.",
  "Unresolved normalization decision: DESeq2 default median-of-ratios normalization was retained, following task guidance to keep default DESeq2 settings.",
  "Unresolved control-direction decision: labels were retained as given. Direction-signature diagnostics were consistent (mean cross-correlation 0.316668; all group scores positive) and marker diagnostics were unclear, so no control/treatment inversion was applied.",
  "Unresolved sample-label, coverage, and result-plausibility questions were not used to exclude or relabel samples because later settled diagnostics found labels/cell lines correct, no duplicates, and all libraries usable.",
  "",
  "## Output summary",
  paste0("t98g_c10_vs_control_degs.csv: ", res[["T98G_C10"]]$n_sig, " DEGs from ", res[["T98G_C10"]]$n_tested, " tested genes."),
  paste0("t98g_f7_vs_control_degs.csv: ", res[["T98G_F7"]]$n_sig, " DEGs from ", res[["T98G_F7"]]$n_tested, " tested genes."),
  paste0("t98g_g2_vs_control_degs.csv: ", res[["T98G_G2"]]$n_sig, " DEGs from ", res[["T98G_G2"]]$n_tested, " tested genes."),
  paste0("t98g_d4_vs_control_degs.csv: ", res[["T98G_D4"]]$n_sig, " DEGs from ", res[["T98G_D4"]]$n_tested, " tested genes."),
  paste0("hek293_d10_vs_control_degs.csv: ", res[["HEK293_D10"]]$n_sig, " DEGs from ", res[["HEK293_D10"]]$n_tested, " tested genes."),
  paste0("hek293_e6_vs_control_degs.csv: ", res[["HEK293_E6"]]$n_sig, " DEGs from ", res[["HEK293_E6"]]$n_tested, " tested genes."),
  paste0("t98g_slow_shared_degs.csv: ", nrow(t98g_shared), " shared concordant genes."),
  paste0("hek293_shared_degs.csv: ", nrow(hek_shared), " shared concordant genes."),
  paste0("t98g_d4_specific_upregulated_genes.csv: ", nrow(d4_specific_out), " genes.")
)
writeLines(methods, file.path(outdir, "METHODS.md"))
