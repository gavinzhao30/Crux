# Controlled DESeq2 smoke path. An agent supplies the full study pipeline.
suppressPackageStartupMessages(library(DESeq2))

meta <- read.delim("/data/samples.tsv", check.names = FALSE, colClasses = "character")
counts <- read.delim(
    "/data/tagseq_featurecounts_matrix.tsv", check.names = FALSE, comment.char = "#"
)
stopifnot(all(c("sample_id", "cell_line", "condition") %in% names(meta)))
stopifnot(nrow(meta) == 16L, !anyDuplicated(meta$sample_id))
gene_col <- if ("Geneid" %in% names(counts)) "Geneid" else names(counts)[[1]]
meta <- meta[meta$cell_line == cell_line & meta$condition %in% c(reference, treatment), ]
stopifnot(all(table(factor(meta$condition, levels = c(reference, treatment))) >= 2L))
count_columns <- if ("count_column" %in% names(meta)) meta$count_column else meta$sample_id
stopifnot(!anyDuplicated(count_columns), all(count_columns %in% names(counts)))
matrix <- as.matrix(counts[, count_columns, drop = FALSE])
stopifnot(is.numeric(matrix), all(is.finite(matrix)), all(matrix >= 0), all(matrix == floor(matrix)))
rownames(matrix) <- sub("\\.[0-9]+$", "", counts[[gene_col]])
stopifnot(!anyDuplicated(rownames(matrix)))
colnames(matrix) <- meta$sample_id
rownames(meta) <- meta$sample_id
meta$condition <- factor(meta$condition, levels = c(reference, treatment))
dds <- DESeqDataSetFromMatrix(matrix, meta, design = ~ condition)
dds <- DESeq(dds, quiet = TRUE)
res <- results(dds, contrast = c("condition", treatment, reference), alpha = 0.05)
output <- data.frame(
    gene_id = rownames(res),
    gene_symbol = if ("gene_symbol" %in% names(counts)) counts$gene_symbol else "",
    base_mean = res$baseMean,
    log2_fold_change = res$log2FoldChange,
    lfc_se = res$lfcSE,
    stat = res$stat,
    p_value = res$pvalue,
    padj = res$padj
)
write.csv(output, "/outputs/smoke_de.csv.tmp", row.names = FALSE, na = "")
stopifnot(file.rename("/outputs/smoke_de.csv.tmp", "/outputs/smoke_de.csv"))
writeLines(
    c("# Task 9 smoke analysis", paste(cell_line, treatment, "versus", reference),
      "DESeq2 design: ~ condition; contrast alpha: 0.05."),
    "/outputs/METHODS.md"
)
cat('RESULT: {"output":"smoke_de.csv"}\n')
