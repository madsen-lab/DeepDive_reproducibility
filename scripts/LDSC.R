library(genio)
library(liftOver)
library(mapgen)
library(GenomicRanges)
library(ggplot2)
library(data.table)

window_size = 1000
min_peaks = 200
annotation_folder <- "results/GWAS_enrich/annotation/"

# Setup to lift BED file from hg38 to hg19
path <- system.file(package="liftOver", "extdata", "hg38ToHg19.over.chain")
ch <- import.chain(path)

# Setup for processing peaks
anno.list <- list()

# Loop across peaks for each cell type
for (g in c("Both","BMI", 'T2D', 'NS')) {
        # Define filename and check it exists and has lines
        filename <- paste("results/GWAS_enrich/peaks/", g, ".bed", sep="")
        if (file.exists(filename) & file.size(filename) > 0) {
          # Load peaks and make GRanges object
          bed <- read.delim(filename, header = FALSE)
          bed_gr <- GRanges(bed$V1, ranges = IRanges(start = bed$V2, end = bed$V3))

          # Lift from hg38 to hg19
          lo <- liftOver(bed_gr, ch)
          lo_df <- lapply(lo, "as.data.frame")
          counter <- 1
          for (i in 1:length(lo_df)) {
            df_tmp <- lo_df[[i]]
            if (nrow(df_tmp) == 1) {
              if (counter == 1) {
                df_res <- df_tmp
                counter <- 2
              } else {
                df_res <- rbind(df_res, df_tmp)
              }
            }
          }

          # Create windowed GR
          df_res$center <- as.integer((df_res$start + df_res$end) / 2)
          df_res$start <- df_res$center - window_size / 2
          df_res$end <- df_res$center + window_size / 2
          bed_gr <- GRanges(df_res$seqnames, ranges = IRanges(start = df_res$start, end = df_res$end))

          # Append to annotation list if more than min_peaks
          if (length(bed_gr) >= min_peaks) {
            anno.list[[length(anno.list) + 1]] <- bed_gr
            names(anno.list)[length(anno.list)] <- paste("group_", g, sep="")
          }
        }
      }


# Create folder structure and save the annotation for later use
system(paste("mkdir -p", annotation_folder))
saveRDS(anno.list, paste(annotation_folder, "annotation_", window_size, ".rds", sep=""))

# Loop across chromosomes and overlap SNPs with peaks
for (chr in 1:22) {
  bim <- read_bim(paste("results/GWAS_enrich/1000G_EUR_Phase3_plink/1000G.EUR.QC.", chr, ".bim", sep=""))
  bim <- as.data.frame(bim)
  bim <- bim[,c(1,4,2,3)]
  colnames(bim) <- c("CHR", "BP","SNP","CM")
  bim_gr <- GRanges(paste("chr", bim$CHR, sep=""), ranges = IRanges(start = bim$BP, width = 1), rsid = bim$id)
  for (annoidx in 1:length(anno.list)) {
    anno_gr <- anno.list[[annoidx]]
    bim[,ncol(bim) + 1] <- 0
    colnames(bim)[ncol(bim)] <- names(anno.list)[annoidx]
    hits <- unique(as.data.frame(findOverlaps(bim_gr, anno_gr))[,1])
    if (length(hits) > 0) { bim[hits, ncol(bim)] <- 1 } 
  }
  write.table(bim, paste(annotation_folder, "/1000G.EUR.QC.", chr, ".annot", sep=""), row.names = FALSE, col.names = TRUE, quote = FALSE, sep="\t")
  system(paste("gzip ", annotation_folder, "/1000G.EUR.QC.", chr, ".annot", sep=""))
}
