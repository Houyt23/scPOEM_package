#'@title PPN
#'
#'@description Make peak-peak net.
#'@name PPN
#'@import monocle3
#'@import cicero
#'@importFrom stringr str_split_fixed
#'@import Matrix
#'@importFrom SingleCellExperiment reducedDim
#'@importFrom tictoc tic toc
#'@importFrom magrittr %>%
#'@importFrom stats setNames
#'@param X The scATAC-seq data, sparse matrix.
#'@param peak_data The information for peaks, must have a col names 'peak_name'.
#'@param cell_data The information for cells, must have a col names 'cell_name'.
#'@param genome The genome length for the species.
#'@param dirpath The folder path to read or write file.
#'@param rebuild_PPN Logical. Whether to rebuild the peak-peak network (PPN) from scratch. If FALSE, the function will attempt to read from 'PPN.mtx' under `dirpath/test` in `single` mode or `dirpath/state_name/test` in `compare` mode.
#'@param seed An integer specifying the random seed to ensure reproducible results.
#'@return The PPN network.
#'@examples
#'\dontrun{
#' library(scPOEM)
#' library(Matrix)
#' library(data.table)
#' dirpath <- "./example_data"
#' # Download single mode example data
#' data(input_single)
#' # Construct PPN net.
#' pp_net <- PPN(input_single$X,
#'               input_single$peak_data,
#'               input_single$cell_data,
#'               input_single$genome,
#'               file.path(dirpath, "single"))
#'}
#'
#' @export

PPN <-function(X, peak_data, cell_data, genome, dirpath, rebuild_PPN=T, seed=0){
  if (!rebuild_PPN){
    cat("Load peak-peak network\n")
    pp_net <- readMM(file.path(dirpath, "test/PPN.mtx"))
    cat("\n")
    return(pp_net)
  }
  cat("Construct peak-peak matrix from scratch...\n")
  use_cicero <- function(indata, cellinfo, peakinfo, genome, dirpath){
    # make CDS
    input_cds <-  suppressWarnings(new_cell_data_set(indata,
                                                     cell_metadata = cellinfo,
                                                     gene_metadata = peakinfo))

    input_cds <- detect_genes(input_cds)

    #Ensure there are no peaks included with zero reads
    input_cds <- input_cds[rowSums(exprs(input_cds)) != 0,]

    # use UMAP to reduce dimensionality
    set.seed(as.integer(seed))
    input_cds <- estimate_size_factors(input_cds)
    input_cds <- preprocess_cds(input_cds, method = "LSI")
    input_cds <- reduce_dimension(input_cds, reduction_method = 'UMAP',
                                  preprocess_method = "LSI")
    plot_cells(input_cds)

    #access the UMAP coordinates from the input CDS object
    umap_coords <- reducedDim(input_cds)

    #run make_cicero_cds
    cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = umap_coords)
    gc()

    #genome <- fread(genome_file, header = FALSE)#human.hg38.genome,mouse.mm10.genome
    conns <- run_cicero(cicero_cds, genome, sample_num = 100)
    head(conns)


    conns$Peak2 <- as.character(conns$Peak2)

    mask <- !(is.na(conns$coaccess) | (conns$coaccess==0))
    co <- conns$coaccess[mask]
    Peak1 <- conns$Peak1[mask]
    Peak2 <- conns$Peak2[mask]

    x <- setNames(seq(nrow(peakinfo)), peakinfo$peak_name)
    id1 <- x[Peak1]
    id2 <- x[Peak2]
    names(id1) <- NULL
    names(id2) <- NULL
    conn_mx <- sparseMatrix(i = id1, j = id2, x = co, dims = c(nrow(peakinfo), nrow(peakinfo)))

    if (!dir.exists(file.path(dirpath, "test"))) {
      dir.create(file.path(dirpath, "test"))
    }
    writeMM(conn_mx, file.path(dirpath, "test/PPN.mtx"))
    cat("PNN is saved in:", file.path(dirpath, "test/PPN.mtx"), "\n")
    return(conn_mx)
  }

  indata <- X#Matrix::readMM(file.path(dirpath, "X.mtx"))
  indata@x[indata@x > 0] <- 1
  indata <- t(indata)

  # format cell info
  #cell_data <- read.csv(file.path(dirpath, "cell_data.csv"))
  #colnames(cell_data) <- c('rank','cell_name')
  cell_name <- cell_data$cell_name
  cellinfo <- data.frame(
    cell_name = cell_name,
    row.names = cell_name
  )

  # format peak info
  #peak_data <- read.csv(file.path(dirpath, "peak_data.csv"))
  #colnames(peak_data) <- c('rank', 'peak_name')
  peak_name <- peak_data$peak_name
  peakinfo=str_split_fixed(peak_name,"-",2)%>%data.frame()
  peakinfo[,c(3,4)] <- str_split_fixed(peakinfo$X2,"-",2)
  peakinfo[,5] <- peak_name
  peakinfo <- peakinfo[,-2]
  names(peakinfo) <- c("chr", "bp1", "bp2", "peak_name")
  row.names(peakinfo) <- peakinfo$peak_name
  peakinfo$id <- seq(1, nrow(peakinfo))

  row.names(indata) <- row.names(peakinfo)
  colnames(indata) <- row.names(cellinfo)

  tic()
  pp_net <- use_cicero(indata, cellinfo, peakinfo, genome, dirpath)
  toc()
  cat("\n")
  return(pp_net)
}

#args <- commandArgs(trailingOnly = TRUE)
#dirpath <- NULL

#for (i in seq(1, length(args))) {
#  if (args[i] == "--dirpath") {
#    dirpath <- args[i+1]
#  }
#}

#if (is.null(dirpath)){
#  dirpath <- "data_example/single"
#}

#PPN(dirpath)


