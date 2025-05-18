utils::globalVariables(c("make_PGN_RF"))
#'@title PGN_PF
#'
#'@description Make peak-gene net via random forest.
#'@name PGN_RF
#'@import Matrix
#'@importFrom reticulate source_python
#'@importFrom tictoc tic toc
#'@param X The scATAC-seq data, sparse matrix.
#'@param Y The scRNA-seq data, sparse matrix.
#'@param gene_data The information for genes, must have a col names 'gene_name'.
#'@param neibor_peak The peak IDs within a certain range of each gene, must have cols c("gene_name", "start_use", "end_use"). The id numbers in "start_use" and "end_use" are start from 0.
#'@param dirpath The folder path to read or write file.
#'@param count_device The number of cpus used to train the Lasso model.
#'@param rebuild_PGN_RF Logical. Whether to rebuild the peak-gene network via random forest from scratch. If FALSE, the function will attempt to read from 'PGN_RF.mtx' under `dirpath/test` in `single` mode or `dirpath/state_name/test` in `compare` mode.
#'@param seed An integer specifying the random seed to ensure reproducible results.
#'@return The PGN_RF network.
#'@examples
#'\dontrun{
#' library(scPOEM)
#' library(Matrix)
#' dirpath <- "./example_data"
#' # Download single mode example data
#' data(input_single)
#' # Construct PGN net via random forest (RF).
#' net_RF <- PGN_RF(input_single$X,
#'                  input_single$Y,
#'                  input_single$gene_data,
#'                  input_single$neibor_peak,
#'                  file.path(dirpath, "single"))
#'}
#'
#' @export

PGN_RF <- function(X, Y, gene_data, neibor_peak, dirpath, count_device=1, rebuild_PGN_RF=T, seed=0) {
  if (!rebuild_PGN_RF){
    cat("Load peak-gene network constructed by random forest\n")
    net_RF <- readMM(file.path(dirpath, "test/PGN_RF.mtx"))
    cat("\n")
    return(net_RF)
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Please install it with install.packages('reticulate').")
  }
  cat("Construct peak-gene network via random forest from scratch...\n")
  tic()
  py_script <-system.file("python/PGN_RF.py", package = "scPOEM")

  source_python(py_script)

  net_RF <- make_PGN_RF(X, Y, gene_data, neibor_peak, dirpath, as.integer(count_device), as.integer(seed))
  toc()
  cat("PGN_RF is saved in:", file.path(dirpath, "test/PGN_RF.mtx"), "\n")
  cat("\n")
  return(net_RF)
}
