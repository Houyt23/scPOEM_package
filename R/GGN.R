utils::globalVariables(c("make_GGN"))
#'@title GGN
#'
#'@description Make gene-gene net via principle component regression.
#'@name GGN
#'@importFrom Matrix readMM
#'@importFrom reticulate source_python
#'@importFrom tictoc tic toc
#'@importFrom utils globalVariables
#'@param Y The scRNA-seq data, sparse matrix.
#'@param dirpath The folder path to read or write file.
#'@param count_device The number of cpus used to train the Lasso model.
#'@param nComp The number of PCs used for regression
#'@param rebuild_GGN Logical. Whether to rebuild the gene-gene network (GGN) from scratch. If FALSE, the function will attempt to read from 'GGN.mtx' under `dirpath/test` in `single` mode or `dirpath/state_name/test` in `compare` mode.
#'@return The GGN network.
#'@examples
#'\dontrun{
#' library(scPOEM)
#' library(Matrix)
#' dirpath <- "./example_data"
#' # Download single mode example data
#' data(input_single)
#' # Construct GGN net.
#' gg_net <- GGN(input_single$Y, file.path(dirpath, "single"))
#'}
#'
#' @export

GGN <- function(Y, dirpath, count_device, nComp=5, rebuild_GGN=T) {
  if (!rebuild_GGN){
    cat("Load gene-gene network\n")
    gg_net <- readMM(file.path(dirpath, "test/GGN.mtx"))
    cat("\n")
    return(gg_net)
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Please install it with install.packages('reticulate').")
  }

  cat("Construct gene-gene network from scratch...\n")
  tic()
  py_script <- system.file("python/GGN.py", package = "scPOEM")

  source_python(py_script)

  gg_net <- make_GGN(Y, dirpath, as.integer(nComp), as.integer(count_device))
  toc()
  cat("GGN is saved in:", file.path(dirpath, "test/GGN.mtx"), "\n")
  cat("\n")
  return(gg_net)
}
