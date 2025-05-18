utils::globalVariables(c("metapath2vec_pg"))
#'@title pg_embedding
#'
#'@description Learn the low-dimensional representations for peaks and genes with a meta-path based method.
#'@name pg_embedding
#'@import Matrix
#'@importFrom reticulate source_python
#'@importFrom tictoc tic toc
#'@importFrom utils read.csv
#'@param gg_net The gene-gene network.
#'@param pp_net The peak-peak network.
#'@param net_lasso The peak-gene network constructed by Lasso.
#'@param net_RF The peak-gene network constructed by Random Forest.
#'@param net_XGB The peak-gene network constructed by XGBoost.
#'@param dirpath The folder path to read or write file.
#'@param relearn_pg_embedding Logical. Whether to relearn the low-dimensional representations for peaks and genes from scratch. If FALSE, the function will attempt to read from \cr'node_embeddings.mtx', 'node_used_peak.csv', 'node_used_gene.csv' \cr under `dirpath/embedding` in `single` mode or \cr `dirpath/state_name/embedding` in `compare` mode.
#'@param d The dimension of latent space.
#'@param seed An integer specifying the random seed to ensure reproducible results.
#'@return a list containing the following \describe{
#'\item{\code{E}}{low-dimensional representations of peaks and genes}
#'\item{\code{peak_node}}{Peak ids that are associated with other peaks or genes.}
#'\item{\code{gene_node}}{Gene ids that are associated with other peaks or genes.}
#'}
#'@examples
#'\dontrun{
#' library(scPOEM)
#' library(Matrix)
#' library(data.table)
#' dirpath <- "./example_data"
#' # Download single mode example data
#' data(input_single)
#' gg_net <- GGN(input_single$Y, file.path(dirpath, "single"), 1, 5, T)
#' pp_net <- PPN(input_single$X, input_single$peak_data,
#'               input_single$cell_data, input_single$genome,
#'               file.path(dirpath, "single"))
#' net_Lasso <- PGN_Lasso(input_single$X, input_single$Y,
#'                        input_single$gene_data, input_single$neibor_peak,
#'                        file.path(dirpath, "single"))
#' net_RF <- PGN_RF(input_single$X, input_single$Y,
#'                  input_single$gene_data, input_single$neibor_peak,
#'                  file.path(dirpath, "single"))
#' net_XGB <- PGN_XGBoost(input_single$X, input_single$Y,
#'                        input_single$gene_data, input_single$neibor_peak,
#'                        file.path(dirpath, "single"))
#' E_result <- pg_embedding(gg_net, pp_net, net_lasso, net_RF, net_XGB,
#'                          file.path(dirpath, "single"))
#'}
#'
#' @export

pg_embedding <- function(gg_net, pp_net, net_lasso, net_RF, net_XGB, dirpath, relearn_pg_embedding=T, d=100, seed=0) {
  if (!relearn_pg_embedding) {
    cat("Load embedding results\n")
    E <- readMM(file.path(dirpath, "embedding/node_embeddings.mtx"))
    peak_node <- read.csv(file.path(dirpath, "embedding/node_used_peak.csv"), header = FALSE)[[1]]
    gene_node <- read.csv(file.path(dirpath, "embedding/node_used_gene.csv"), header = FALSE)[[1]]
    cat("\n")
    return(list(E=E, peak_node=peak_node, gene_node=gene_node))
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Please install it with install.packages('reticulate').")
  }
  cat("Learn representations of peaks and genes...\n")
  tic()
  py_script <- system.file("python/pg_embedding.py", package = "scPOEM")

  source_python(py_script)

  E_result <- metapath2vec_pg(gg_net, pp_net, net_lasso, net_RF, net_XGB, dirpath, as.integer(d), as.integer(seed))
  toc()
  E <- E_result[[1]]
  peak_node <- as.vector(E_result[[2]])
  gene_node <- as.vector(E_result[[3]])
  cat("\n")
  return(list(E=E, peak_node=peak_node, gene_node=gene_node))
}
