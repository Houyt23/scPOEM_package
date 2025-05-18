#'@title scPOEM
#'
#'@description A embedding method that jointly projects chromatin accessibility peaks and expressed genes into a shared low-dimensional space.
#'@name scPOEM
#'@importFrom Matrix readMM
#'@importFrom utils read.csv
#'@param mode The mode indicating whether to analyze data from a single condition or to compare two conditions.
#'@param input_data A list of input data.
#'
#'If \code{mode = "single"}, \code{input_data} must be a list containing the following **seven objects**:
#' \itemize{
#'   \item \code{X}: Gene expression matrix.
#'   \item \code{Y}: Peak accessibility matrix.
#'   \item \code{peak_data}: A data.frame containing peak information.
#'   \item \code{gene_data}: A data.frame containing gene information (must contain column \code{"gene_name"}).
#'   \item \code{cell_data}: A data.frame containing cell metadata.
#'   \item \code{neibor_peak}: The peak IDs within a certain range of each gene, must have cols c("gene_name", "start_use", "end_use"). The id numbers in "start_use" and "end_use" are start from 0.
#'   \item \code{genome}: The genome length for the species.
#' }
#'
#'If \code{mode = "compare"}, \code{input_data} must be a **named list of two elements**, with names corresponding to two state names (e.g., "state1" and "state2"). Each element must itself be a list containing the same seven components as described above for \code{mode = "single"}.
#'@param dirpath The folder path to read or write file.
#'@param count_device The number of cpus used to train models.
#'@param nComp The number of PCs used for regression in constructing GGN.
#'@param d The dimension of latent space.
#'@param seed An integer specifying the random seed to ensure reproducible results.
#'@param rebuild_GGN Logical. Whether to rebuild the gene-gene network from scratch. If FALSE, the function will attempt to read from 'GGN.mtx' under `dirpath/test` in `single` mode or `dirpath/state_name/test` in `compare` mode.
#'@param rebuild_PPN Logical. Whether to rebuild the peak-peak network from scratch. If FALSE, the function will attempt to read from 'PPN.mtx' under `dirpath/test` in `single` mode or `dirpath/state_name/test` in `compare` mode.
#'@param rebuild_PGN_Lasso Logical. Whether to rebuild the peak-gene network via Lasso from scratch. If FALSE, the function will attempt to read from 'PGN_Lasso.mtx' under \cr `dirpath/test` in `single` mode or `dirpath/state_name/test` in `compare` mode.
#'@param rebuild_PGN_RF Logical. Whether to rebuild the peak-gene network via random forest from scratch. If FALSE, the function will attempt to read from 'PGN_RF.mtx' under `dirpath/test` in `single` mode or `dirpath/state_name/test` in `compare` mode.
#'@param rebuild_PGN_XGB Logical. Whether to rebuild the peak-gene network via XGBoost from scratch. If FALSE, the function will attempt to read from 'PGN_XGB.mtx' under \cr `dirpath/test` in `single` mode or `dirpath/state_name/test` in `compare` mode.
#'@param relearn_pg_embedding Logical. Whether to relearn the low-dimensional representations for peaks and genes from scratch. If FALSE, the function will attempt to read from \cr'node_embeddings.mtx', 'node_used_peak.csv', 'node_used_gene.csv' \cr under `dirpath/embedding` in `single` mode or \cr `dirpath/state_name/embedding` in `compare` mode.
#'@return The scPOEM result.
#'@examples
#'\dontrun{
#' library(scPOEM)
#' library(Matrix)
#' library(data.table)
#' dirpath <- "./example_data"
#' # An example for analysing a single dataset.
#' # Download and read data.
#' data(input_single)
#' single_result <- scPOEM(mode = "single",
#'                         input_data=input_single,
#'                         dirpath=file.path(dirpath, "single"))
#'
#' # An example for analysing and comparing datasets from two conditions.
#' # Download compare mode example data
#' data(input_compare)
#' compare_result <- scPOEM(mode = "compare",
#'                          input_data=input_compare,
#'                          dirpath=file.path(dirpath, "compare"))
#'
#'}
#'@export

scPOEM <- function(mode = c("single", "compare"), input_data, dirpath, count_device=1, nComp=5, seed=0, d=100, rebuild_GGN=T, rebuild_PPN=T, rebuild_PGN_Lasso=T, rebuild_PGN_RF=T, rebuild_PGN_XGB=T, relearn_pg_embedding=T) {

  required_keys <- c("X", "Y", "peak_data", "gene_data", "cell_data", "neibor_peak", "genome")

  check_keys <- function(data, data_name = NULL) {
    missing_keys <- setdiff(required_keys, names(data))
    if (length(missing_keys) > 0) {
      stop(paste("Missing keys in", ifelse(is.null(data_name), "input", data_name), ":",
                 paste(missing_keys, collapse = ", ")))
    }
  }

  check_rebuild <- function(rebuild_GGN, rebuild_PPN, rebuild_PGN_Lasso, rebuild_PGN_RF, rebuild_PGN_XGB, relearn_pg_embedding, expected_dims, dirpath){
    # pg_embedding
    if (!relearn_pg_embedding) {
      if (!file.exists(file.path(dirpath, "embedding/node_embeddings.mtx"))) {
        stop("File 'node_embeddings.mtx' not found. Either set reblearn_pg_embedding = TRUE or provide this file.")
      }
      if (!file.exists(file.path(dirpath, "embedding/node_used_peak.csv"))) {
        stop("File 'node_used_peak.csv' not found. Either set reblearn_pg_embedding = TRUE or provide this file.")
      }
      if (!file.exists(file.path(dirpath, "embedding/node_used_gene.csv"))) {
        stop("File 'node_used_gene.csv' not found. Either set reblearn_pg_embedding = TRUE or provide this file.")
      }
      E <- readMM(file.path(dirpath, "embedding/node_embeddings.mtx"))
      peak_node <- read.csv(file.path(dirpath, "embedding/node_used_peak.csv"), header = FALSE)
      gene_node <- read.csv(file.path(dirpath, "embedding/node_used_gene.csv"), header = FALSE)
      if (!all.equal(dim(E), c(dim(peak_node)[1]+dim(gene_node)[1], d))) {
        stop(sprintf("pg_embedding dimension incorrect. Expected %s * %s.", dim(peak_node)[0]+dim(gene_node)[0], d))
      }
    }

    # GGN
    if (!rebuild_GGN) {
      ggn_path <- file.path(dirpath, "test/GGN.mtx")
      if (!file.exists(ggn_path)) {
        stop("File 'GGN.mtx' not found. Either set rebuild_GGN = TRUE or provide this file.")
      }
      gg_net <- readMM(ggn_path)
      if (!identical(dim(gg_net), expected_dims$GGN)) {
        stop(sprintf("GGN dimension incorrect. Expected %s * %s.", expected_dims$GGN[1], expected_dims$GGN[2]))
      }
    }

    # PPN
    if (!rebuild_PPN) {
      ppn_path <- file.path(dirpath, "test/PPN.mtx")
      if (!file.exists(ppn_path)) {
        stop("File 'PPN.mtx' not found. Either set rebuild_PPN = TRUE or provide this file.")
      }
      pp_net <- readMM(ppn_path)
      if (!identical(dim(pp_net), expected_dims$PPN)) {
        stop(sprintf("PPN dimension incorrect. Expected %s * %s.", expected_dims$PPN[1], expected_dims$PPN[2]))
      }
    }

    # PGN_Lasso
    if (!rebuild_PGN_Lasso) {
      lasso_path <- file.path(dirpath, "test/PGN_Lasso.mtx")
      if (!file.exists(lasso_path)) {
        stop("File 'PGN_Lasso.mtx' not found. Either set rebuild_PGN_Lasso = TRUE or provide this file.")
      }
      net_lasso <- readMM(lasso_path)
      if (!identical(dim(net_lasso), expected_dims$PGN_Lasso)) {
        stop(sprintf("PGN_Lasso dimension incorrect. Expected %s * %s.", expected_dims$PGN_Lasso[1], expected_dims$PGN_Lasso[2]))
      }
    }

    # PGN_RF
    if (!rebuild_PGN_RF) {
      rf_path <- file.path(dirpath, "test/PGN_RF.mtx")
      if (!file.exists(rf_path)) {
        stop("File 'PGN_RF.mtx' not found. Either set rebuild_PGN_RF = TRUE or provide this file.")
      }
      net_RF <- readMM(rf_path)
      if (!identical(dim(net_RF), expected_dims$PGN_RF)) {
        stop(sprintf("PGN_RF dimension incorrect. Expected %s * %s.", expected_dims$PGN_RF[1], expected_dims$PGN_RF[2]))
      }
    }

    # PGN_XGB
    if (!rebuild_PGN_XGB) {
      xgb_path <- file.path(dirpath, "test/PGN_XGB.mtx")
      if (!file.exists(xgb_path)) {
        stop("File 'PGN_XGB.mtx' not found. Either set rebuild_PGN_XGB = TRUE or provide this file.")
      }
      net_XGB <- readMM(xgb_path)
      if (!identical(dim(net_XGB), expected_dims$PGN_XGB)) {
        stop(sprintf("PGN_XGB dimension incorrect. Expected %s * %s.", expected_dims$PGN_XGB[1], expected_dims$PGN_XGB[2]))
      }
    }

  }

  if (mode == "single") {
    if (!is.list(input_data)) stop("For mode='single', input_data must be a list.")
    check_keys(input_data)
    cat("Processing single state...\n")
    n_peaks <- dim(input_data$X)[2]
    n_genes <- dim(input_data$Y)[2]
    expected_dims <- list(
      GGN = c(n_genes, n_genes),
      PPN = c(n_peaks, n_peaks),
      PGN_Lasso = c(n_peaks, n_genes),
      PGN_RF = c(n_peaks, n_genes),
      PGN_XGB = c(n_peaks, n_genes)
    )
    cat("Check whether to rebuild data or load from existing files with correct dimensions.\n")
    check_rebuild(rebuild_GGN, rebuild_PPN, rebuild_PGN_Lasso, rebuild_PGN_RF, rebuild_PGN_XGB, relearn_pg_embedding, expected_dims, dirpath)
    cat("Construct gene-gene net.\n")
    gg_net <- GGN(input_data$Y, dirpath, count_device, nComp, rebuild_GGN)
    cat("Construct peak-peak net.\n")
    pp_net <- PPN(input_data$X, input_data$peak_data, input_data$cell_data, input_data$genome, dirpath, rebuild_PPN, seed=seed)
    cat("Construct peak-gene net via Lasso.\n")
    net_lasso <- PGN_Lasso(input_data$X, input_data$Y, input_data$gene_data, input_data$neibor_peak, dirpath, count_device, rebuild_PGN_Lasso)
    cat("Construct peak-gene net via random forest.\n")
    net_RF <- PGN_RF(input_data$X, input_data$Y, input_data$gene_data, input_data$neibor_peak, dirpath, count_device, rebuild_PGN_RF, seed=seed)
    cat("Construct peak-gene net via XGBoost.\n")
    net_XGB <- PGN_XGBoost(input_data$X, input_data$Y, input_data$gene_data, input_data$neibor_peak, dirpath, count_device, rebuild_PGN_XGB)
    cat("Learn low-dimensional representations for peaks and genes.\n")
    single_result <- pg_embedding(gg_net, pp_net, net_lasso, net_RF, net_XGB, dirpath, relearn_pg_embedding, d=d, seed=seed)
    return(single_result)

  } else if (mode == "compare") {
    if (!is.list(input_data) || is.null(names(input_data))) {
      stop("For mode='compare', input_data must be a named list of states.")
    }
    state_names <- names(input_data)
    compare_result <- list()
    for (state_name in state_names) {
      dirpath_s <- file.path(dirpath, state_name)
      state_data <- input_data[[state_name]]
      if (!is.list(state_data)) stop(paste("State", state_name, "must be a list."))
      check_keys(state_data, state_name)
      cat(paste("Processing state:", state_name, "\n"))
      n_peaks <- dim(state_data$X)[2]
      n_genes <- dim(state_data$Y)[2]
      expected_dims <- list(
        GGN = c(n_genes, n_genes),
        PPN = c(n_peaks, n_peaks),
        PGN_Lasso = c(n_peaks, n_genes),
        PGN_RF = c(n_peaks, n_genes),
        PGN_XGB = c(n_peaks, n_genes)
      )
      cat("Check whether to rebuild data or load from existing files with correct dimensions.\n")
      check_rebuild(rebuild_GGN, rebuild_PPN, rebuild_PGN_Lasso, rebuild_PGN_RF, rebuild_PGN_XGB, relearn_pg_embedding, expected_dims, dirpath_s)
      cat("Construct gene-gene net.\n")
      gg_net <- GGN(state_data$Y, dirpath_s, count_device, nComp, rebuild_GGN)
      cat("Construct peak-peak net.\n")
      pp_net <- PPN(state_data$X, state_data$peak_data, state_data$cell_data, state_data$genome, dirpath_s, rebuild_PPN, seed=seed)
      cat("Construct peak-gene net via Lasso.\n")
      net_lasso <- PGN_Lasso(state_data$X, state_data$Y, state_data$gene_data, state_data$neibor_peak, dirpath_s, count_device, rebuild_PGN_Lasso)
      cat("Construct peak-gene net via random forest.\n")
      net_RF <- PGN_RF(state_data$X, state_data$Y, state_data$gene_data, state_data$neibor_peak, dirpath_s, count_device, rebuild_PGN_RF, seed=seed)
      cat("Construct peak-gene net via XGBoost.\n")
      net_XGB <- PGN_XGBoost(state_data$X, state_data$Y, state_data$gene_data, state_data$neibor_peak, dirpath_s, count_device, rebuild_PGN_XGB)
      cat("Learn low-dimensional representations for peaks and genes.\n")
      compare_result[[state_name]] <- pg_embedding(gg_net, pp_net, net_lasso, net_RF, net_XGB, dirpath_s, relearn_pg_embedding, d=d, seed=seed)
    }
    cat("Align genes between two states and identify differentially regulated genes.\n")
    compare_result$compare <- align_embedding(gene_data1 = input_data[[state_names[1]]]$gene_data,
                                              gene_node1 = compare_result[[state_names[1]]]$gene_node,
                                              E1 = compare_result[[state_names[1]]]$E,
                                              gene_data2 = input_data[[state_names[2]]]$gene_data,
                                              gene_node2 = compare_result[[state_names[2]]]$gene_node,
                                              E2 = compare_result[[state_names[2]]]$E,
                                              dirpath,
                                              d=d)
    return(compare_result)
  }
  cat("mode is wrong!\n")
}
