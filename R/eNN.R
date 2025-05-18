utils::globalVariables(c("get_eNN"))
#'@title eNN
#'
#'@description Make gene-gene net after meta-path based embedding via epsilon-NN.
#'@name eNN
#'@importFrom reticulate source_python
#'@importFrom tictoc tic toc
#'@param E_g Embedding representations of genes.
#'@return The eNN network.
#'
#'@export

eNN <- function(E_g) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Please install it with install.packages('reticulate').")
  }
  cat("Construct gene-gene network using low-dimensional representations via epsilon-NN...\n")
  tic()
  py_script <- system.file("python/eNN.py", package = "scPOEM")

  source_python(py_script)

  epsilon_net <- get_eNN(E_g)
  toc()
  cat("\n")
  return(epsilon_net)
}
