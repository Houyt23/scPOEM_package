% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/scPOEM.R
\name{scPOEM}
\alias{scPOEM}
\title{scPOEM}
\usage{
scPOEM(
  mode = c("single", "compare"),
  input_data,
  dirpath,
  count_device = 1,
  nComp = 5,
  seed = 0,
  d = 100,
  rebuild_GGN = T,
  rebuild_PPN = T,
  rebuild_PGN_Lasso = T,
  rebuild_PGN_RF = T,
  rebuild_PGN_XGB = T,
  relearn_pg_embedding = T
)
}
\arguments{
\item{mode}{The mode indicating whether to analyze data from a single condition or to compare two conditions.}

\item{input_data}{A list of input data.

If \code{mode = "single"}, \code{input_data} must be a list containing the following \strong{seven objects}:
\itemize{
\item \code{X}: Gene expression matrix.
\item \code{Y}: Peak accessibility matrix.
\item \code{peak_data}: A data.frame containing peak information.
\item \code{gene_data}: A data.frame containing gene information (must contain column \code{"gene_name"}).
\item \code{cell_data}: A data.frame containing cell metadata.
\item \code{neibor_peak}: The peak IDs within a certain range of each gene, must have cols c("gene_name", "start_use", "end_use"). The id numbers in "start_use" and "end_use" are start from 0.
\item \code{genome}: The genome length for the species.
}

If \code{mode = "compare"}, \code{input_data} must be a \strong{named list of two elements}, with names corresponding to two state names (e.g., "state1" and "state2"). Each element must itself be a list containing the same seven components as described above for \code{mode = "single"}.}

\item{dirpath}{The folder path to read or write file.}

\item{count_device}{The number of cpus used to train models.}

\item{nComp}{The number of PCs used for regression in constructing GGN.}

\item{seed}{An integer specifying the random seed to ensure reproducible results.}

\item{d}{The dimension of latent space.}

\item{rebuild_GGN}{Logical. Whether to rebuild the gene-gene network from scratch. If FALSE, the function will attempt to read from 'GGN.mtx' under \code{dirpath/test} in \code{single} mode or \code{dirpath/state_name/test} in \code{compare} mode.}

\item{rebuild_PPN}{Logical. Whether to rebuild the peak-peak network from scratch. If FALSE, the function will attempt to read from 'PPN.mtx' under \code{dirpath/test} in \code{single} mode or \code{dirpath/state_name/test} in \code{compare} mode.}

\item{rebuild_PGN_Lasso}{Logical. Whether to rebuild the peak-gene network via Lasso from scratch. If FALSE, the function will attempt to read from 'PGN_Lasso.mtx' under \cr \code{dirpath/test} in \code{single} mode or \code{dirpath/state_name/test} in \code{compare} mode.}

\item{rebuild_PGN_RF}{Logical. Whether to rebuild the peak-gene network via random forest from scratch. If FALSE, the function will attempt to read from 'PGN_RF.mtx' under \code{dirpath/test} in \code{single} mode or \code{dirpath/state_name/test} in \code{compare} mode.}

\item{rebuild_PGN_XGB}{Logical. Whether to rebuild the peak-gene network via XGBoost from scratch. If FALSE, the function will attempt to read from 'PGN_XGB.mtx' under \cr \code{dirpath/test} in \code{single} mode or \code{dirpath/state_name/test} in \code{compare} mode.}

\item{relearn_pg_embedding}{Logical. Whether to relearn the low-dimensional representations for peaks and genes from scratch. If FALSE, the function will attempt to read from \cr'node_embeddings.mtx', 'node_used_peak.csv', 'node_used_gene.csv' \cr under \code{dirpath/embedding} in \code{single} mode or \cr \code{dirpath/state_name/embedding} in \code{compare} mode.}
}
\value{
The scPOEM result.
}
\description{
A embedding method that jointly projects chromatin accessibility peaks and expressed genes into a shared low-dimensional space.
}
\examples{
\dontrun{
library(scPOEM)
library(Matrix)
library(data.table)
dirpath <- "./example_data"
# An example for analysing a single dataset.
# Download and read data.
data(input_single)
single_result <- scPOEM(mode = "single",
                        input_data=input_single,
                        dirpath=file.path(dirpath, "single"))

# An example for analysing and comparing datasets from two conditions.
# Download compare mode example data
data(input_compare)
compare_result <- scPOEM(mode = "compare",
                         input_data=input_compare,
                         dirpath=file.path(dirpath, "compare"))

}
}
