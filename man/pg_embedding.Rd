% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pg_embedding.R
\name{pg_embedding}
\alias{pg_embedding}
\title{pg_embedding}
\usage{
pg_embedding(
  gg_net,
  pp_net,
  net_lasso,
  net_RF,
  net_XGB,
  dirpath,
  relearn_pg_embedding = T,
  d = 100,
  seed = 0
)
}
\arguments{
\item{gg_net}{The gene-gene network.}

\item{pp_net}{The peak-peak network.}

\item{net_lasso}{The peak-gene network constructed by Lasso.}

\item{net_RF}{The peak-gene network constructed by Random Forest.}

\item{net_XGB}{The peak-gene network constructed by XGBoost.}

\item{dirpath}{The folder path to read or write file.}

\item{relearn_pg_embedding}{Logical. Whether to relearn the low-dimensional representations for peaks and genes from scratch. If FALSE, the function will attempt to read from \cr'node_embeddings.mtx', 'node_used_peak.csv', 'node_used_gene.csv' \cr under \code{dirpath/embedding} in \code{single} mode or \cr \code{dirpath/state_name/embedding} in \code{compare} mode.}

\item{d}{The dimension of latent space.}

\item{seed}{An integer specifying the random seed to ensure reproducible results.}
}
\value{
a list containing the following \describe{
\item{\code{E}}{low-dimensional representations of peaks and genes}
\item{\code{peak_node}}{Peak ids that are associated with other peaks or genes.}
\item{\code{gene_node}}{Gene ids that are associated with other peaks or genes.}
}
}
\description{
Learn the low-dimensional representations for peaks and genes with a meta-path based method.
}
\examples{
\dontrun{
library(scPOEM)
library(Matrix)
library(data.table)
dirpath <- "./example_data"
# Download single mode example data
data(input_single)
gg_net <- GGN(input_single$Y, file.path(dirpath, "single"), 1, 5, T)
pp_net <- PPN(input_single$X, input_single$peak_data,
              input_single$cell_data, input_single$genome,
              file.path(dirpath, "single"))
net_Lasso <- PGN_Lasso(input_single$X, input_single$Y,
                       input_single$gene_data, input_single$neibor_peak,
                       file.path(dirpath, "single"))
net_RF <- PGN_RF(input_single$X, input_single$Y,
                 input_single$gene_data, input_single$neibor_peak,
                 file.path(dirpath, "single"))
net_XGB <- PGN_XGBoost(input_single$X, input_single$Y,
                       input_single$gene_data, input_single$neibor_peak,
                       file.path(dirpath, "single"))
E_result <- pg_embedding(gg_net, pp_net, net_lasso, net_RF, net_XGB,
                         file.path(dirpath, "single"))
}

}
