#'@title Example Input Data for Single Mode Analysis
#'@description A list containing example single-cell multi-omics data used in "single" mode of the `scPOEM` package.
#'
#'@format A named list with 7 elements:
#'\describe{
#'    \item{\code{X}}{Gene expression matrix.}
#'    \item{\code{Y}}{Peak accessibility matrix.}
#'    \item{\code{peak_data}}{A data.frame containing peak information.}
#'    \item{\code{gene_data}}{A data.frame containing gene information (must contain column "gene_name").}
#'    \item{\code{cell_data}}{A data.frame containing cell metadata.}
#'    \item{\code{neibor_peak}}{The peak IDs within a certain range of each gene, must have cols c("gene_name", "start_use", "end_use"). The id numbers in "start_use" and "end_use" are start from 0.}
#'    \item{\code{genome}}{The genome length for the species.}
#'}
#'
#'@usage data(input_single)
#'
#'@keywords datasets
#'
#'@examples
#'data(input_single)
#'
#'@name input_single
"input_single"
