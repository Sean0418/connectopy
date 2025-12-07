#' Create a Connectome Data Loader
#'
#' Initialize a data loader for HCP connectome data.
#'
#' @param data_dir Path to the data directory containing raw/ and processed/
#'   subdirectories.
#'
#' @return A ConnectomeDataLoader object (Python reference)
#' @export
#'
#' @examples
#' \dontrun{
#' loader <- ConnectomeDataLoader("data/")
#' data <- load_merged_dataset(loader)
#' }
ConnectomeDataLoader <- function(data_dir) {
  bc <- get_connectopy()
  bc$ConnectomeDataLoader(data_dir)
}

#' Load Structural Connectome
#'
#' Load raw structural connectome matrices from .mat files.
#'
#' @param loader A ConnectomeDataLoader object
#'
#' @return A list with 'connectome' (3D array) and 'subject_ids' (vector)
#' @export
load_structural_connectome <- function(loader) {
  result <- loader$load_structural_connectome()
  list(
    connectome = result[[1]],
    subject_ids = result[[2]]
  )
}

#' Load Functional Connectome
#'
#' Load raw functional connectome matrices from .mat files.
#'
#' @param loader A ConnectomeDataLoader object
#'
#' @return A list with 'connectome' (list of matrices) and 'subject_ids' (vector)
#' @export
load_functional_connectome <- function(loader) {
  result <- loader$load_functional_connectome()
  list(
    connectome = result[[1]],
    subject_ids = result[[2]]
  )
}

#' Load TNPCA Structural Coefficients
#'
#' Load Tensor Network PCA coefficients for structural connectome.
#'
#' @param loader A ConnectomeDataLoader object
#'
#' @return A list with 'coefficients' (matrix) and 'subject_ids' (vector)
#' @export
load_tnpca_structural <- function(loader) {
  result <- loader$load_tnpca_structural()
  list(
    coefficients = result[[1]],
    subject_ids = result[[2]]
  )
}

#' Load TNPCA Functional Coefficients
#'
#' Load Tensor Network PCA coefficients for functional connectome.
#'
#' @param loader A ConnectomeDataLoader object
#'
#' @return A list with 'coefficients' (matrix) and 'subject_ids' (vector)
#' @export
load_tnpca_functional <- function(loader) {
  result <- loader$load_tnpca_functional()
  list(
    coefficients = result[[1]],
    subject_ids = result[[2]]
  )
}

#' Load Traits Data
#'
#' Load and merge trait data from CSV files.
#'
#' @param loader A ConnectomeDataLoader object
#'
#' @return A data.frame with trait data
#' @export
load_traits <- function(loader) {
  loader$load_traits()
}

#' Load Merged Dataset
#'
#' Load and merge all data sources into a single data frame.
#'
#' @param loader A ConnectomeDataLoader object
#' @param include_raw_pca Logical, whether to include raw PCA scores
#' @param include_vae Logical, whether to include VAE latent dimensions
#'
#' @return A data.frame with all merged features and traits
#' @export
#'
#' @examples
#' \dontrun{
#' loader <- ConnectomeDataLoader("data/")
#' data <- load_merged_dataset(loader)
#' head(data)
#' }
load_merged_dataset <- function(loader, include_raw_pca = FALSE, include_vae = FALSE) {
  loader$load_merged_dataset(
    include_raw_pca = include_raw_pca,
    include_vae = include_vae
  )
}
