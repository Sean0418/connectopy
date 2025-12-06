# Package environment to store Python module references
.pkg_env <- new.env(parent = emptyenv())

#' @importFrom reticulate import py_module_available
.onLoad <- function(libname, pkgname) {
  # Delay loading Python modules until they're needed
  .pkg_env$brain_connectome <- NULL
}

#' Get the brain_connectome Python module
#'
#' @return The brain_connectome Python module
#' @keywords internal
get_brain_connectome <- function() {
  if (is.null(.pkg_env$brain_connectome)) {
    if (!reticulate::py_module_available("brain_connectome")) {
      stop(
        "Python package 'brain_connectome' is not installed.\n",
        "Install it with: pip install -e /path/to/Brain-Connectome",
        call. = FALSE
      )
    }
    .pkg_env$brain_connectome <- reticulate::import("brain_connectome")
  }
  .pkg_env$brain_connectome
}

#' Configure Python environment for brainconnectome
#'
#' Set the Python environment to use. Call this before using any other
#' functions if you need to use a specific Python installation.
#'
#' @param python Path to Python executable, or NULL to use default
#' @param virtualenv Path to virtualenv, or NULL
#' @param condaenv Name of conda environment, or NULL
#'
#' @export
#' @examples
#' \dontrun{
#' # Use a specific virtualenv
#' configure_python(virtualenv = "~/.venv/brain-connectome")
#'
#' # Use a conda environment
#' configure_python(condaenv = "brain-connectome")
#' }
configure_python <- function(python = NULL, virtualenv = NULL, condaenv = NULL) {
  if (!is.null(virtualenv)) {
    reticulate::use_virtualenv(virtualenv, required = TRUE)
  } else if (!is.null(condaenv)) {
    reticulate::use_condaenv(condaenv, required = TRUE)
  } else if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
  }

  # Reset cached module

.pkg_env$brain_connectome <- NULL

  invisible(TRUE)
}
