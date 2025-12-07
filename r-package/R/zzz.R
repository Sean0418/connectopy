# Package environment to store Python module references
.pkg_env <- new.env(parent = emptyenv())

#' @importFrom reticulate import py_module_available
.onLoad <- function(libname, pkgname) {
  # Delay loading Python modules until they're needed
  .pkg_env$connectopy <- NULL
}

#' Get the connectopy Python module
#'
#' @return The connectopy Python module
#' @keywords internal
get_connectopy <- function() {
  if (is.null(.pkg_env$connectopy)) {
    if (!reticulate::py_module_available("connectopy")) {
      stop(
        "Python package 'connectopy' is not installed.\n",
        "Install it with: pip install -e /path/to/Brain-Connectome",
        call. = FALSE
      )
    }
    .pkg_env$connectopy <- reticulate::import("connectopy")
  }
  .pkg_env$connectopy
}

#' Configure Python environment for connectopy
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
#' configure_python(virtualenv = "~/.venv/connectopy")
#'
#' # Use a conda environment
#' configure_python(condaenv = "connectopy")
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

.pkg_env$connectopy <- NULL

  invisible(TRUE)
}
