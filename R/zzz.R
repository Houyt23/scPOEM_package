.onLoad <- function(libname, pkgname) {
  #python conda environment name
  options(scPOEM_envname = "scPOEM_env")

  # try to use scPOEM_env）
  tryCatch({
    reticulate::use_condaenv("scPOEM_env", required = TRUE)
  }, error = function(e) {
    options(scPOEM_env_missing = TRUE)
  })
}

.onAttach <- function(libname, pkgname) {
  envname <- getOption("scPOEM_envname", "scPOEM_env")

  if (isTRUE(getOption("scPOEM_env_missing"))) {
    packageStartupMessage(
      "Could not find the conda environment named '", envname, "'.\n",
      "Please create the environment and install the required packages via the terminal:\n",
      "conda create --name scPOEM_env python=3.9\n",
      "conda activate scPOEM_env\n",
      "pip install numpy\n",
      "pip install scipy\n",
      "pip install pandas\n",
      "pip install scikit-learn\n",
      "pip install matplotlib\n",
      "pip install tqdm\n",
      "pip install ray\n",
      "pip install tensorflow\n"
    )
  }

  # Check whether required Python packages are installed
  required_py_packages <- c(
    tensorflow = "tensorflow",
    scipy = "scipy",
    numpy = "numpy",
    pandas = "pandas",
    sklearn = "scikit-learn",
    matplotlib = "matplotlib",
    tqdm = "tqdm",
    ray = "ray"
  )
  missing <- sapply(names(required_py_packages), function(pkg) !reticulate::py_module_available(pkg))

  if (any(missing)) {
    # Get install names for missing packages
    missing_install_names <- required_py_packages[missing]
    packageStartupMessage(
      "Could not find these Python packages in '", envname, "':\n",
      paste(missing_install_names, collapse = "\n"), "\n",
      "Please install them in the conda environment '", envname, "'."
    )
  }
}
