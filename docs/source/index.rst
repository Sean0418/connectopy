Connectopy Documentation
==============================

**connectopy** is a Python package for analyzing brain structural and functional
connectomes from the Human Connectome Project (HCP).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index

Features
--------

* **Data Loading**: Load and merge HCP structural/functional connectome data with traits
* **Dimensionality Reduction**: PCA and VAE for connectome feature extraction
* **Statistical Analysis**: Sexual dimorphism analysis with effect sizes and FDR correction
* **Machine Learning**: Random Forest and XGBoost classifiers for trait prediction
* **Visualization**: Publication-ready plots for connectome analysis

Quick Example
-------------

.. code-block:: python

   from connectopy import ConnectomeDataLoader, ConnectomePCA, DimorphismAnalysis

   # Load data
   loader = ConnectomeDataLoader("data/")
   merged_data = loader.load_merged_dataset()

   # Analyze sexual dimorphism
   analysis = DimorphismAnalysis(merged_data)
   results = analysis.analyze(feature_columns=["Struct_PC1", "Struct_PC2"])

   # Get significant features
   significant = results[results["Significant"]]
   print(significant)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
