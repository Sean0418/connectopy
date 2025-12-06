Quickstart Guide
================

This guide will walk you through the basic usage of the brain-connectome package.

Loading Data
------------

The ``ConnectomeDataLoader`` class handles loading and merging HCP data:

.. code-block:: python

   from brain_connectome.data import ConnectomeDataLoader

   # Initialize with data directory
   loader = ConnectomeDataLoader("data/")

   # Load raw structural connectome
   sc_matrix, subject_ids = loader.load_structural_connectome()

   # Load TNPCA coefficients
   struct_pca, struct_ids = loader.load_tnpca_structural()
   func_pca, func_ids = loader.load_tnpca_functional()

   # Load merged dataset with all features
   merged = loader.load_merged_dataset()
   print(f"Merged dataset: {merged.shape}")

Preprocessing
-------------

Preprocess raw connectome matrices for analysis:

.. code-block:: python

   from brain_connectome.data import preprocess_connectome

   # Full preprocessing pipeline
   X_processed, metadata = preprocess_connectome(
       sc_matrix,
       log_transform_data=True,
       normalize=True,
       remove_zero_var=True
   )

   print(f"Processed shape: {X_processed.shape}")
   print(f"Features removed: {metadata.get('n_removed_features', 0)}")

PCA Analysis
------------

Perform PCA on connectome features:

.. code-block:: python

   from brain_connectome.analysis import ConnectomePCA

   # Initialize and fit PCA
   pca = ConnectomePCA(n_components=60)
   scores = pca.fit_transform(X_processed)

   # Check variance explained
   print(f"Total variance explained: {pca.total_variance_explained_:.2%}")

   # Get variance report
   report = pca.get_variance_report()
   print(report.head(10))

   # Convert to DataFrame with subject IDs
   pca_df = pca.to_dataframe(X_processed, subject_ids, prefix="Raw_Struct_PC")

Sexual Dimorphism Analysis
--------------------------

Analyze sex differences in brain connectivity:

.. code-block:: python

   from brain_connectome.analysis import DimorphismAnalysis

   # Initialize analysis
   analysis = DimorphismAnalysis(merged, gender_column="Gender")

   # Run analysis on PCA features
   pc_columns = [f"Struct_PC{i}" for i in range(1, 61)]
   results = analysis.analyze(feature_columns=pc_columns)

   # Get significant features
   significant = results[results["Significant"]]
   print(f"Significant features: {len(significant)}")

   # Get top features by effect size
   top_features = analysis.get_top_features(n=10)
   print(top_features)

   # Summary statistics
   summary = analysis.summary()
   print(summary)

Machine Learning Classification
-------------------------------

Train classifiers to predict traits from connectome features:

.. code-block:: python

   from brain_connectome.models import ConnectomeRandomForest
   from sklearn.model_selection import train_test_split

   # Prepare data
   feature_cols = [c for c in merged.columns if c.startswith("Struct_PC")]
   X = merged[feature_cols].values
   y = (merged["Gender"] == "M").astype(int).values

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
   )

   # Train Random Forest
   clf = ConnectomeRandomForest(n_estimators=500)
   clf.fit(X_train, y_train, feature_names=feature_cols)

   # Evaluate
   metrics = clf.evaluate(X_test, y_test)
   print(f"Accuracy: {metrics['accuracy']:.2%}")

   # Feature importance
   top_features = clf.get_top_features(n=10)
   print(top_features)

Visualization
-------------

Create publication-ready plots:

.. code-block:: python

   from brain_connectome.visualization import (
       plot_pca_scatter,
       plot_dimorphism_comparison,
       plot_feature_importance,
   )
   import matplotlib.pyplot as plt

   # PCA scatter plot by gender
   fig, ax = plot_pca_scatter(
       merged,
       pc_x="Struct_PC1",
       pc_y="Struct_PC2",
       hue="Gender",
       title="Structural Connectome by Gender"
   )
   plt.show()

   # Feature importance plot
   fig, ax = plot_feature_importance(
       clf.feature_importances_,
       n_features=15,
       title="Top Biomarkers for Gender Classification"
   )
   plt.show()

Complete Pipeline Example
-------------------------

Here's a complete example combining all steps:

.. code-block:: python

   from brain_connectome import (
       ConnectomeDataLoader,
       ConnectomePCA,
       DimorphismAnalysis,
   )
   from brain_connectome.models import ConnectomeRandomForest
   from brain_connectome.visualization import plot_pca_scatter

   # 1. Load data
   loader = ConnectomeDataLoader("data/")
   data = loader.load_merged_dataset()

   # 2. Analyze dimorphism
   analysis = DimorphismAnalysis(data)
   features = [f"Struct_PC{i}" for i in range(1, 61)]
   results = analysis.analyze(feature_columns=features)

   print("Top 5 dimorphic features:")
   print(analysis.get_top_features(5))

   # 3. Train classifier
   X = data[features].values
   y = (data["Gender"] == "M").astype(int).values

   clf = ConnectomeRandomForest()
   clf.fit(X, y, feature_names=features)

   print(f"\\nFeature importance (top 5):")
   print(clf.get_top_features(5))

   # 4. Visualize
   fig, ax = plot_pca_scatter(data, title="Sexual Dimorphism in Brain Connectivity")
   fig.savefig("output/dimorphism_plot.png", dpi=300)

