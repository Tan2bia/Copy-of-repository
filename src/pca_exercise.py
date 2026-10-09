"""
INTEG 275 — Technical Setup Exercise: Dimension Reduction

Goal: reduce four body-measurement variables for the Palmer Penguins
dataset down to two dimensions using Principal Component Analysis
(PCA), then make a scatter plot to see whether the three penguin
species separate out.

Data: data/penguins.csv (344 penguins, 3 species, Palmer Archipelago,
Antarctica). Source: Gorman, Williams & Fraser (2014), PLOS ONE,
via the palmerpenguins project (Horst, Hill & Gorman, 2020),
https://allisonhorst.github.io/palmerpenguins/

Before starting this exercise, have a brief look at the data by 
opening the CSV file to understand its structure and check for missing values.

How to use this script:
  Each TODO below describes one step. Use GitHub Copilot (inline
  suggestions, or Copilot Chat with Ctrl+I / Cmd+I) to help you
  write the code for that step. A suggested prompt is included as a
  comment under each TODO if you want a starting point, but try
  describing it in your own words first.

  Run the script from the integrated terminal with:
      python src/pca_exercise.py

  A successful run saves a plot to outputs/pca_scatter.png. Compare
  your plot to docs/expected_output_example.png to see roughly what
  a correct result should look like (your exact colours/layout may
  differ, that's fine).
"""

# TODO 1: Import the libraries you'll need.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# TODO 2: Load the dataset.
penguins = pd.read_csv("data/penguins.csv", na_values="NA")

# TODO 3: Drop incomplete rows.
penguins = penguins.dropna().reset_index(drop=True)

# TODO 4: Select the numeric feature columns.
features = penguins[
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
].copy()

# TODO 5: Standardize the features.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# TODO 6: Run PCA.
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

# TODO 7: Put the results in a DataFrame.
pca_df = pd.DataFrame(
    {
        "PC1": principal_components[:, 0],
        "PC2": principal_components[:, 1],
        "species": penguins["species"].values,
    }
)

# TODO 8: Plot and save the result.
colors = {
    "Adelie": "#4C72B0",
    "Chinstrap": "#DD8452",
    "Gentoo": "#55A868",
}

plt.figure(figsize=(10, 7))
for species, color in colors.items():
    subset = pca_df[pca_df["species"] == species]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=45,
        color=color,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.8,
        label=species,
    )

plt.title("PCA of Palmer Penguins body measurements", fontsize=22)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance explained)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance explained)")
plt.legend(title="Species", loc="upper left", frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig("outputs/pca_scatter.png", dpi=200)
plt.close()

# TODO 9 (optional stretch): print a short summary.
print(
    "Total variance explained by first two components:",
    sum(pca.explained_variance_ratio_),
)
print(pca_df.head())

# TODO 10 (optional stretch): Include the year column as a feature.
# The required result for the main assignment is already saved above.
