import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
data = pd.read_csv("Iris dataset.csv")

# Get the different flower species
species = data["species"].unique()


# GRAPH 1: Number of flowers in each species
species_counts = data["species"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(species_counts.index, species_counts.values)
plt.title("Distribution of Iris Flower Species")
plt.xlabel("Flower Species")
plt.ylabel("Number of Flowers")
plt.savefig("species_distribution.png")
plt.show()


# GRAPH 2: Sepal Length vs Sepal Width
plt.figure(figsize=(8, 5))

for flower in species:
    subset = data[data["species"] == flower]
    plt.scatter(
        subset["sepal_length"],
        subset["sepal_width"],
        label=flower
    )

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.savefig("sepal_scatter.png")
plt.show()


# GRAPH 3: Petal Length vs Petal Width
plt.figure(figsize=(8, 5))

for flower in species:
    subset = data[data["species"] == flower]
    plt.scatter(
        subset["petal_length"],
        subset["petal_width"],
        label=flower
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.savefig("petal_scatter.png")
plt.show()


# GRAPH 4: Correlation Heatmap
correlation = data[
    ["sepal_length", "sepal_width", "petal_length", "petal_width"]
].corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar(label="Correlation")

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(j, i, round(correlation.iloc[i, j], 2),
                 ha="center", va="center")

plt.title("Correlation Between Iris Features")
plt.tight_layout()
plt.savefig("feature_correlation.png")
plt.show()


print("All four graphs have been created and saved!")