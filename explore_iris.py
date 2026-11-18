import pandas as pd

# Load the Iris dataset
data = pd.read_csv("iris dataset.csv")

# Display the first five rows
print("FIRST FIVE ROWS")
print(data.head())

# Display the dataset size
print("\nDATASET SHAPE")
print(data.shape)

# Display column information
print("\nDATASET INFORMATION")
print(data.info())

# Check for missing values
print("\nMISSING VALUES")
print(data.isnull().sum())

# Count the different flower species
print("\nFLOWER SPECIES COUNT")
print(data["species"].value_counts())

# Display basic statistical information
print("\nSTATISTICAL SUMMARY")
print(data.describe())