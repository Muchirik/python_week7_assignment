import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
# Load dataset (Iris dataset from seaborn for simplicity)
df = sns.load_dataset("iris")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# No missing values, so no cleaning needed

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
species_mean = df.groupby("species").mean()
print("\nMean values per species:")
print(species_mean)

# Task 3: Data Visualization
plt.figure(figsize=(12, 8))

# Line plot (dummy example since iris has no time variable)
plt.subplot(2, 2, 1)
df.groupby("species").mean()["sepal_length"].plot(kind="line", marker='o', title="Average Sepal Length per Species")
plt.ylabel("Sepal Length")

# Bar chart
plt.subplot(2, 2, 2)
sns.barplot(x="species", y="petal_length", data=df, ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length")

# Histogram
plt.subplot(2, 2, 3)
sns.histplot(df["sepal_length"], bins=15, kde=True)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")

# Scatter plot
plt.subplot(2, 2, 4)
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title="Species")

plt.tight_layout()
plt.show()

# Summary of findings
print("\nFindings:")
print("- The dataset contains three species: setosa, versicolor, and virginica.")
print("- Setosa species tend to have smaller petal lengths and widths.")
print("- Virginica species have the highest petal and sepal lengths on average.")
print("- There is a strong positive correlation between sepal length and petal length.")