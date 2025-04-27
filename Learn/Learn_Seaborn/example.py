import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load an example dataset
data = sns.load_dataset("penguins")

# Create a pairplot to visualize relationships between numerical features
sns.pairplot(data, hue="species", diag_kind="kde")
plt.suptitle("Pairplot of Penguin Dataset", y=1.02)
plt.show()

# Create a heatmap to visualize correlations between numerical features
correlation_matrix = data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Create a boxplot to compare flipper length across species
sns.boxplot(data=data, x="species", y="flipper_length_mm", palette="Set2")
plt.title("Boxplot of Flipper Length by Species")
plt.xlabel("Species")
plt.ylabel("Flipper Length (mm)")
plt.show()

# Create a violin plot to visualize the distribution of body mass across species
sns.violinplot(data=data, x="species", y="body_mass_g", inner="quartile", palette="muted")
plt.title("Violin Plot of Body Mass by Species")
plt.xlabel("Species")
plt.ylabel("Body Mass (g)")
plt.show()

# Create a swarmplot to visualize individual data points for bill length
sns.swarmplot(data=data, x="species", y="bill_length_mm", hue="sex", dodge=True, palette="deep")
plt.title("Swarm Plot of Bill Length by Species and Sex")
plt.xlabel("Species")
plt.ylabel("Bill Length (mm)")
plt.legend(title="Sex")
plt.show()