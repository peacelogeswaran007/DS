# IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# SET STYLES
sns.set(style="whitegrid")

# LOAD DATA
df = pd.read_csv('train.csv')

# ----------------------
# 1. CORRELATION HEATMAP
# ----------------------

# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Select top correlated features with SalePrice
top_corr = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(11)
top_features = top_corr.index

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Top Correlated Features with SalePrice", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------------------------
# 2. PRICE DISTRIBUTION PLOT
# ---------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df['SalePrice'], kde=True, bins=40, color='skyblue')
plt.title("Distribution of House Sale Prices", fontsize=14)
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()