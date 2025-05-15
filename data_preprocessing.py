# BEFORE: Show top 5 columns with most missing values
print("🔍 Missing Values BEFORE Transformation:\n")
print(df.isnull().sum().sort_values(ascending=False).head())

# Fill missing values (for numeric + object columns)
df_filled = df.fillna(0)

# AFTER: Confirm missing values handled
print("\n✅ Missing Values AFTER Transformation:\n")
print(df_filled.isnull().sum().sort_values(ascending=False).head())