 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
df = pd.read_csv('AirQualityDataset.csv')

# CLEAN COLUMN NAMES
df.columns = df.columns.str.strip().str.lower()

# BASIC INFO
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# OUTLIER DETECTION
plt.figure(figsize=(7,4))
sns.boxplot(x=df["pollutant_avg"], color="orange")

plt.title("Outlier Detection using Boxplot")
plt.tight_layout()
plt.show()
