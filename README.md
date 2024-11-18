# WEEK-7-PYTHON
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset from sklearn
iris_data = load_iris()

# Convert it to a pandas DataFrame
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target

# Display the first few rows of the dataset
print(df.head())

# Check data types and missing values
print(df.info())
print(df.isnull().sum())  # Check for missing values

# Drop missing values (if any)
df.dropna(inplace=True)

# Or, alternatively, fill missing values with the mean
# df.fillna(df.mean(), inplace=True)
