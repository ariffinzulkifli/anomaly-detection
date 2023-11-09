import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Read the dataset
file_path = './dataset/melbourne_houses_price.csv'  # Update to the correct file path
houses_price_data = pd.read_csv(file_path)

# Select only the numeric columns for median calculation
numeric_columns = houses_price_data.select_dtypes(include=[np.number])

# Calculate medians only for numeric columns
numeric_medians = numeric_columns.median()

# Apply the median to fill missing values only in the numeric columns
houses_price_data[numeric_columns.columns] = numeric_columns.fillna(numeric_medians)

# Standardize the numeric columns
X = StandardScaler().fit_transform(houses_price_data[numeric_columns.columns])
X1 = pd.DataFrame(X, columns=numeric_columns.columns)

def plot_model(labels, alg_name, plot_index):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, plot_index)
    color_code = {'anomaly':'red', 'normal':'green'}
    colors = [color_code[x] for x in labels]

    ax.scatter(X1.iloc[:, 0], X1.iloc[:, 1], color=colors, marker='.', label='red = anomaly')
    ax.legend(loc="lower right")
    ax.set_title(alg_name)

    # In a Jupyter notebook environment, use plt.show() instead of fig.show()
    plt.show()

# Fit the DBSCAN model
model = DBSCAN(eps=0.63).fit(X1)
labels = model.labels_

# Label -1 is considered an anomaly
labels = ['anomaly' if x == -1 else 'normal' for x in labels]

# Plot the model
plot_model(labels, 'DBSCAN', 1)
