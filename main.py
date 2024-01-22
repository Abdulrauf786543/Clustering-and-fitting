# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import scipy.optimize as opt


def csv_reader(filename):
    '''Function to read and clean the csv dataset using pandas'''
    # Read csv using pandas
    df = pd.read_csv(filename, skiprows=(0, 2), index_col='Country Name')
    # Clean dataframe
    df = df.drop(['Country Code', 'Indicator Name',
                  'Indicator Code', 'Unnamed: 67'], axis=1)
    # Take transpose of dataframe
    df = df.T
    # Remove the column name on the index column
    df = df.rename_axis(None, axis=1)
    # Convert the index datatype to integer
    df.index = df.index.astype(int)
    # Extract the rows with non NAN values
    df = df.loc[1990:2020]

    return df


def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """
    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)  # fit done on x,y pairs
    labels = kmeans.labels_

    # Calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score


def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)

    The function does not have a plt.show() at the end so that the user 
    can savethe figure.
    """

    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.colorbar()
    # no plt.show() at the end


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


df_1 = csv_reader('Carbon_Emission.csv')
df_2 = csv_reader('Access_to_Electricity.csv')
df_3 = csv_reader('Agricultural_Land.csv')
df_4 = csv_reader('Arable_Land.csv')
df_5 = csv_reader('Electric_Power_Consumption.csv')
df_6 = csv_reader('Population.csv')

df = pd.DataFrame({'Carbon Emission': df_1['World'],
                   'Access to Electricity': df_2['World'],
                   'Agricultural Land': df_3['World'],
                   'Arable Land': df_4['World'],
                   'Electric Power Consumption': df_5['World'],
                   'Population': df_6['World']})

df_clus = pd.DataFrame({'Carbon Emission': df_1['World'],
                        'Access to Electricity': df_2['World']})

# Drop rows with missing values
df_clus = df_clus.dropna()
df_clus

xy = np.array(df_clus)

# Extract x and y vectors
x = xy[:, 0]
y = xy[:, 1]

n_clus = 3

kmeans = cluster.KMeans(n_clusters=n_clus, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(xy)
labels = kmeans.labels_

# Extract the estimated cluster centres
cen = kmeans.cluster_centers_

# Extract x and y values
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

plt.figure(figsize=(12, 8))
# Plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o")
# Show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d", label="kmeans centres")
plt.scatter(xkmeans, ykmeans, 45, "y", marker="+", label="original centres")

plt.xlabel("Carbon Emission (Kt)")
plt.ylabel("Access to Electricity (%)")
plt.legend()
plt.show()

for ic in range(2, 11):
    score = one_silhoutte(xy, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

map_corr(df)

pd.plotting.scatter_matrix(df, figsize=(14, 14))
plt.show()

# Create a scaler object
scaler = pp.RobustScaler()

# Set up the scaler
scaler.fit(df_clus)

# Apply the scaling
df_norm = scaler.transform(df_clus)

# Plot the figure
plt.figure(figsize=(12, 8))
plt.scatter(df_norm[:, 0], df_norm[:, 1], 10, marker="o")
plt.xlabel("Carbon Emission (Kt)")
plt.ylabel("Access to Electricity (%)")
plt.show()


# Set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=n_clus, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm)
# Extract cluster labels
labels = kmeans.labels_
# Extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

# Extract x and y values of data points
x = df_clus["Carbon Emission"]
y = df_clus["Access to Electricity"]
plt.figure(figsize=(12, 8))
# Plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o")
# Show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Carbon Emission (Kt)")
plt.ylabel("Access to Electricity (%)")
plt.show()

param, covar = opt.curve_fit(logistic, df.index, df['Carbon Emission'])

# Create array for forecasting
year = np.linspace(1990, 2040, 100)
forecast = logistic(year, 3.4e7, 0.10, 1985)

plt.figure()
plt.plot(df.index, df['Carbon Emission'], label="Orignal")
plt.plot(year, forecast, label="Forecast")
plt.xlabel("Year")
plt.ylabel("Carbon Emission (Kt)")
plt.legend()
plt.show()
