from pandas import DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

# remove warnings
pd.set_option('mode.chained_assignment', None)


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# drop last column with classes and divide into 3 clusters

# import data
df = pd.read_csv('iris.data.csv')

# print original table
print('\n-----------------------------------')
print('Original: \n')
print(df)
print('\n-----------------------------------')

# drop class column
df.drop(['class'], axis='columns', inplace=True)
df.head()

# print table without class column
print('\n-----------------------------------')
print('Table without original class column: \n')
print(df)
print('\n-----------------------------------')

# get all the columns
iris = list(df.columns)

# get the iris data
data = df[iris]

# perform clustering here
clustering_kmeans = KMeans(n_clusters=3, precompute_distances="auto", n_jobs=-1)
data['clusters'] = clustering_kmeans.fit_predict(data)
print("PREDICTED CLUSTERS \n", data.to_string())

pca_num_components = 2

# Well, you cannot do it directly if you have more than 3 columns.
# However, you can apply a Principal Component Analysis to reduce
# the space in 2 columns and visualize this instead.

# run PCA (Principal Component Analysis) on the data and reduce dimensions in pca_num_components dimensions
reduced_data = PCA(n_components=pca_num_components).fit_transform(data)
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=data['clusters'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()
