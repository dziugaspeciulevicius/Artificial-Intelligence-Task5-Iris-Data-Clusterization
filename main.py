from pandas import DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import warnings

# remove warnings
pd.set_option('mode.chained_assignment', None)


def warn(*args, **kwargs):
    pass


warnings.warn = warn

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
data['CLUSTERS'] = clustering_kmeans.fit_predict(data)
print("PREDICTED CLUSTERS \n", data.to_string())

inputs = data.drop('CLUSTERS', axis='columns')
target = data.CLUSTERS

dummies_sepal_length_cm = pd.get_dummies(inputs.sepal_length_cm)
dummies_sepal_length_cm.head(3)
dummies_sepal_width_cm = pd.get_dummies(inputs.sepal_width_cm)
dummies_sepal_width_cm.head(3)

inputs = pd.concat([dummies_sepal_width_cm, dummies_sepal_length_cm, inputs], axis='columns')
# inputs.head
print(inputs)

# run PCA (Principal Component Analysis) on the data and reduce dimensions in pca_num_components dimensions
pca_num_components = 4
reduced_data = PCA(n_components=pca_num_components).fit_transform(data)
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2', 'pca3', 'pca4'])

# visualize data
sns.scatterplot(x="pca1", y="pca2", hue=data['CLUSTERS'], data=results)
# plt.title('K-means Clustering with 2 dimensions')
plt.show()

# now we use sklearn strain test split method to split our data into training and test sample
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.20)  # train/test samples (90/10)

print("\n-----------------------")
print("SPLIT DATA: \n")
print("X_train: ")
print(len(X_train))
print("Y_train: ")
print(len(y_train))
print("X_test: ")
print(len(X_test))
print("Y_test: ")
print(len(y_test))
print("Inputs: ")
print(len(data))
print("-----------------------\n")

print("\n-----------------------")
print("X_train data: \n")
print(X_train)
print("-----------------------\n")
print("\n-----------------------")
print("y_train data: \n")
print(y_train)
print("-----------------------\n")

# Now we can use NB model
model = GaussianNB()

# we use fit method whenever we want to train the model
model.fit(X_train, y_train)  # when we execute this it will train it

print("\n-----------------------")
print('X_test[:10]:')
print(X_test[:10])

print('\ny_test[:10]:')
print(y_test[:10])

print('\npredict: ')
print(model.predict(X_test[:10]))
print("-----------------------\n")

# after training first thing we can do is measure the score to find the accuracy
print("\n-----------------------")
print("ACCURACY:")
print(model.score(X_test, y_test))
print("-----------------------\n")

# ----------------------------
# clustering_kmeans = KMeans(n_clusters=3).fit(df)
# centroids = clustering_kmeans.cluster_centers_
# print("CENTROIDS: \n", centroids)
#
# plt.scatter(df['sepal_length_cm'], df['sepal_width_cm'], c= clustering_kmeans.labels_.astype(float), s=50, alpha=0.5)
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
# plt.show()
# ----------------------------


# ask user if he wants to save results into a new file
def ask_user(question):
    check = str(input("\n Do you want to save/overwrite updated info into a file? (Y/N): ")).lower().strip()
    try:
        if check[0] == 'y':
            return True
        elif check[0] == 'n':
            return False
        else:
            print('Invalid Input')
            return ask_user()
    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return ask_user()


# save results into a file
if ask_user("\n Do you want to save/overwrite updated info into a file?"):
    # write updated data into a file
    data.to_csv('updated_data.csv', date_format='%s', index=False)
    # df.sort_values("Weight", ascending=True).to_csv('updated_data.csv', date_format='%s', index=False)
    print('\n\n\n *** Updated data has been saved into updated_data.csv *** \n\n\n')
else:
    print('\n\n\n *** Data has not been saved *** \n\n\n')