import sys

# SciPy
import scipy

# NumPy
import numpy

# MatplotLib
import matplotlib

# Pandas
import pandas

# Scikit-Learn
import sklearn
   




# Load libraries
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)



# Summarise dataset
print("\n", dataset.shape) # Number of rows and columns
print("\n", dataset.head(15)) # Select first 15 rows to take a peek at the data
print("\n", dataset.describe()) # Summary statistics of data
print("\n", dataset.groupby('class').size()) # Get rows in each class




# Visualise the data
# box plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histogram
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()




# Evaluate algorithms
# Get test and validation set
array = dataset.values
X = array[:,0:4] # Select Sepal and Petal lengths and widths
Y = array[:,4] # Select class
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size = validation_size, random_state = seed)

# Test Harness
seed = 7
scoring = 'accuracy'

# Spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
kFoldsplits = 10
print("")

for name, model in models:
    kfold = model_selection.KFold(n_splits = kFoldsplits, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    print("{}:     {}:     ({})".format(name, format(cv_results.mean(), '.3f'), format(cv_results.std(), '.3f')))

# Compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#######################
########  Step 7 ########
#######################

# Making predictions
SVM = SVC()
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
acc_score = accuracy_score(Y_validation, predictions)

print("\n", "Accuracy of SVM model: {}".format(format(acc_score,'.3f')))
print("\n", "Confusion matrix: \n", confusion_matrix(Y_validation, predictions))
print("\n", "Classification report: \n", classification_report(Y_validation, predictions))