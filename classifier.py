# IMPORTING PACKAGES

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import itertools 

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score 




df = pd.read_csv('/Users/manneyaajayasanker/Downloads/synthetic.csv')
df.drop('type', axis = 1, inplace = True)
df.drop('nameOrig', axis = 1, inplace = True)
df.drop('nameDest', axis = 1, inplace = True)
# DATA SPLIT

X = df.drop('isFraud', axis = 1).values
y = df['isFraud'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)



# MODELING

# Decision Tree

tree_model = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
tree_model.fit(X_train, y_train)
tree_yhat = tree_model.predict(X_test)


# RandomForest

rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)


# EVALUATION

# 1. Accuracy score

print('ACCURACY SCORE')
print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, tree_yhat)))
print('Accuracy score of the Random Forest Tree model is {}'.format(accuracy_score(y_test, rf_yhat)))
# 2. F1 score

print('F1 SCORE')
print('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, tree_yhat)))
print('F1 score of the Random Forest Tree model is {}'.format(f1_score(y_test, rf_yhat)))

# 3. Confusion Matrix

# defining the plot function

def plot_confusion_matrix(cma, classes, title, normalize = False, cmap = plt.cma.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cma= cma.astype(float) / cma.sum(axis=1)[:, np.newaxis]

    plt.imshow(cma, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cma.max() / 2.
    for i, j in itertools.product(range(cma.shape[0]), range(cma.shape[1])):
        plt.text(j, i, format(cma[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cma[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix for the models


tree_matrix = confusion_matrix(y_test, tree_yhat, labels = [0, 1]) # Decision Tree
randomforest_matrix = confusion_matrix(y_test, rf_yhat, labels = [0, 1]) # Random Forest Tree


# Plot the confusion matrix

plt.rcParams['figure.figsize'] = (6, 6)

# Decision tree
tree_cm_plot = plot_confusion_matrix(tree_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'Decision Tree')
plt.show()

# Random forest tree

random_forest_plot = plot_confusion_matrix(randomforest_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'Random Forest Tree')

plt.show()

