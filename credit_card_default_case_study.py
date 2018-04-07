
"""
DATABASE LAYER

Loading Excel to the Database in sqlite3

%% REMOVED THE FIRST ROW OF THE EXCEL,
PLEASE RUN THE PROGRAM BY REMOVING THE FIRST ROW OF THE EXCEL
I HAVE UPLOADED THE SAME EXCEL I USED ALONG WITH THE PROGRAM %%


Excel File :default_of_credit_card_clients.xlsx
Database Name : CREDIT_DEFAULT
Table : DATA
Primary Key : GENERATED_PRIMARY_KEY 

"""

import sqlite3
import openpyxl
from openpyxl import load_workbook
import re
import pandas as pd

def slugify(text, lower=1):
    if lower == 1:
        text = text.strip().lower()
    text = re.sub(r'[^\w _-]+', '', text)
    text = re.sub(r'[- ]+', '_', text)
    return text


#Replace with a database name
con = sqlite3.connect('CREDIT_DEFAULT.db')
#replace with the complete path to youe excel workbook
wb = load_workbook(filename=r'default_of_credit_card_clients.xlsx')

sheets = wb.get_sheet_names()

for sheet in sheets:
    ws = wb[sheet] 

    columns= []
    query = 'CREATE TABLE ' + str(slugify(sheet)) + '(GENERATED_PRIMARY_KEY INTEGER PRIMARY KEY AUTOINCREMENT'
    for row in next(ws.rows):
        query += ', ' + slugify(row.value) + ' TEXT'
        columns.append(slugify(row.value))
    query += ');'

    con.execute(query)

    tup = []
    for i, rows in enumerate(ws):
        tuprow = []
        if i == 0:
            continue
        for row in rows:
            tuprow.append(str(row.value).strip()) if str(row.value).strip() != 'None' else tuprow.append('')
        tup.append(tuple(tuprow))


    insQuery1 = 'INSERT INTO ' + str(slugify(sheet)) + '('
    insQuery2 = ''
    for col in columns:
        insQuery1 += col + ', '
        insQuery2 += '?, '
    insQuery1 = insQuery1[:-2] + ') VALUES('
    insQuery2 = insQuery2[:-2] + ')'
    insQuery = insQuery1 + insQuery2

    con.executemany(insQuery, tup)
    con.commit()

con.close()



"""
ALGORITHM LAYER

Creating the Algorithm in machine learning to evaluvate the data

%% REMOVED THE FIRST ROW OF THE EXCEL,
PLEASE RUN THE PROGRAM BY REMOVING THE FIRST ROW OF THE EXCEL 
I HAVE UPLOADED THE SAME EXCEL I USED ALONG WITH THE PROGRAM %%


"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


# Importing the dataset
dataset = pd.read_excel('default_of_credit_card_clients.xlsx')
X = dataset.iloc[:,0:24].values
y = dataset.iloc[:, 24].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# predict the test result
y_pred = classifier.predict(X_test)


# making the confusioin matrix

cm = confusion_matrix(y_test , y_pred)

Accuracy = (accuracy_score(y_test,y_pred))


"""
VISUALIZATION LAYER
VISUALISING OF THE DATA 
AND  DATA DERIVED IN THE PRINCIPAL COMPONENT ANALYSIS (PCA)

"""


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('KNN (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


Education_Factor = pd.crosstab(index=dataset['EDUCATION'],columns=dataset['default payment next month'])

Gender_Factor = pd.crosstab(index=dataset['SEX'],columns=dataset['default payment next month'])

Marriage_Factor = pd.crosstab(index=dataset['MARRIAGE'],columns=dataset['default payment next month'])

AGE_Factor = pd.crosstab(index=dataset['AGE'],columns=dataset['default payment next month'])

sns.countplot(x="EDUCATION",data=dataset,hue="default payment next month")

sns.jointplot(x="EDUCATION",data=dataset,y="default payment next month",kind="kde")

sns.countplot(x="MARRIAGE",data=dataset,hue="default payment next month")

sns.countplot(x="AGE",data=dataset,hue="default payment next month")

sns.countplot(x="SEX",data=dataset,hue="default payment next month")

sns.pairplot(dataset);




