"""
Titanic Machine Learning
HTML: https://www.kaggle.com/c/titanic

"""

#Step 1: Import all the necessary packages we will need

import pandas as pd;
import numpy as np;
import seaborn as sbn;
import pylab as plot;
from matplotlib import pyplot as plt;
pd.set_option('display.max_rows', 100); #increasing display of pandas packages for ease of analysis
pd.set_option('display.max_columns', 100); #increasing display of pandas packages for ease of analysis

#Step 2: Import the data we will use to train our model, the file name is 'train.csv'

train_data = pd.read_csv('./data/train.csv');

#Step 3: See the size of the dataset we are importing

print(train_data.shape);

#Step 4: Get a list of all the columns in the dataset to understand the data we are working with

print(train_data.columns);

#Step 5: Use the pandas method 'describe' to get a high level description of the data

train_data.describe();


#Step 6: Look at completeness of data

total = train_data.isnull().sum().sort_values(ascending=False)
percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100
percent_2 = (round(percent_1,1)).sort_values(ascending=False)
missing_data = pd.concat([total,percent_1,percent_2], axis=1, keys = ['Total', '%-1', '%-2'])

# cabin has the most missing data, followed by age

#Step 7: Look at age and sex

survived = 'survived'
not_survived = 'not survived'

fig,axes = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
women = train_data[train_data['Sex'] == 'female']
men = train_data[train_data['Sex'] == 'male']
ax = sns.distplot(women[women['Survived'] == 1]['Age'].dropna(), bins=18, label = survived, ax = axes[0], kde = False)
ax = sns.distplot(women[women['Survived'] == 0]['Age'].dropna(), bins=18, label = not_survived, ax = axes[0], kde = False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived'] == 1]['Age'].dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived'] == 0]['Age'].dropna(), bins=18, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male')


#Step 8: Embarked, Pclass, and Sex

FacetGrid = sns.FacetGrid(train_data, row = 'Embarked', size = 4.5, aspect = 1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None, order=None, hue_order=None)
FacetGrid.add_legend()


#Step 9: Pclass

sns.barplot(x='Pclass', y = 'Survived', data=train_data)


grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()

#Step 10: Combine SibSp and Parch to a 'Alone/not Alone' and 'Relatives' attributes

train_data['relatives'] = train_data['SibSp'] + train_data['Parch']
train_data.loc[train_data['relatives'] > 0, 'not_alone'] = 0
train_data.loc[train_data['relatives'] == 0, 'not_alone'] = 1
train_data['not_alone'] = train_data['not_alone'].astype(int)

train_data['not_alone'].value_counts()



axes = sns.factorplot('relatives','Survived', data = train_data, aspect = 2.5)


#Step 11: extract deck number from Cabin

import re

deck = {"A" : 1, "B" : 2, "C":3,'D':4,'E':5,'F':6, 'G':7, 'U': 8}
train_data['Cabin'] = train_data['Cabin'].fillna("U0")
train_data['Deck'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
train_data['Deck'] = train_data['Deck'].map(deck)
train_data['Deck'] = train_data['Deck'].fillna(0)
train_data['Deck'] = train_data['Deck'].astype(int)

train_data = train_data.drop(['Cabin'],axis=1)


# Step 12: create an array that contains random numbers, computed based on the mean age value in regards to the standard deviation and is null

mean_age = train_data['Age'].mean()
std_age = train_data['Age'].std()
isnull_age = train_data['Age'].isnull().sum()

rand_age = np.random.randint(mean_age - std_age, mean_age + std_age, size = isnull_age)
age_slice = train_data['Age'].copy()
age_slice[np.isnan(age_slice)] = rand_age
train_data['Age'] = age_slice
train_data['Age'] = train_data['Age'].astype(int)


# Step 13: fill in missing 'Embarked' with most common value

train_data['Embarked'].describe()

common_value = 'S'

train_data['Embarked'] = train_data['Embarked'].fillna(common_value)


# Step 14: more data processing


train_data['Fare'] = train_data['Fare'].fillna(0)
train_data['Fare'] = train_data['Fare'].astype(int)


titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data['Title'] = train_data['Title'].map(titles)
train_data['Title'] = train_data['Title'].fillna(0)
train_data = train_data.drop(['Name'], axis = 1)

genders = {'male' :0, 'female':1}
train_data['Sex'] = train_data['Sex'].map(genders)

train_data = train_data.drop(['Ticket'], axis = 1)

ports = {'S':0, 'C':1, 'Q':2}
train_data['Embarked'] = train_data['Embarked'].map(ports)


# Step 15: Creating Categories

# Age Category

train_data['Age'] = train_data['Age'].astype(int)
train_data.loc[ train_data['Age'] <= 11, 'Age'] = 0
train_data.loc[(train_data['Age'] > 11) & (train_data['Age'] <= 18), 'Age'] = 1
train_data.loc[(train_data['Age'] > 18) & (train_data['Age'] <= 22), 'Age'] = 2
train_data.loc[(train_data['Age'] > 22) & (train_data['Age'] <= 27), 'Age'] = 3
train_data.loc[(train_data['Age'] > 27) & (train_data['Age'] <= 33), 'Age'] = 4
train_data.loc[(train_data['Age'] > 33) & (train_data['Age'] <= 40), 'Age'] = 5
train_data.loc[(train_data['Age'] > 40) & (train_data['Age'] <= 66), 'Age'] = 6
train_data.loc[ train_data['Age'] > 66, 'Age'] = 6

# Fare Categories

pd.qcut(train_data['Fare'], 6)

train_data.loc[ train_data['Fare'] <= 7.0, 'Fare'] = 0
train_data.loc[(train_data['Fare'] > 7) & (train_data['Fare'] <= 8), 'Fare'] = 1
train_data.loc[(train_data['Fare'] > 8) & (train_data['Fare'] <= 14), 'Fare']   = 2
train_data.loc[(train_data['Fare'] > 14) & (train_data['Fare'] <= 26), 'Fare']   = 3
train_data.loc[(train_data['Fare'] > 26) & (train_data['Fare'] <= 52), 'Fare']   = 4
train_data.loc[ train_data['Fare'] > 52, 'Fare'] = 5
train_data['Fare'] = train_data['Fare'].astype(int)

train_data.Fare.value_counts()

# Age * Class

train_data['Age_Class'] = train_data['Age']*train_data['Pclass']

# Fare per person

train_data['Fare_Per_Person'] = train_data['Fare']/(train_data['relatives']+1)
train_data['Fare_Per_Person'].astype(int)





####################
####TRAIN MODELS####
####################


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB



x = train_data.drop('Survived',axis = 1)
y = train_data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y)


# Stochastic Gradient Descent (SGD)

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(x_train, y_train)
sgd_pred = sgd.predict(x_test)

sgd.score(x_test, y_test)
acc_sgd = round(sgd.score(x_test, y_test)*100,2) 


# Random Forest:

random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(x_train, y_train)
rf_pred = random_forest.predict(x_test)

random_forest.score(x_test, y_test)
acc_random_forest = round(random_forest.score(x_test,y_test)*100,2)

# Logistic Regression:

logreg = LogisticRegression()
logreg.fit(x_train,y_train)

logreg_pred = logreg.predict(x_test)

acc_logreg = round(logreg.score(x_test, y_test)*100,2)

#K Nearest Neighbor

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

acc_knn = round(knn.score(x_train, y_train)*100, 2)

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)

gaussian_pred = gaussian.predict(x_test)

acc_gaussian = round(gaussian.score(x_test, y_test)*100,2)

# Perceptron

perceptron = Perceptron(max_iter = 5)
perceptron.fit(x_train,y_train)

perceptrion_pred = perceptron.predict(x_test)

acc_perceptron = round(perceptron.score(x_test, y_test)*100,2)

# Linear Support Vector Machine

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)

linear_svc_pred = linear_svc.predict(x_test)

acc_linear_svc = round(linear_svc.score(x_test, y_test)*100, 2)

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
decision_tree.pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x_train, y_train)*100,2)



# Which is the best model for our data?

results = pd.DataFrame({
	'Model' : ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Decision Tree'],
	'Score' : [acc_linear_svc, acc_knn, acc_logreg, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd, acc_decision_tree]
	})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df



# K_Fold Cross Validation

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators = 100)
scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = 'accuracy')

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# Feature Importance

importances = pd.DataFrame({'feature':x_train.columns, 'importance': np.round(random_forest.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')

importances

importances.plot.bar()


# Removing unnecessary features


x_test2 = x_test.drop(['Parch','not_alone'], axis=1)
x_train2 = x_train.drop(['Parch','not_alone'],axis=1)


# training random forest again

random_forest = RandomForestClassifier(n_estimators = 100, oob_score = True)
random_forest.fit(x_train2, y_train)
random_forest_2_pred = random_forest.predict(x_test2)

acc_random_forest = round(random_forest.score(x_test2, y_test)*100,2)
print(round(acc_random_forest,2),"%")
