import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

## Загружаємо данні
train_df = pd.read_csv("C:/Users/Hvorost/Downloads/train.csv")
test_df = pd.read_csv("C:/Users/Hvorost/Downloads/test.csv")
combine = [train_df, test_df]

## Аналізуємо данні.
##Виведемо інформацію по наших данних.
print(train_df.head())
print(train_df.info())
print(test_df.info())

##Розглянемо зведення по числовим данним.
describe_fields = ["Age", "Fare", "Pclass", "SibSp", "Parch"]

print(train_df[train_df["Sex"] == "male"][describe_fields].describe())
print(test_df[test_df["Sex"] == "male"][describe_fields].describe())
print(train_df[train_df["Sex"] == "female"][describe_fields].describe())
print(test_df[test_df["Sex"] == "female"][describe_fields].describe())

## Статистика виживаємості в залежності від класа та статі.
print(train_df.groupby(['Pclass','Sex'])['Survived'].value_counts(normalize=True))

## Статистика виживаємості в залежності від родинних зв'язків.
print(train_df.groupby(['Parch','SibSp'])['Survived'].value_counts(normalize=True))

##Візуалізація виживших за віком.
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

##Візуалізація виживших за класом.
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.show()

##Видаляємо неінфомативні ознаки
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

##Удосконалимо існуючі ознаки.
##Виділення освовних іменних приставок і перетворення їх в числові ознаки.

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

##Тепер можемо видалити імена та ідентифікатор для тестової вибірки.
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

##Приведемо ознаку статі до числового виду
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

##Так як маємо багото пропусків в вікових данних спробуємо їх зпрогнозувати
guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

##Виділимо 5 вікових груп і подивимося як вони співвідносяться з вижившими.
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
##print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

##Закодуємо вік віковими групами, а потім видалимо.
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

##Створимо нові ознаки.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

##Використані ознаки можна видалити.
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

##Будуємо моделі різних алгоритмів

X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(logreg.score(X_train, y_train))

svc = SVC()
svc.fit(X_train, y_train)
print(svc.score(X_train, y_train))

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
print(gaussian.score(X_train, y_train))

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
print(linear_svc.score(X_train, y_train))

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
print(sgd.score(X_train, y_train))

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
print(decision_tree.score(X_train, y_train))

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
print(random_forest.score(X_train, y_train))
Y_pred = random_forest.predict(X_test)

##Найкращий результат показали RandomForestClassifier та DecisionTreeClassifier.
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('C:/Users/Hvorost/Downloads/submission.csv', index=False)

