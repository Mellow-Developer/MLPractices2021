import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix


# Loading the dataset
dataset = pd.read_csv('iris.csv')
dataset.describe()

# splitting the dataset into the training set and test set
X = dataset.iloc[:, [0, 1, 2, 3]].values
Y = dataset.iloc[:, 4].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting LR to Training set
C = [10, 1, 0.1, 0.001]
for c in C:
    classifier = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', C=c,  multi_class='auto')
    classifier.fit(X_train, Y_train)


# Predicting Test set
Y_pred = classifier.predict(X_test)
# Predicting probabilities
probs_y = classifier.predict_proba(X_test)

# result
probs_y = np.round(probs_y, 2)

# confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# Plot confusion matrix
# confusion matrix sns heatmap
ax = plt.axes()
df_cm = cm
sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = ax )
ax.set_title('Confusion Matrix')
plt.show()

