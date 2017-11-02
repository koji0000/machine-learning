from xgboost import XGBClassifier
from numpy import loadtxt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

dataset = loadtxt('pima-indians-diabates.csv', delimiter=',')

X = dataset[:, 0:8]
Y = dataset[:, 8]

print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model = XGBClassifier()

eval_set = [(X_test, y_test)]

model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
print(y_pred)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print(accuracy * 100)