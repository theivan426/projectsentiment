import pandas as pd

# read data
aaii = pd.read_excel("AAII.xlsx", sheet_name="Sheet1")

# training and testing data
x = aaii.drop(columns=["S&P 500 Weekly Close"], axis=1)
x = x.drop(columns=["Date", "Trend"], axis=1)
if "Predicted Trend" in x.columns:
	x = x.drop(columns=["Predicted Trend"], axis=1)
y = aaii["Trend"]
x = x[:-1]
y = y[:-1]
print(len(x))
# train ridge classifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44)
model = linear_model.RidgeClassifier(alpha=0.1)
model.fit(X_train, y_train)
for state in [22, 63, 93]:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=state)
    print("Accuracy for test set: {}".format(model.score(X_test, y_test)))
print("Accuracy for test set: {}".format(model.score(X_test, y_test)))
print("Accuracy for train set: {}".format(model.score(X_train, y_train)))

# run metrics testing
from sklearn.metrics import f1_score

for state in [22, 63, 93]:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=state)
    res = model.predict(X_test)
    print("F1 for test set: {}".format(f1_score(y_test, res)))

res = model.predict(X_test)
print("F1 for test set: {}".format(f1_score(y_test, res)))
res = model.predict(X_train)
print("F1 for train set: {}".format(f1_score(y_train, res)))

formula = "=IF("
for i, coef in enumerate(model.coef_[0]):
	formula += "{} * {}2+".format(coef, chr(ord('B')+i))
def calc(x):
	res = 0
	for i, coef in enumerate(model.coef_[0]):
		res += coef * x[i]
	res += model.intercept_[0]
	return res
formula += str(model.intercept_[0])
formula += " > 0, 1, 0)"
print(formula)
aaii["Predicted Trend"] = formula

print(model.decision_function(x.iloc[0:1]))
aaii.to_excel("AAII.xlsx", index=False)


	
 