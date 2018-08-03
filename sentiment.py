import pandas as pd

# read data
vix = pd.read_csv("^VIX.csv")
tnx = pd.read_csv("^TNX.csv")
aaii = pd.read_csv("AAII-AAII_SENTIMENT.csv")
vix = pd.read_csv("^VIX.csv")
tnx = pd.read_csv("^TNX.csv")

# merge data
aaii = pd.merge(aaii, vix, on="Date")
aaii = pd.merge(aaii, tnx, on="Date")

# drop irrelevant
aaii = aaii.drop(columns=["S&P 500 Weekly High"], axis=1)
aaii = aaii.drop(columns=["S&P 500 Weekly Low"], axis=1)

# Prepare data
x = aaii.drop(columns=["S&P 500 Weekly Close"], axis=1)
x['Date'] = pd.to_datetime(x['Date'], format="%m/%d/%Y")
x = x.sort_values('Date')
x = x.drop(columns=["Date"], axis=1)
x = x.rolling(5).mean()
aaii['Date'] = pd.to_datetime(aaii['Date'], format="%m/%d/%Y")
aaii = aaii.sort_values('Date')
x['Date'] = aaii['Date']
aaii = pd.merge(aaii, x, on="Date", how="left", suffixes=["", "_sm5"])
aaii = aaii.dropna()

# add trend column
trend = []
closes = aaii['S&P 500 Weekly Close']
closes = list(closes)
for i in range(len(closes)-1):
    if closes[i+1] > closes[i]:
        trend.append(1)
    else:
        trend.append(0)
trend.append(1)
aaii.insert(loc=8, column="Trend", value=trend)

# training and testing data
x = aaii.drop(columns=["S&P 500 Weekly Close"], axis=1)
x = x.drop(columns=["Date"], axis=1)
x = x.drop(columns=["Trend"], axis=1)
y = aaii["Trend"]

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
model.coef_
#aaii.to_excel("AAII.xlsx", index=False)

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
model.coef_
model.intercept_

def calc(feature):
    return 1.17387091e+00 * feature[0] + -1.28896617e+00 * feature[1] + 1.13146579e-01 * feature[2] + 1.06072433e+00 * feature[3] + -1.95666063e-05 * feature[4] + 7.47401113e-01 * feature[5] + -9.32243309e-02 * feature[6] + -1.01756564e+00 * feature[7] + 1.10965424e+00 * feature[8] + -1.20287857e+00 * feature[9] + 2.32989827e-02 * feature[10] + -8.43499847e-01 * feature[11] + 0.16662908
calc(X_train.iloc[0])


