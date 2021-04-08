# %%
import pandas as pd
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# %% Loading data
boston = load_boston()
X = pd.DataFrame(boston['data'])
y = pd.Series(boston['target'])
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# %% Standardizing data
scaler = StandardScaler()
# Xtrain_new = scaler.transform(Xtrain)
# Xtest_new = scaler.transform(Xtest)
Xtrain_new = scaler.fit_transform(Xtrain)
Xtest_new = scaler.transform(Xtest)
# %% sgd model
n_iter = 1000
model = SGDRegressor(max_iter=n_iter, alpha=0.0015)
model.fit(Xtrain_new, ytrain)
ytest_pred1 = model.predict(Xtest_new)
mse1 = mean_squared_error(ytest, ytest_pred1)
# %%
model2 = LinearRegression()
model2.fit(Xtrain_new, ytrain)
ytest_pred2 = model2.predict(Xtest_new)
mse2 = mean_squared_error(ytest, ytest_pred2)
print(mse1, mse2)
# %%
