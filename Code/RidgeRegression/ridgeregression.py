# necessary imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#matplotlib inline
plt.style.use('fivethirtyeight')

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
#add target to df
df['HousePrice'] = data.target

X = df.drop(columns = "HousePrice",axis=1)
y = df["HousePrice"]

from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
print(X_scaled.shape)
# checking for multicollinearity using `VIF` and `correlation matrix`

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif['Features'] = X.columns


X_scaled = StandardScaler().fit_transform(X.drop(columns = ['Latitude'],axis =1))
print(X_scaled.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import Ridge

rr = Ridge(alpha=0.025)
rr.fit(X_train, y_train)

y_pred = rr.predict(X_test)

print(rr.score(X_train, y_train),"\n",
rr.score(X_test, y_test))

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# note: using standard scaler would somehow reduce the performance of the model, why?