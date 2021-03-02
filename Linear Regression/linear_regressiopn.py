import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston = load_boston()
# Transform the boston data set into data frame
df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

# Initializing linear regression
l_reg = linear_model.LinearRegression()

# 67% for learning and 33% for testing
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)

# Train the model (fit)
l_reg.fit(x_train, y_train)

# coefficient of each column (feature)
print(l_reg.coef_)

# prediction with test data
y_pred = l_reg.predict(x_test)

print("predicted price is : " + f"{y_pred[2]}")
print("actual price list :" + f"{y_test[0]}")
