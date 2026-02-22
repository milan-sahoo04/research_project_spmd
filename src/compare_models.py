# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor

# df = pd.read_csv("data/processed/clean_air_data.csv")

# X = df.drop("AQI",axis=1)
# y = df["AQI"]

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# models = {
#     "RandomForest": RandomForestRegressor(),
#     "LinearRegression": LinearRegression(),
#     "DecisionTree": DecisionTreeRegressor()
# }

# for name,model in models.items():

#     model.fit(X_train,y_train)
#     pred = model.predict(X_test)

#     rmse = mean_squared_error(y_test,pred,squared=False)

#     print(name,"RMSE:",rmse)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("data/processed/clean_air_data.csv")

X = df.drop("AQI",axis=1)
y = df["AQI"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
    "RandomForest": RandomForestRegressor(),
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor()
}

for name,model in models.items():

    model.fit(X_train,y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test,pred))

    print(name,"RMSE:",rmse)