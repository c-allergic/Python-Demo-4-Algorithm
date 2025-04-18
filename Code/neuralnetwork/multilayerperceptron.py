import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import load_model

# load the dataset
path = "/Users/sheldon/.cache/kagglehub/datasets/camnugent/california-housing-prices/versions/1"
housing = pd.read_csv(path + "/housing.csv")
target = housing['median_house_value']
housing = housing.drop(columns=['median_house_value'])

# data preprocessing
# feature engineering
housing['ocean_proximity'] = housing['ocean_proximity'].map({
    '<1H OCEAN': 0,
    'NEAR OCEAN': 1,
    'NEAR BAY': 2,
    'INLAND': 3,
    'ISLAND': 4
})
# na values
housing = housing.fillna(housing.mean())
# standardize the features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(housing)
scaler_y = RobustScaler()
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1)).ravel()
# split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target_scaled, test_size=0.2, random_state=42)

# create a multilayer perceptron
def create_model():
    model = Sequential(
        [
            Dense(256, input_dim=X_train.shape[1], activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ]
    )
    return model

# compile the model
model = create_model()
model.compile(
    optimizer= Adam(learning_rate=0.001), 
    loss='mean_squared_error',
    )

# train the model
print("Training start")
# callback
checkpoint = ModelCheckpoint(
    'best_model.keras',  # file name
    monitor='loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=80,
    callbacks=[checkpoint]  # add callback
)

# load the best model
model = load_model('best_model.keras')

# evaluate the model, using percentage error
y_pred = model.predict(X_test)
y_pred_unscaled = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel() 
percentage_error = np.mean(np.abs(y_pred_unscaled.flatten() - y_test_original) / y_test_original)
print(f"Percentage error: {percentage_error} %")



















