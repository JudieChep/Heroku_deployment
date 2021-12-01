import pandas as pd
import numpy as np
import pickle

dataset=pd.read_csv('https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv')

#print(dataset.isnull().sum())
#Train the model
x = dataset[['temperature', 'humidity', 'windspeed']]
y = dataset['count']


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('bike_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('bike_model.pkl','rb'))
print(model.predict([[13, 42, 33]]))