# EX-1 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Designing and implementing a neural network regression model aims to accurately predict a continuous target variable based on a set of input features from the provided dataset. The neural network learns complex relationships within the data through interconnected layers of neurons. The model architecture includes an input layer for the features, several hidden layers with non-linear activation functions like ReLU to capture complex patterns, and an output layer with a linear activation function to produce the continuous target prediction. 
## Neural Network Model
![Screenshot 2024-08-19 200135](https://github.com/user-attachments/assets/ff77dfe9-6f72-4de0-9e0d-0e0fe6e8bf51)

## DESIGN STEPS
### STEP 1:
Loading the dataset
### STEP 2:
Split the dataset into training and testing
### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4:
Build the Neural Network Model and compile the model.
### STEP 5:
Train the model with the training data.
### STEP 6:
Plot the performance plot
### STEP 7:
Evaluate the model with the testing data.
## PROGRAM
```
DEVELOPED BY : VASUNDRA SRI R
REGISTER NUMBER : 212222230168
```

## Importing Required packages
```py
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
```

## Authenticate the Google sheet
```py
auth.authenticate_user()
creds, _ = default()
gc=gspread.authorize(creds)
worksheet = gc.open('data').sheet1
data=worksheet.get_all_values()
```
## Construct Data frame using Rows and columns
```py
dataset1=pd.DataFrame(data[1:], columns=data[0])
dataset1=dataset1.astype({'x':'float'})
dataset1=dataset1.astype({'y':'float'})
dataset1
x=dataset1[['x']].values
y=dataset1[['y']].values
```
## Split the testing and training data
```py
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
x_train1=Scaler.transform(x_train)
```

## Build the Deep learning Model
```py
ai_brain = Sequential([
    Dense(8,activation = 'relu',input_shape=[1]),
    Dense(10, activation = 'relu'),
    Dense(1)

])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs=1997)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```

## Evaluate the Model
```py
x_test1=Scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)

x_n1=[[18]]
x_n1_1=Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```
## Dataset Information
![Screenshot 2024-08-19 121809](https://github.com/user-attachments/assets/d2217db0-ad26-4fc4-bb18-b7f9c879a355)

## OUTPUT
## Training Loss Vs Iteration Plot
![Screenshot 2024-08-19 195909](https://github.com/user-attachments/assets/5eb98867-ec67-4587-93c7-62c4ee2c0356)

## Test Data Root Mean Squared Error
![Screenshot 2024-08-19 195856](https://github.com/user-attachments/assets/fd2b0842-dadb-480e-befc-c551c708cd98)

## New Sample Data Prediction
![Screenshot 2024-08-19 195919](https://github.com/user-attachments/assets/391b8cca-e8d6-4eb9-9377-62e255b4b91d)
![Screenshot 2024-08-19 195930](https://github.com/user-attachments/assets/6cb318e9-c1e5-4c37-9cfc-3240db8ee2bb)

## RESULT
Thus a Neural network for Regression model is Implemented
