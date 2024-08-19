# EX-1 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

1) Developing a Neural Network Regression Model AIM To develop a neural network regression model for the given dataset. THEORY Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it.

2) Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly.

3) First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model


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
import gspread
import pandas as pd
from google.auth import default
import pandas as pd
```

## Authenticate the Google sheet
```py
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Data').sheet1
```
## Construct Data frame using Rows and columns
```py
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
X=df[['Input']].values
Y=df[['Output']].values
```
## Split the testing and training data
```py
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=40)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
X_train1 = Scaler.transform(x_train)
```

## Build the Deep learning Model
```py
ai_brain=Sequential([
    Dense(9,activation = 'relu',input_shape=[1]),
    Dense(16,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer='adam',loss='mse')
ai_brain.fit(X_train1,y_train.astype(np.float32),epochs=1999)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```

## Evaluate the Model
```py
test=Scaler.transform(x_test)
ai_brain.evaluate(test,y_test.astype(np.float32))
n1=[[18]]
n1_1=Scaler.transform(n1)
ai_brain.predict(n1_1)
```
## Dataset Information
![Screenshot 2024-08-19 121809](https://github.com/user-attachments/assets/d2217db0-ad26-4fc4-bb18-b7f9c879a355)

## OUTPUT
## Training Loss Vs Iteration Plot
![Screenshot 2024-08-19 122057](https://github.com/user-attachments/assets/9a624b56-2e28-462d-97c8-6e566e2a3b99)

## Test Data Root Mean Squared Error
![Screenshot 2024-08-19 122025](https://github.com/user-attachments/assets/acb6b415-b3b7-46b2-b5b5-075894cc1655)

## New Sample Data Prediction
![Screenshot 2024-08-19 122127](https://github.com/user-attachments/assets/57b43d3f-0aab-49f0-9adf-864de31c6674)
![Screenshot 2024-08-19 122140](https://github.com/user-attachments/assets/71da3250-c86b-4543-b32d-aad6df0ad08c)


## RESULT
Thus a Neural network for Regression model is Implemented
