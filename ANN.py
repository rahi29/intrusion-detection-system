import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

#################################################################### Read data ##########################################################################


data = pd.read_csv('./Data/filtered_data.csv')



###################################################### Convert 'class' column to binary numeric values ####################################################


data['class'] = (data['class'] != "normal").astype(int)


####################################################### Split features and target #########################################################################


X = data.drop(columns=['class'])
y = data['class']


############################################################## Split data into train and test sets ########################################################


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


##################################################################### Standardize the data ##################################################################


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

######################################################################## Build the model ###################################################################


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

######################################################################## Compile the model ###################################################################


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


########################################################################## Train the model ###################################################################


model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)


##################################################################### Evaluate the model on test data ########################################################


loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Accuracy:", accuracy)
