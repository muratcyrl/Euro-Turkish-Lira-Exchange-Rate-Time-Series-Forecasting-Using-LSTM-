import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def loadData_windowsliding(data, window_size, step_size=1):
    X, y = [], []    
    for i in range(0, len(data) - window_size):
        window = data[i:i + window_size]
        next_value = data[i + window_size]  # only the next single step
        X.append(window)
        y.append(next_value)
    return np.array(X), np.array(y)

df1 = pd.read_csv("EUR-TUR.csv")
df1["Tarih"] = pd.to_datetime(df1["Tarih"], dayfirst=True, errors="coerce")
df1 = df1.sort_values(by="Tarih", ascending=True)
df1 = df1.drop('Tarih', axis=1)
num_cols = ["Şimdi", "Açılış", "Yüksek", "Düşük"]
for col in num_cols:
    df1[col] = df1[col].str.replace('.', '', regex=False)  
    df1[col] = df1[col].str.replace(',', '.', regex=False) 
    df1[col] = df1[col].astype(float)
df1[num_cols] = df1[num_cols].astype(float)

window_size = 30
step_size = 1
num_rows = 30
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(df1)
df1=scaler.transform(df1)

X, y = loadData_windowsliding(df1, window_size, step_size)

n_total = len(X)
train_start_pct = 0.3  
train_middle_pct = 0.3 
train_end_pct = 0.1    
n_start = int(train_start_pct * n_total)
n_middle = int(train_middle_pct * n_total)
n_end = int(train_end_pct * n_total)
start_indices = np.arange(0, n_start)
middle_indices = np.arange(n_total//2 - n_middle//2, n_total//2 + n_middle//2)
end_indices = np.arange(n_total - n_end, n_total)
train_indices = np.concatenate([start_indices, middle_indices, end_indices])
all_indices = np.arange(n_total)
remaining_indices = np.setdiff1d(all_indices, train_indices)
n_remaining = len(remaining_indices)
n_val = n_remaining // 2
val_indices = remaining_indices[:n_val]
test_indices = remaining_indices[n_val:]

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)
features=X_train.shape[2]
print(X_train.shape,X_val.shape,X_test.shape)
print(X_train[-1,:,0])

early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

model = Sequential([
    LSTM(128, input_shape=(window_size, X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(X_train.shape[2])])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=40, batch_size=32,
                        validation_data=(
                            X_val, y_val), verbose=1, shuffle=False)
predictions = []

for i in range(0,30):
    step_pred = model.predict(X_test[i:i+1,:,:]) 
    print(step_pred.shape)
    predictions.append(step_pred[0]) 
current_window  = X_test[-1]
predictions = np.array(predictions)  # convert to numpy array at the end
X_test_scaled=scaler.inverse_transform(X_test.reshape(-1,features))
predictions=scaler.inverse_transform(predictions)
X_test=X_test_scaled.reshape(X_test.shape[0],window_size,features)


real_window1 = X_test[0, :, 0]       
real_window1=np.append(real_window1,predictions[0,0])
real_window2 = X_test[30, :, 0]             

plt.figure(figsize=(10, 5))

plt.plot(range(window_size+1), real_window1, label="Real Window 1", color='blue')
plt.plot(range(window_size, 2*window_size), real_window2, '--', label="Real Window 2", color='green')
plt.plot(range(window_size, 2*window_size), predictions[:,0], 'ro-', label="Predicted Window 2", markersize=4, linewidth=2)

plt.xlabel("Timestep")
plt.ylabel("Feature 1 Value")
plt.title("Predicted vs Real Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
real_predictions=[]
current_window=df1[-30:]
for i in range(30):  
    pred_scaled = model.predict(current_window[np.newaxis, :, :]) 
    real_predictions.append(pred_scaled[0])
    current_window = np.roll(current_window, -1, axis=0)
    current_window[-1, :] = pred_scaled 

real_predictions = np.array(real_predictions)
real_predictions = scaler.inverse_transform(real_predictions)
df1=scaler.inverse_transform(df1)
#Plot the Next 30 Days
plt.figure(figsize=(10,5))
plt.plot(df1[-500:,0],color='blue', label="Actual", markersize=4)
plt.plot(range(500,530),real_predictions[:,0], color="red", label="Predicted Feature 1", markersize=4)
plt.xlabel("Timestep")
plt.ylabel("Feature 1 Value")
plt.title("Predicted Next 30 Days")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

