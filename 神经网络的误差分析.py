import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data_file_1 = r'C:\Users\12864\Desktop\数据科学大作业/数据.xlsx'
data_file_2 = r'C:\Users\12864\Desktop\数据科学大作业/死亡人数筛选后求和.xlsx'

data_1 = pd.read_excel(data_file_1)
data_2 = pd.read_excel(data_file_2)

merged_data = pd.merge(data_1, data_2, on='year_month')
merged_data = merged_data.drop(columns=['year_month'])

missing_values = merged_data.isnull().sum()

if missing_values.sum() > 0:
    merged_data = merged_data.fillna(merged_data.mean())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data)

X = scaled_data[:, :-1]
y = scaled_data[:, -1]

# Split the data into training, validation, and testing sets (each 33%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

mlp_improved = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_improved.fit(X_train, y_train)

y_train_pred = mlp_improved.predict(X_train)
y_val_pred = mlp_improved.predict(X_val)
y_test_pred = mlp_improved.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Display results
results = pd.DataFrame({
    'Set': ['Training', 'Validation', 'Testing'],
    'Samples': [len(y_train), len(y_val), len(y_test)],
    'MSE': [mse_train, mse_val, mse_test],
    'R²': [r2_train, r2_val, r2_test]
})

print(results)

# Plotting error histogram
train_errors = y_train - y_train_pred
val_errors = y_val - y_val_pred
test_errors = y_test - y_test_pred

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.hist(train_errors, bins=20, color='blue', edgecolor='black')
plt.title('Training Set Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(val_errors, bins=20, color='orange', edgecolor='black')
plt.title('Validation Set Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(test_errors, bins=20, color='green', edgecolor='black')
plt.title('Testing Set Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()