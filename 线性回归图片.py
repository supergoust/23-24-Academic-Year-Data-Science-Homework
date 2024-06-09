import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# Load the original Excel file（这里记得将文件地址改为本地保存的地址）
file_path = r'C:\Users\12864\Desktop\数据科学大作业/数据.xlsx'
data = pd.read_excel(file_path)

# Load the new Excel file with death counts
death_counts_path =r'C:\Users\12864\Desktop\数据科学大作业/死亡人数筛选后求和.xlsx'
death_counts_data = pd.read_excel(death_counts_path)

# Merge the datasets on 'year_month'
merged_data = pd.merge(data, death_counts_data, on='year_month')

# Check the size of the merged data
print("Size of merged data:", merged_data.shape)

# Define target variable
target = merged_data['1COVID-19death']
data_features = merged_data.drop(columns=['year_month', '1COVID-19death'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_features)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)

# Select the number of components to use (5 components)
num_components = 2
X = principal_components[:, :num_components]

# Standardize the target variable (death counts)
scaler_target = StandardScaler()
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, scaled_target, test_size=0.3, random_state=42)

# Check the size of the training and testing sets
print("Size of training set:", X_train.shape)
print("Size of testing set:", X_test.shape)

# Add constant to the features for OLS regression
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit the model using OLS
model = sm.OLS(y_train, X_train_sm).fit()

# Make predictions
y_pred_sm = model.predict(X_test_sm)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_sm, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Death Count (Standardized)')
plt.ylabel('Predicted Death Count (Standardized)')
plt.title('Actual vs Predicted Death Count (Standardized)')
plt.show()

# Generate ANOVA table manually
ssr = np.sum((y_pred_sm - y_test.mean())**2)
sse = np.sum((y_test - y_pred_sm)**2)
sst = np.sum((y_test - y_test.mean())**2)
df_model = X_train_sm.shape[1] - 1
df_total = len(y_test) - 1
df_resid = df_total - df_model
msr = ssr / df_model
mse = sse / df_resid
f_stat = msr / mse
p_value = stats.f.sf(f_stat, df_model, df_resid)

anova_table = pd.DataFrame({
    'Source': ['Model', 'Residual', 'Total'],
    'Sum of Squares': [ssr, sse, sst],
    'Degrees of Freedom': [df_model, df_resid, df_total],
    'Mean Square': [msr, mse, ''],
    'F-Statistic': [f_stat, '', ''],
    'P-Value': [p_value, '', '']
})

anova_table
