import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r'C:\Users\12864\Desktop\数据科学大作业/数据.xlsx'
data = pd.read_excel(file_path)

# Remove the 'year_month' column and standardize the data
data_features = data.drop(columns=['year_month'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_features)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)
principal_df = pd.DataFrame(data=principal_components, columns=[f'Principal Component {i+1}' for i in range(principal_components.shape[1])])

# Combine with the 'year_month' column for visualization purposes
final_df = pd.concat([principal_df, data[['year_month']]], axis=1)

# Plot the PCA result for the first three components
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_df['Principal Component 1'], final_df['Principal Component 2'], final_df['Principal Component 3'])
for i, txt in enumerate(final_df['year_month']):
    ax.text(final_df['Principal Component 1'][i], final_df['Principal Component 2'][i], final_df['Principal Component 3'][i], txt, size=8)
ax.set_title('PCA of COVID-19 Behavioral Data (First Three Components)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.grid(False)  # Remove grid lines
plt.show()

# Get the explained variance ratio (contribution rate)
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_df = pd.DataFrame({
    'Principal Component': [f'Principal Component {i+1}' for i in range(len(explained_variance_ratio))],
    'Explained Variance Ratio': explained_variance_ratio
})

# Get the eigenvalues
eigenvalues = pca.explained_variance_
eigenvalues_df = pd.DataFrame({
    'Principal Component': [f'Principal Component {i+1}' for i in range(len(eigenvalues))],
    'Eigenvalue': eigenvalues
})

# Get the eigenvectors (feature vectors)
eigenvectors = pca.components_
eigenvectors_df = pd.DataFrame(data=eigenvectors, columns=data_features.columns)
eigenvectors_df.index = [f'Principal Component {i+1}' for i in range(eigenvectors.shape[0])]

# Display the dataframes
explained_variance_df, eigenvalues_df, eigenvectors_df

# Plot explained variance ratio with rotated x-axis labels
plt.figure(figsize=(10, 6))
plt.bar(explained_variance_df['Principal Component'], explained_variance_df['Explained Variance Ratio'], color='skyblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio of Principal Components')
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.grid(False)  # Remove grid lines
plt.show()

# Plot eigenvalues with rotated x-axis labels
plt.figure(figsize=(10, 6))
plt.bar(eigenvalues_df['Principal Component'], eigenvalues_df['Eigenvalue'], color='salmon')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Principal Components')
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.grid(False)  # Remove grid lines
plt.show()

# Plot eigenvectors for the first three principal components with more space between subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 18))
components_to_plot = ['Principal Component 1', 'Principal Component 2', 'Principal Component 3']

for i, component in enumerate(components_to_plot):
    axes[i].bar(eigenvectors_df.columns, eigenvectors_df.loc[component], color='lightgreen')
    axes[i].set_title(f'Eigenvectors of {component}')
    axes[i].set_xlabel('Features')
    axes[i].set_ylabel('Component Loading')
    axes[i].tick_params(axis='x', rotation=90)
    axes[i].grid(False)  # Remove grid lines

plt.subplots_adjust(hspace=0.5)  # Add space between subplots
plt.show()
