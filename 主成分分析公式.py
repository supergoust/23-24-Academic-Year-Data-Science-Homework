# Load the Excel file
file_path = r'C:\Users\12864\Desktop\数据科学大作业/数据.xlsx'
data = pd.read_excel(file_path)

# Remove the 'year_month' column and standardize the data
data_features = data.drop(columns=['year_month'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_features)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

# Get the principal components
components = pca.components_

# Create formulas for each principal component
formulas = []
for i, component in enumerate(components):
    formula = f"F_{i+1} = " + " + ".join([f"{round(value, 3)}*X_{j+1}" for j, value in enumerate(component)])
    formulas.append(formula)

formulas

