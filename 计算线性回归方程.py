# Get the regression coefficients
coefficients = model.params
intercept = coefficients[0]
coef_1 = coefficients[1]
coef_2 = coefficients[2]

# Construct the regression equation
regression_equation = f"Death_Count = {intercept:.3f} + {coef_1:.3f} * PC1 + {coef_2:.3f} * PC2"
regression_equation
