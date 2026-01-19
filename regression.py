import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
data = pd.read_csv("data/cgpa_package_data.csv")

# Select variables
X = data[["cgpa"]]
Y = data["Package"]

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Results
print("R2 Score:", r2_score(y_test, y_pred))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot
sns.scatterplot(x="cgpa", y="Package", data=data)
plt.plot(X, model.predict(X))
plt.title("CGPA vs Package Regression")
plt.show()
