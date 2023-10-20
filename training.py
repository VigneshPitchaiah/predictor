from l1_reg import L1RegularizedLinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/Salary_Data.csv')
X = df['YearsExperience']
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = L1RegularizedLinearRegression()
model.fit(X_train, y_train)