import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Simple sample dataset
data = {
    "hours": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [50,55,60,65,70,75,80,85,90,95],
    "score": [40,45,50,55,60,65,70,75,80,85]
}

df = pd.DataFrame(data)

X = df[["hours", "attendance"]]
y = df["score"]

model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "student_model.pkl")

print("Model trained and saved!")
