from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("student_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    # Setup lists for dropdowns
    student_numbers = list(range(1, 71))
    roll_numbers = list(range(1, 71))
    classes = ["10A", "10B", "10C", "10D", "10E"]  # Add/remove as needed
    study_hours = list(range(1, 13))   # 1 to 12 hours
    attendance_options = list(range(75, 81))   # 75 to 80

    result = ""
    if request.method == "POST":
        student_number = request.form["student_number"]
        student_name = request.form["student_name"]
        roll_number = request.form["roll_number"]
        student_class = request.form["student_class"]
        hours = int(request.form["hours"])
        attendance = int(request.form["attendance"])

        data = np.array([[hours, attendance]])
        prediction = model.predict(data)[0]

        result = (
            f"Student No.: {student_number}<br>"
            f"Name: {student_name}<br>"
            f"Roll No.: {roll_number}<br>"
            f"Class: {student_class}<br>"
            f"Predicted Score: {prediction:.2f}"
        )

    return render_template(
        "index.html",
        student_numbers=student_numbers,
        roll_numbers=roll_numbers,
        classes=classes,
        study_hours=study_hours,
        attendance_options=attendance_options,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)
