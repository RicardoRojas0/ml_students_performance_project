from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


# Route for home page
@app.route("/")
def index():
    return render_template("index.html")


# Route for prediction
@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        
        # Get the data from the form and create a CustomData object
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            writing_score=float(request.form.get("writing_score")),
            reading_score=float(request.form.get("reading_score")),
        )

        # Get the data as a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data=pred_df)

        return render_template("home.html", results=results[0])
    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
