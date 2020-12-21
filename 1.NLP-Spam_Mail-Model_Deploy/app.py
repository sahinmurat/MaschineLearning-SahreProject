from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
import joblib

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize a new Flask instance that can find HTML Folder in templete directory
app = Flask(__name__)


# Route decorator specify the URL that should trigger the execution of the "home" function that rendered the "home.html" file
@app.route("/")
def home():
    return render_template("home.html")


# Make the model and let user enter his message to predict.
# "POST" method to transport the form data to the server in the message body.
@app.route("/predict", methods=["POST"])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    df.columns = ["Label", "SMS"]
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

    X = df["SMS"].values
    y = df["Label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    cv = CountVectorizer(max_features=3700)  # feature that accure more than one time

    X_train = cv.fit_transform(X_train).toarray()
    X_test = cv.transform(X_test).toarray()

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    if request.method == "POST":
        message = request.form["message"]
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template("result.html", prediction=my_prediction)


# "run" function to only run the application on the server
# when this script is directly executed by the Python interpreter,
# which we ensured using the if statement with __name__ == '__main__'.
if __name__ == "__main__":
    app.run(debug=True)  # "debug=True" to activate Flask's debugger
