from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    description = ""
    emoji = ""
    if request.method == "POST":
        user_input = request.form["text"]
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        result = model.predict(transformed)[0]
        # Optional: model confidence
        try:
            proba = model.predict_proba(transformed)[0]
            confidence = round(max(proba) * 100, 2)
        except:
            confidence = None

        if result == 0:
            prediction = "Introvert 😌"
            emoji = "😌"
            description = "You prefer calm, minimally stimulating environments and enjoy deep thinking and personal time."
        else:
            prediction = "Extrovert 😄"
            emoji = "😄"
            description = "You're energized by social interaction and enjoy being around people, events, and action."

    return render_template("index.html", prediction=prediction, description=description, emoji=emoji)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    app.run(debug=True)
