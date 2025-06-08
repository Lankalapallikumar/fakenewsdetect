from flask import Flask, render_template, request, session
import pickle

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    # Initialize counts in session if not present
    if 'fake_count' not in session:
        session['fake_count'] = 0
    if 'real_count' not in session:
        session['real_count'] = 0

    total = session['fake_count'] + session['real_count']
    if total > 0:
        fake_percent = round((session['fake_count'] / total) * 100, 2)
        real_percent = round((session['real_count'] / total) * 100, 2)
    else:
        fake_percent = real_percent = 0

    return render_template("index.html", 
                           fake_percent=fake_percent, 
                           real_percent=real_percent,
                           total=total)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        title = request.form["title"]
        text = request.form["text"]
        combined_text = title + " " + text

        vectorized_text = vectorizer.transform([combined_text])
        prediction = model.predict(vectorized_text)[0]

        # Update session counts
        if 'fake_count' not in session:
            session['fake_count'] = 0
        if 'real_count' not in session:
            session['real_count'] = 0

        if prediction == "FAKE":
            session['fake_count'] += 1
        else:
            session['real_count'] += 1

        session.modified = True  # To save session changes

        total = session['fake_count'] + session['real_count']
        fake_percent = round((session['fake_count'] / total) * 100, 2)
        real_percent = round((session['real_count'] / total) * 100, 2)

        return render_template("index.html", 
                               prediction=prediction, 
                               fake_percent=fake_percent, 
                               real_percent=real_percent,
                               total=total)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
