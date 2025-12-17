from flask import Flask, jsonify
import main

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Daily Brief is running."

@app.route("/run")
def run_pipeline():
    result = main.run_pipeline()
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
