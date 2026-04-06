from flask import Flask, render_template, request
from chatbot import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", query=None, response=None)

@app.route("/chat", methods=["POST"])
def chat():
    query = request.form.get("query")
    response = get_response(query)
    return render_template("index.html", query=query, response=response)

if __name__ == "__main__":
    app.run(debug=True)