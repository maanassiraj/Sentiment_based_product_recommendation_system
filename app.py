from flask import Flask, request, render_template
from model import *

app = Flask(__name__, template_folder="template")

@app.route("/")
def home():
    return render_template("index.html")
    

@app.route("/submit", methods=["POST"])
def predict():
    username = request.form.get("username")
    return render_template('index.html', recommendations=generate_top5_prod_recom(username))


if __name__ == "__main__":
    app.run()

