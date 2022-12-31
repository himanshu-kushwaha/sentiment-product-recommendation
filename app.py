from flask import Flask, request, render_template
from model import get_top5_recommendations

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getRecommendation', methods=['POST'])
def getRecommendation():
    user_name = request.form['username'].lower()
    recommendation = get_top5_recommendations(user_name)

    return render_template("index.html", output=recommendation, message_display=None, user_name=user_name) if recommendation is not None \
            else render_template("index.html",
                               message_display=f"User Name {user_name} doesn't exist. Please provide a valid user!")


if __name__ == '__main__':
    app.run()
