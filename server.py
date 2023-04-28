from flask import Flask, request, render_template

from main import main
from src.data_querying import test, generate_insights

llm, vector_db = main()

app = Flask(__name__, template_folder='templates')


@app.route('/ask', methods=['POST'])
def get_answer():
    question = request.json['question']
    answer = generate_insights(llm, vector_db, question)

    return answer
    # return question
    # return f'User {escape(username)}'


@app.route("/")
def hello_world():
    return render_template('index.html')