from flask import Flask, request, render_template

from src.data_querying import generate_insights
from src.lang_chain import init_vector_db

llm, vector_db = init_vector_db()

app = Flask(__name__, template_folder='templates')


@app.route('/ask', methods=['POST'])
def get_answer():
    question = request.json['question']
    answer = generate_insights(llm, vector_db, question)

    return answer


@app.route("/")
def hello_world():
    return render_template('index.html')
