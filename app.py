import json

from flask import Flask, request

from Myopia.svm import svm

app = Flask(__name__)


@app.route('/', methods=['POST','GET'])
def hello_world():  # put application's code here
    gender = request.args['gender']
    age = request.args['age']
    study = request.args['study']
    parents = request.args['parents']
    gparents = request.args['gparents']
    siblings = request.args['siblings']
    smoking = request.args['smoking']
    glasses = request.args['glasses']
    eyeExam = request.args['eyeExam']
    indoorAct = request.args['indoorAct']
    readingTime = request.args['readingTime']
    books = request.args['books']
    bookDistance = request.args['bookDistance']
    outdoorAct = request.args['outdoorAct']
    sleepingTime = request.args['sleepingTime']
    goToSleep = request.args['goToSleep']
    wakeUpTime = request.args['wakeUpTime']
    exercise = request.args['exercise']
    return svm(
        gender,
        age,
        study,
        parents,
        gparents,
        siblings,
        smoking,
        glasses,
        eyeExam,
        indoorAct,
        readingTime,
        books,
        bookDistance,
        outdoorAct,
        sleepingTime,
        goToSleep,
        wakeUpTime,
        exercise,
    )


if __name__ == '__main__':
    app.run()
