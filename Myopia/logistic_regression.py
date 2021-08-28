import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from Myopia.datasets import Myopia


def logistic_regression(gender, age, study, parents, gparents,siblings,smoking,glasses,eyeExam,indoorAct,readingTime,books,bookDistance,outdoorAct,sleepingTime,goToSleep,wakeUpTime,exercise):
    jsonFile = {}
    responses = Myopia.data
    output = Myopia.target
    xTrain, xTest, yTrain, yTest = train_test_split(responses, output, test_size=0.20, random_state=42)
    sc = StandardScaler()
    xTrain = sc.fit_transform(xTrain)
    xTest = sc.transform(xTest)
    logisticRegressor = LogisticRegression(random_state=0)
    logisticRegressor.fit(xTrain, yTrain)
    predictionY = logisticRegressor.predict(xTest)
    predictionCustom = logisticRegressor.predict(np.array(
        [gender, age, study, parents, gparents, siblings, smoking, glasses, eyeExam, indoorAct, readingTime, books, bookDistance, outdoorAct, sleepingTime, goToSleep, wakeUpTime, exercise],
    ).reshape(1, -1))
    jsonFile['Myopia'] = str(predictionCustom[0])
    jsonFile['ModelAccuracy'] = str(metrics.accuracy_score(yTest, predictionY))
    jsonFile['ModelUsed'] = "Logistic Regression Algorithm"
    return jsonFile
