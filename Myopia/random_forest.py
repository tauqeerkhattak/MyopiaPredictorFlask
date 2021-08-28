import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Myopia.datasets import Myopia


def random_forest(gender,age,study,parents,gparents,siblings,smoking,glasses,eyeExam,indoorAct,readingTime,books,bookDistance,outdoorAct,sleepingTime,goToSleep,wakeUpTime,exercise):
    jsonFile = {}
    responses = Myopia.data
    output = Myopia.target
    xTrain, xTest, yTrain, yTest = train_test_split(responses, output, test_size=0.20, random_state=42)
    sc = StandardScaler()
    xTrain = sc.fit_transform(xTrain)
    xTest = sc.transform(xTest)
    randomForestClassifier = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=-1)
    randomForestClassifier.fit(xTrain, yTrain)
    predictionY = randomForestClassifier.predict(xTest)
    predictionCustom = randomForestClassifier.predict(np.array(
        [gender, age, study, parents, gparents, siblings, smoking, glasses, eyeExam, indoorAct, readingTime, books,
         bookDistance, outdoorAct, sleepingTime, goToSleep, wakeUpTime, exercise],
    ).reshape(1, -1))
    jsonFile['Myopia'] = str(predictionCustom[0])
    jsonFile['ModelAccuracy'] = str(metrics.accuracy_score(yTest,predictionY))
    jsonFile['ModelUsed'] = "Random Forest Algorithm"
    return jsonFile
