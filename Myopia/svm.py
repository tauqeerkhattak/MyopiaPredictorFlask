from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Myopia.datasets import Myopia

def svm(gender,age,study,parents,gparents,siblings,smoking,glasses,eyeExam,indoorAct,readingTime,books,bookDistance,outdoorAct,sleepingTime,goToSleep,wakeUpTime,exercise):
    print('Support Vector Machine: ')
    responsesData = Myopia.data
    outputData = Myopia.target
    xTrain, xTest, yTrain, yTest = train_test_split(responsesData, outputData, test_size=0.20)
    sc = StandardScaler()
    xTrain = sc.fit_transform(xTrain)
    xTest = sc.transform(xTest)
    svmClassifier = SVC(kernel='linear')
    svmClassifier.fit(xTrain, yTrain)
    predictionY = svmClassifier.predict(xTest)
    return 'Accuracy: '+str(metrics.accuracy_score(yTest, predictionY))

# def svm():
#     print(classification_report(yTest,predictionY))
#     predictionCustom = svmClassifier.predict(np.array(
#         [1, 19, 2, 0, 2, 0, 0, 0, 0, 5, 2, 1, 35, 5, 8, 2300, 700, 2, ]
#     ).reshape(1,-1))
#     print('Custom Prediction: ', predictionCustom)