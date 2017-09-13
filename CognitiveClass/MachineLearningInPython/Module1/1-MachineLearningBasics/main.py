from sklearn.datasets import load_digits
from sklearn import svm

digits = load_digits()

print(digits.data.shape)
print(digits.target.shape)

X = digits.data
Y = digits.target

clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X,Y)

print("Prediction: ",clf.predict(digits.data[-1]))
print("Actual:", Y[-1])



