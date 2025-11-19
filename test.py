import joblib
from sklearn.metrics import accuracy_score
bundle=joblib.load('savedmodel.pth')
clf=bundle['model']
X_test=bundle['X_test']
y_test=bundle['y_test']
preds=clf.predict(X_test)
print('Loaded accuracy:',accuracy_score(y_test,preds))
