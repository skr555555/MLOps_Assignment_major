from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

data=fetch_olivetti_faces()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
clf=DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)
preds=clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,preds))
joblib.dump({'model':clf,'X_test':X_test,'y_test':y_test},'savedmodel.pth')
print('savedmodel.pth generated')
