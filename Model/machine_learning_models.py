# Machine learning model RandomForest and Logistic Regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def train_RF(X_train, Y_train, X_test, n_estimators, max_features, random_state):
    RF_clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=random_state)
    RF_clf.fit(X_train, Y_train)
    Y_predict = RF_clf.predict(X_test)
    return Y_predict


def train_LR(X_train, Y_train, X_test, random_state, max_iter):
    LR_clf = LogisticRegression(random_state=random_state, max_iter=max_iter)
    LR_clf.fit(X_train, Y_train)
    Y_predict = LR_clf.predict(X_test)
    return Y_predict


def train_QDA(X_train, Y_train, X_test, reg_param):
    clf_Quad = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    clf_Quad.fit(X_train, Y_train)
    Y_predict = clf_Quad.predict(X_test)
    return Y_predict

