import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


def ensembleVoting1(X,y):
    # Crear clasificadores individuales
    # Random Forest + Gradient Boosting + Voting Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Crear el ensemble con VotingClassifier
    ensemble_clf = VotingClassifier(estimators=[('rf', rf_clf), ('gb', gb_clf)], voting='soft')

    # Validación cruzada de 10 pliegues
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_scores = cross_val_score(ensemble_clf, X, y, cv=kf)
    print(f'10-Fold CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})')

def ensembleVoting2(X,y):
    # Random Forest + SVM + Voting Classifier
    # Crear clasificadores individuales
    from sklearn.svm import SVC
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = SVC(probability=True, random_state=42)

    # Crear el ensemble con VotingClassifier
    ensemble_clf = VotingClassifier(estimators=[('rf', rf_clf), ('svm', svm_clf)], voting='soft')

    # Validación cruzada de 10 pliegues
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_scores = cross_val_score(ensemble_clf, X, y, cv=kf)
    print(f'10-Fold CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})')

def ensembleVoting3(X,y):
    # Random Forest + KNN + Voting Classifier
    # Crear clasificadores individuales
    from sklearn.neighbors import KNeighborsClassifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    knn_clf = KNeighborsClassifier(n_neighbors=5)

    # Crear el ensemble con VotingClassifier
    ensemble_clf = VotingClassifier(estimators=[('rf', rf_clf), ('knn', knn_clf)], voting='soft')

    # Validación cruzada de 10 pliegues
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_scores = cross_val_score(ensemble_clf, X, y, cv=kf)
    print(f'10-Fold CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})')


# Cargar datos desde el archivo CSV
data = pd.read_csv('DataChurn.csv')
X = data.drop('Churn', axis=1)
y = data['Churn']

ensembleVoting1(X,y)
ensembleVoting2(X,y)
ensembleVoting3(X,y)