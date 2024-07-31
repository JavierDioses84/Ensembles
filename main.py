import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

# Cargar datos desde el archivo CSV
data = pd.read_csv('data.csv')
X = data.drop('churn', axis=1)
y = data['churn']

# Crear clasificadores individuales
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Crear el ensemble con VotingClassifier
ensemble_clf = VotingClassifier(estimators=[('rf', rf_clf), ('gb', gb_clf)], voting='soft')

# Validación cruzada de 10 pliegues
kf = KFold(n_splits=10, random_state=42, shuffle=True)
cv_scores = cross_val_score(ensemble_clf, X, y, cv=kf)
print(f'10-Fold CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})')
