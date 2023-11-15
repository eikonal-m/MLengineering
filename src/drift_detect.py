#
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import alibi
from alibi_detect.cd import TabularDrift

wine_data = load_wine()
feature_names = wine_data.feature_names

X, y = wine_data.data, wine_data.target

X_ref, X_test, y_ref, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

cd = TabularDrift(X_ref=X_ref, p_val=0.05)

preds = cd.predict(X_test)
labels = ['No', 'Yes']
print('Drift: {}'.format(labels[preds['data']['is_drift']]))



