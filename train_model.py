# train_model.py
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from fe import build_training_frame, FEATURES

# 1) load
df = pd.read_csv('data/train.csv', parse_dates=['Date'])

# 2) build features
df_feat = build_training_frame(df)

X = df_feat[FEATURES]
y = df_feat['Weekly_Sales']

# 3) simple time-based split (last 10% as test)
cut = int(len(df_feat)*0.9)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

# 4) fit
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 5) quick eval
pred = rf.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("R2 :", round(r2_score(y_test, pred), 4))

# 6) save
joblib.dump(rf, 'models/rf_model.pkl')
with open('models/features.json','w') as f:
    json.dump(FEATURES, f)
print("Saved models/rf_model.pkl and models/features.json")
