import pandas as pd
import optuna
import time

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score

df=pd.read_csv("dataset/womens_clothing_analysis.csv")
df["Rating"]=pd.to_numeric(df["Rating"])
df=df.dropna(subset=["Rating"])
df=df[df["Rating"].between(1,5)]

x=pd.DataFrame({

    "length":df["Review"].astype(str).str.len(),
    "word":df["Review"].astype(str).str.split().str.len()
})

y=(df["Rating"]>=4).astype(int)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


results = []
params = {"n_estimators": [50, 100], "max_depth": [5, 10]}


grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    params, cv=3, scoring="f1", n_jobs=-1)
start = time.time()
grid.fit(x_train, y_train)
results.append(["Grid Search", f1_score(y_test, grid.predict(x_test)),
                accuracy_score(y_test, grid.predict(x_test)),
                time.time()-start, grid.best_params_])


random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [50, 75, 100, 125, 150],
     "max_depth": [3, 5, 8, 10, 15, None]},
    n_iter=6, cv=3, scoring="f1", random_state=42, n_jobs=-1
)
start = time.time()
random.fit(x_train, y_train)
results.append(["Random Search", f1_score(y_test, random.predict(x_test)),
                accuracy_score(y_test, random.predict(x_test)),
                time.time()-start, random.best_params_])


def objective(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 150, step=25),
        max_depth=trial.suggest_categorical("max_depth", [3, 5, 8, 10, 15, None]),
        random_state=42
    )
    return cross_val_score(model, x_train, y_train, cv=3, scoring="f1").mean()

study = optuna.create_study(direction="maximize")
start = time.time()
study.optimize(objective, n_trials=6)
model = RandomForestClassifier(**study.best_params, random_state=42)
model.fit(x_train, y_train)

results.append(["Bayesian Optimization", f1_score(y_test, model.predict(x_test)),
                accuracy_score(y_test, model.predict(x_test)),
                time.time()-start, study.best_params])


result = pd.DataFrame(results, columns=[
    "Method", "F1 Score", "Accuracy", "Time (sec)", "Best Parameters"
])
print(result.to_string(index=False))