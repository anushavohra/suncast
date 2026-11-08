import itertools
import os
import pandas as pd
from model import load_split, build_model, scores

X_train, X_val, X_test, y_train, y_val, y_test = load_split()

# Parameters to test
learning_rates = [0.05, 0.1, 0.2]
max_depths = [3, 6, None]
max_iters = [100, 300]

rows = []
for lr, depth, iters in itertools.product(learning_rates, max_depths, max_iters):
    model = build_model(learning_rate=lr, max_depth=depth, max_iter=iters)
    model.fit(X_train, y_train)
    mae, rmse, r2 = scores(y_val, model.predict(X_val))
    rows.append({"learning_rate": lr, "max_depth": depth, "max_iter": iters,
                 "val_MAE": round(mae, 2), "val_RMSE": round(rmse, 2),
                 "val_R2": round(r2, 4)})
    print(rows[-1])

results = pd.DataFrame(rows).sort_values("val_RMSE")
os.makedirs("results", exist_ok=True)
results.to_csv("results/tuning_results.csv", index=False)
print("Best setting:")
print(results.iloc[0])