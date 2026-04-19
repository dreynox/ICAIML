import json
import re

file_path = "ICAIML.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell.get("source", []))
    
    # 1. Imports
    if "from sklearn.model_selection import (" in source and "train_test_split" in source:
        if "GroupShuffleSplit" not in source:
            new_imports = """from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GroupShuffleSplit,
    GroupKFold
)"""
            source = re.sub(r"from sklearn\.model_selection import \([\s\S]*?\)", new_imports, source)
    
    # 2. X and y definitions + Split (Cell 70 and 71)
    if "X = df.drop(columns=['Plant_Health_Status_Encoded'])" in source:
        source = source.replace("X = df.drop(columns=['Plant_Health_Status_Encoded'])", 
                                "leakage_features = ['Plant_Health_Status_Encoded', 'Electrochemical_Signal', 'Chlorophyll_Content', 'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']\n"
                                "X = df.drop(columns=leakage_features, errors='ignore')\n"
                                "groups = df['Plant_ID'] if 'Plant_ID' in df.columns else None")
    
    if "X_train, X_test, y_train, y_test = train_test_split(" in source and "test_size=0.2" in source and "stratify=y" in source:
        new_split = """# Grouped Train-Test Split to Prevent Leakage
if groups is not None:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    X_train = X_train.drop(columns=['Plant_ID'], errors='ignore')
    X_test = X_test.drop(columns=['Plant_ID'], errors='ignore')
    groups_train = groups.iloc[train_idx]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    groups_train = None"""
        source = re.sub(r"X_train, X_test, y_train, y_test = train_test_split\([\s\S]*?\)", new_split, source)
        
    # 3. Regularization in Cell 74
    if "min_samples_leaf=10," in source and "RandomForestClassifier" in source:
        source = source.replace("max_depth=7,\n            min_samples_leaf=10,", "max_depth=5,\n            min_samples_leaf=20,")
        source = source.replace('eval_metric="mlogloss",', 'eval_metric="mlogloss",\n            reg_alpha=0.1,\n            reg_lambda=1.0,')

    # 4. Fix X_test_scaled usages in predictions
    if "best_model.predict(X_test_scaled)" in source:
        source = source.replace("best_model.predict(X_test_scaled)", "best_model.predict(X_test)")
        
    # 5. Fix CV in Cell 84
    if "cv = StratifiedKFold(" in source and "cv_results = []" in source:
        source = source.replace("cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)", 
                                "cv = GroupKFold(n_splits=5) if 'groups_train' in locals() and groups_train is not None else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
        source = source.replace("X_train_scaled,", "X_train,")
        source = source.replace("cv=cv,", "cv=cv,\n        groups=groups_train if isinstance(cv, GroupKFold) else None,")

    # 6. Fix Robustness test in Cell 86
    if "X_test_noisy = X_test_scaled +" in source:
        new_noise = """std_devs = X_test.std(axis=0)
noise_arr = np.random.normal(loc=0, scale=1.0, size=X_test.shape) * std_devs.values * noise_factor
X_test_noisy = X_test + noise_arr"""
        source = re.sub(r"X_test_noisy = X_test_scaled \+ np\.random\.normal\([\s\S]*?size=X_test_scaled\.shape\n\)", new_noise, source)
        
        # Remove the refit of model in loop because models are already fit
        source = source.replace("model.fit(X_train_scaled, y_train)", "")
        source = source.replace("y_pred_clean = model.predict(X_test_scaled)", "y_pred_clean = model.predict(X_test)")

    # 7. Fix SHAP values in Cell 87
    if "explainer = shap.TreeExplainer(model)" in source:
        if "model.named_steps" not in source:
            source = source.replace("explainer = shap.TreeExplainer(model)", "explainer = shap.TreeExplainer(model.named_steps['model'])")
            source = source.replace("shap_values = explainer.shap_values(X_test_scaled)", "shap_values = explainer.shap_values(X_test)")
            source = source.replace("X_test_scaled,", "X_test,\n    feature_names=X_test.columns")
            source = source.replace("data=X_test_scaled[sample_idx],", "data=X_test.iloc[sample_idx],")
            source = source.replace("feature_names=X.columns", "feature_names=X_test.columns")

    # Put back into lines
    if "groups=groups_train" in source and "return_train_score=False" in source:
        # Avoid duplicate modification, handle gracefully
        pass
        
    lines = source.splitlines(True)
    if len(lines) == 0 and source:
        lines = [source]
    cell["source"] = lines

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook fixes applied successfully.")
