import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import joblib
import json
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATASET_PATH = "dataset.csv"

EXPECTED_CLASSES = list(range(1, 21)) 
CELL_SIZE = 400                       
RADII = [1, 2]                         
N_SPLITS = 4

BLOCK_SIZE_METERS = 2000              
MIN_TARGET_SHARE = 0.002              

RANDOM_STATE = 42

def to_utm(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    x, y = transformer.transform(df["lon"].values, df["lat"].values)
    out = df.copy()
    out["x"] = x
    out["y"] = y
    return out

def build_cell_counts(df: pd.DataFrame, cell_size: int, expected_classes: list[int]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["cell_x"] = (tmp["x"] // cell_size).astype(int)
    tmp["cell_y"] = (tmp["y"] // cell_size).astype(int)

    cell_counts = (
        tmp.groupby(["cell_x", "cell_y", "actividad_id"])
        .size()
        .unstack(fill_value=0)
    )

    for c in expected_classes:
        if c not in cell_counts.columns:
            cell_counts[c] = 0

    cell_counts = cell_counts[expected_classes]
    cell_counts["target"] = cell_counts.idxmax(axis=1)
    return cell_counts

def entropy_and_simpson(props: np.ndarray) -> tuple[float, float]:
    nonzero = props > 0
    ent = -(props[nonzero] * np.log(props[nonzero])).sum()
    simp = 1.0 - np.sum(props ** 2)
    return float(ent), float(simp)

def topk_accuracy_from_proba(proba: np.ndarray, y_true: np.ndarray, classes_: np.ndarray, k: int) -> float:
    topk_idx = np.argsort(proba, axis=1)[:, -k:]
    topk_labels = classes_[topk_idx]
    hit = np.any(topk_labels == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(hit))

def build_features_multiscale(
    cell_counts: pd.DataFrame,
    cell_size: int,
    radii: list[int],
    block_size_meters: int,
    min_target_share: float
):
    expected_classes = [c for c in cell_counts.columns if c != "target"]

    target_dist = cell_counts["target"].value_counts(normalize=True)
    valid_targets = target_dist[target_dist >= min_target_share].index.tolist()
    cc = cell_counts[cell_counts["target"].isin(valid_targets)].copy()

    idx_set = set(cc.index)

    block_in_cells = max(1, int(round(block_size_meters / cell_size)))

    X_list, y_list, g_list = [], [], []

    for (cx, cy), row in cc.iterrows():
        feat_parts = []
        for r in radii:
            nb_counts = np.zeros(len(expected_classes), dtype=float)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nb = (cx + dx, cy + dy)
                    if nb in idx_set:
                        nb_counts += cc.loc[nb, expected_classes].values

            total = nb_counts.sum()
            if total <= 0:
                props = np.zeros_like(nb_counts)
                ent, simp = 0.0, 0.0
                max_prop, margin = 0.0, 0.0
            else:
                props = nb_counts / total
                ent, simp = entropy_and_simpson(props)
                sorted_props = np.sort(props)
                max_prop = float(sorted_props[-1])
                margin = float(sorted_props[-1] - sorted_props[-2]) if len(sorted_props) >= 2 else float(sorted_props[-1])

            feat_parts.append(nb_counts)
            feat_parts.append(props)
            feat_parts.append(np.array([total, ent, simp, max_prop, margin], dtype=float))

        feat = np.concatenate(feat_parts)

        X_list.append(feat)
        y_list.append(int(row["target"]))

        bx = int(cx // block_in_cells)
        by = int(cy // block_in_cells)
        g_list.append(bx * 1_000_000 + by)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    groups = np.array(g_list, dtype=int)

    return X, y, groups, valid_targets


def get_cv_splitter(n_splits: int):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), "StratifiedGroupKFold"
    except Exception:
        return GroupKFold(n_splits=n_splits), "GroupKFold"


def evaluate_model_cv(model_ctor, X, y, groups, splitter):
    macro_scores, bal_scores, top2_scores, top3_scores = [], [], [], []
    baseline_macro, baseline_bal = [], []

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        maj = Counter(ytr).most_common(1)[0][0]
        base_pred = np.full_like(yte, maj)
        baseline_macro.append(f1_score(yte, base_pred, average="macro"))
        baseline_bal.append(balanced_accuracy_score(yte, base_pred))
        classes_present = np.unique(ytr)
        cw = compute_class_weight(class_weight="balanced", classes=classes_present, y=ytr)
        cw_map = {c: w for c, w in zip(classes_present, cw)}
        sample_weight = np.array([cw_map[yy] for yy in ytr], dtype=float)

        model = model_ctor()
        model.fit(Xtr, ytr, sample_weight=sample_weight)

        pred = model.predict(Xte)
        macro = f1_score(yte, pred, average="macro")
        bal = balanced_accuracy_score(yte, pred)

        macro_scores.append(macro)
        bal_scores.append(bal)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xte)
            classes_ = model.classes_
            top2_scores.append(topk_accuracy_from_proba(proba, yte, classes_, k=2))
            top3_scores.append(topk_accuracy_from_proba(proba, yte, classes_, k=3))

        print(f"\n--- Fold {fold} ---")
        print(f"Baseline: MacroF1={baseline_macro[-1]:.4f} BalAcc={baseline_bal[-1]:.4f}")
        print(f"Model:    MacroF1={macro:.4f} BalAcc={bal:.4f}"
              + (f" Top2={top2_scores[-1]:.4f} Top3={top3_scores[-1]:.4f}" if top2_scores else ""))

    out = {
        "macro_f1_mean": float(np.mean(macro_scores)),
        "macro_f1_std": float(np.std(macro_scores)),
        "bal_acc_mean": float(np.mean(bal_scores)),
        "bal_acc_std": float(np.std(bal_scores)),
        "baseline_macro_f1_mean": float(np.mean(baseline_macro)),
        "baseline_bal_acc_mean": float(np.mean(baseline_bal)),
    }
    if top2_scores:
        out["top2_mean"] = float(np.mean(top2_scores))
        out["top3_mean"] = float(np.mean(top3_scores))
    return out
def lgbm_ctor():
    return LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1  
    )

def xgb_ctor():
    return XGBClassifier(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )
def main():
    df = pd.read_csv(DATASET_PATH)
    for col in ["lat", "lon", "actividad_id"]:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col}")

    df["actividad_id"] = df["actividad_id"].astype(int)
    df = to_utm(df)

    cell_counts = build_cell_counts(df, CELL_SIZE, EXPECTED_CLASSES)
    X, y, groups, valid_targets = build_features_multiscale(
        cell_counts=cell_counts,
        cell_size=CELL_SIZE,
        radii=RADII,
        block_size_meters=BLOCK_SIZE_METERS,
        min_target_share=MIN_TARGET_SHARE
    )
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(y)

    print("=====================================")
    print(f"CELL_SIZE={CELL_SIZE} | RADII={RADII} | CV blocks={BLOCK_SIZE_METERS}m")
    print("=====================================")
    print(f"Total celdas usadas: {len(X)}")
    print(f"Targets válidos (post-filtro): {sorted(valid_targets)}")

    splitter, splitter_name = get_cv_splitter(N_SPLITS)
    print(f"Splitter: {splitter_name}")
    def rf_ctor():
        return RandomForestClassifier(
            n_estimators=1000,
            max_depth=28,
            min_samples_leaf=2,
            min_samples_split=5,
            class_weight=None, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    def et_ctor():
        return ExtraTreesClassifier(
            n_estimators=1200,
            max_depth=28,
            min_samples_leaf=2,
            min_samples_split=5,
            class_weight=None,  
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    print("\n===== Evaluando RandomForest =====")
    rf_res = evaluate_model_cv(rf_ctor, X, y, groups, splitter)

    print("\n===== Evaluando ExtraTrees =====")
    et_res = evaluate_model_cv(et_ctor, X, y, groups, splitter)

    print("\n===== Evaluando LightGBM =====")
    lgbm_res = evaluate_model_cv(lgbm_ctor, X, y, groups, splitter)

    print("\n===== Evaluando XGBoost =====")
    xgb_res = evaluate_model_cv(xgb_ctor, X, y, groups, splitter)

    all_models = {
        "RF": (rf_ctor, rf_res),
        "ET": (et_ctor, et_res),
        "LGBM": (lgbm_ctor, lgbm_res),
        "XGB": (xgb_ctor, xgb_res),
    }

    sorted_models = sorted(
        all_models.items(),
        key=lambda x: x[1][1]["macro_f1_mean"],
        reverse=True
    )

    top2 = sorted_models[:2]
    print("\n===== RESUMEN FINAL =====")

    for name, (_, res) in sorted_models:
        print(f"{name}: {res}")

    print("\n TOP 2 MODELOS:")
    for name, (_, res) in top2:
        print(f"{name}  MacroF1={res['macro_f1_mean']:.4f}  BalAcc={res['bal_acc_mean']:.4f}")

    classes_present = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes_present, y=y)
    cw_map = {c: w for c, w in zip(classes_present, cw)}
    sample_weight = np.array([cw_map[yy] for yy in y], dtype=float)

    for name, (ctor, res) in top2:
        final_model = ctor()
        final_model.fit(X, y, sample_weight=sample_weight)

        joblib.dump(final_model, f"modelo_vecindad_{name}.pkl")
        joblib.dump(le, f"modelo_vecindad_{name}_labelencoder.pkl")

        meta = {
            "model": name,
            "cell_size": CELL_SIZE,
            "radii": RADII,
            "neighbor_shapes": [f"{2*r+1}x{2*r+1}" for r in RADII],
            "block_size_meters": BLOCK_SIZE_METERS,
            "min_target_share": MIN_TARGET_SHARE,
            "valid_target_classes_used": sorted(list(set(y))),
            "cv_results": res,
            "splitter": splitter_name
        }

        with open(f"modelo_vecindad_{name}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n Guardado modelo_vecindad_{name}.pkl")


if __name__ == "__main__":
    main()