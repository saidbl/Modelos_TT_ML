import numpy as np
import pandas as pd
import joblib
from pyproj import Transformer

DATASET_PATH = "dataset.csv"

CELL_SIZE = 400
RADII = [1, 2]
EXPECTED_CLASSES = list(range(1, 21))

MODEL_FILES = {
    "XGB": ("modelo_vecindad_XGB.pkl", "modelo_vecindad_XGB_labelencoder.pkl"),
}

TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)

def to_utm_xy(lon: float, lat: float) -> tuple[float, float]:
    x, y = TRANSFORMER.transform(lon, lat)
    return float(x), float(y)

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

def build_feature_for_cell(cell_counts: pd.DataFrame, cx: int, cy: int, radii: list[int]) -> np.ndarray:
    expected_classes = [c for c in cell_counts.columns if c != "target"]
    idx_set = set(cell_counts.index)

    feat_parts = []
    for r in radii:
        nb_counts = np.zeros(len(expected_classes), dtype=float)

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nb = (cx + dx, cy + dy)
                if nb in idx_set:
                    nb_counts += cell_counts.loc[nb, expected_classes].values

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
    return feat

def topk_from_proba(proba: np.ndarray, k: int) -> list[tuple[int, float]]:
    idx = np.argsort(proba)[-k:][::-1]
    return [(int(i), float(proba[i])) for i in idx]


def predict_for_coordinate(lat: float, lon: float, cell_counts: pd.DataFrame, models: dict):
    x, y = to_utm_xy(lon, lat)
    cx = int(x // CELL_SIZE)
    cy = int(y // CELL_SIZE)

    feat = build_feature_for_cell(cell_counts, cx, cy, RADII)
    X = feat.reshape(1, -1)

    print("\n============================")
    print(f"Coord: lat={lat}, lon={lon}")
    print(f"UTM:   x={x:.1f}, y={y:.1f}")
    print(f"Cell:  cell_x={cx}, cell_y={cy}")
    print("============================")

    for name, (model, le) in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes_internal = model.classes_
            top3_internal = topk_from_proba(proba, 3)
            top3 = []
            for cls_idx, p in top3_internal:
                coded_label = int(classes_internal[cls_idx])
                original = int(le.inverse_transform([coded_label])[0])
                top3.append((original, p))

            # top1
            pred_coded = int(model.predict(X)[0])
            pred_original = int(le.inverse_transform([pred_coded])[0])

            print(f"\n[{name}] Pred top1 actividad_id = {pred_original}")
            print(f"[{name}] Top-3 (actividad_id, prob): {top3}")

        else:
            pred_coded = int(model.predict(X)[0])
            pred_original = int(le.inverse_transform([pred_coded])[0])
            print(f"\n[{name}] Pred top1 actividad_id = {pred_original}")


def main():
    df = pd.read_csv(DATASET_PATH)
    df["actividad_id"] = df["actividad_id"].astype(int)

    x, y = TRANSFORMER.transform(df["lon"].values, df["lat"].values)
    df["x"] = x
    df["y"] = y

    cell_counts = build_cell_counts(df, CELL_SIZE, EXPECTED_CLASSES)
    models = {}
    for name, (model_path, le_path) in MODEL_FILES.items():
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        models[name] = (model, le)

    test_coords = [
        (19.432608, -99.133209),  
        (19.437800, -99.203200),  
        (19.360000, -99.270000),  
        (19.332000, -99.188000), 
        (19.436100, -99.071900),  
        (19.347000, -99.062000),  
        (19.257000, -99.105000),
        (19.420400, -99.181900),
    ]

    for lat, lon in test_coords:
        predict_for_coordinate(lat, lon, cell_counts, models)


if __name__ == "__main__":
    main()