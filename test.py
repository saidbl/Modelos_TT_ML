import numpy as np
from pyproj import Transformer
import joblib
import pandas as pd


CELL_SIZE = 400
RADII = [1, 2]
EXPECTED_CLASSES = list(range(1, 21))

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)

def build_cell_counts(df):
    df = df.copy()
    df["cell_x"] = (df["x"] // CELL_SIZE).astype(int)
    df["cell_y"] = (df["y"] // CELL_SIZE).astype(int)

    cell_counts = (
        df.groupby(["cell_x", "cell_y", "actividad_id"])
        .size()
        .unstack(fill_value=0)
    )

    for c in EXPECTED_CLASSES:
        if c not in cell_counts.columns:
            cell_counts[c] = 0

    return cell_counts[EXPECTED_CLASSES]

def predict_point(lat, lon, model, cell_counts):

    x, y = transformer.transform(lon, lat)

    cell_x = int(x // CELL_SIZE)
    cell_y = int(y // CELL_SIZE)

    feature_parts = []

    idx_set = set(cell_counts.index)

    for r in RADII:

        nb_counts = np.zeros(len(EXPECTED_CLASSES), dtype=float)

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nb = (cell_x + dx, cell_y + dy)
                if nb in idx_set:
                    nb_counts += cell_counts.loc[nb].values

        total = nb_counts.sum()

        if total == 0:
            props = np.zeros_like(nb_counts)
            entropy = 0
            simpson = 0
            max_prop = 0
            margin = 0
        else:
            props = nb_counts / total
            nonzero = props > 0
            entropy = -(props[nonzero] * np.log(props[nonzero])).sum()
            simpson = 1 - np.sum(props**2)

            sorted_props = np.sort(props)
            max_prop = sorted_props[-1]
            margin = sorted_props[-1] - sorted_props[-2] if len(sorted_props) >= 2 else sorted_props[-1]

        feature_parts.append(nb_counts)
        feature_parts.append(props)
        feature_parts.append(np.array([total, entropy, simpson, max_prop, margin]))

    features = np.concatenate(feature_parts)

    probs = model.predict_proba([features])[0]
    classes = model.classes_

    sorted_idx = np.argsort(probs)[::-1]

    top3 = [
        {
            "actividad_id": int(classes[i]),
            "probabilidad": float(probs[i])
        }
        for i in sorted_idx[:3]
    ]

    return top3
model = joblib.load("modelo_vecindad_ET.pkl")
df = pd.read_csv("dataset.csv")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
x, y = transformer.transform(df["lon"].values, df["lat"].values)
df["x"] = x
df["y"] = y

cell_counts = build_cell_counts(df)

resultado = predict_point(
    lat=19.4363,
    lon=-99.0721,
    model=model,
    cell_counts=cell_counts
)

print(resultado)