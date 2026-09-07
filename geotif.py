import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin

# ---------------------------------------------------------------------------
# STEP 1 — Load the trained model and the prediction grid
#          (identical to notebook Cells 0-2)
# ---------------------------------------------------------------------------
model = joblib.load("manganese_xgboost.pkl")
df = pd.read_csv("input.csv")

print("Loaded grid shape:", df.shape)
print("Columns:", list(df.columns))

# ---------------------------------------------------------------------------
# STEP 2 — Rename raw band codes to descriptive sensor-tagged names
#          (identical to notebook Cell 8 — must match model training names)
# ---------------------------------------------------------------------------
df = df.rename(columns={
    "B2": "S2_B2_Blue",
    "B3": "S2_B3_Green",
    "B4": "S2_B4_Red",
    "B5": "S2_B5_RedEdge1",
    "B6": "S2_B6_RedEdge2",
    "B7": "S2_B7_RedEdge3",
    "B8A": "S2_B8A_NIR",
    "B11": "S2_B11_SWIR1",
    "B12": "S2_B12_SWIR2",
    "S2_NDVI": "S2_NDVI",
    "Fe_Oxide_Proxy": "S2_Fe_Oxide_Proxy",
    "SWIR_Ratio": "S2_SWIR_Ratio",
    "LST_C": "Landsat_LST_C",
    "VV": "S1_VV_dB",
    "VH": "S1_VH_dB",
    "VV_VH_Difference": "S1_VV_VH_Difference_dB"
})

# ---------------------------------------------------------------------------
# STEP 3 — Select the exact 23-feature matrix used at inference time
#          (identical to notebook Cell 9 — same order the model expects)
# ---------------------------------------------------------------------------
feature_cols = [
    'S2_B2_Blue',
    'S2_B3_Green',
    'S2_B4_Red',
    'S2_B5_RedEdge1',
    'S2_B6_RedEdge2',
    'S2_B7_RedEdge3',
    'S2_B8A_NIR',
    'S2_B11_SWIR1',
    'S2_B12_SWIR2',
    'S2_NDVI',
    'S2_Fe_Oxide_Proxy',
    'S2_SWIR_Ratio',
    'Landsat_LST_C',
    'S1_VV_dB',
    'S1_VH_dB',
    'S1_VV_VH_Difference_dB',
    'lithology_class_id',
    'geomorphology_class_id',
    'distance_to_lineament_m',
    'lineament_density_1km',
    'distance_to_structure_m',
    'structure_density_1km',
    'bhuvan_lulc_class_id'
]

# --- Safety check that was MISSING in the original notebook (see analysis §13,#4) ---
missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    raise ValueError(
        f"These expected feature columns are missing from input.csv after "
        f"renaming: {missing_features}. Check Step 2's rename map against "
        f"your actual input.csv column names."
    )

X_new = df[feature_cols]

# ---------------------------------------------------------------------------
# STEP 4 — Run inference: continuous prospectivity probability per cell
#          (identical to notebook Cell 10)
# ---------------------------------------------------------------------------
probs = model.predict_proba(X_new)
df["prospectivity"] = probs[:, 1]

print(df["prospectivity"].describe())

# ---------------------------------------------------------------------------
# STEP 5 — Jenks Natural Breaks classification (identical to notebook Cells 12-13)
#          NOTE: breaks are read directly from the `breaks` variable this time,
#          NOT re-typed as hardcoded literals (fixes analysis §13, issue #2).
# ---------------------------------------------------------------------------
import jenkspy

sample = df["prospectivity"].sample(n=min(100_000, len(df)), random_state=42)
breaks = jenkspy.jenks_breaks(sample, n_classes=5)
print("Jenks breaks:", breaks)

df["prospectivity_class"] = pd.cut(
    df["prospectivity"],
    bins=breaks,
    labels=["Very Low", "Low", "Moderate", "High", "Very High"],
    include_lowest=True
)

df.to_csv("manganese_prospectivity_final.csv", index=False)
# ===========================================================================
# STEP 6 — Build Raster Grid
# ===========================================================================

required_raster_cols = ["row", "col", "x_utm_m", "y_utm_m"]

missing_raster_cols = [c for c in required_raster_cols if c not in df.columns]

if missing_raster_cols:
    raise ValueError(
        f"Missing raster columns: {missing_raster_cols}"
    )

n_rows = int(df["row"].max()) + 1
n_cols = int(df["col"].max()) + 1

unique_x_sorted = np.sort(df["x_utm_m"].unique())
pixel_size_x = np.median(np.diff(unique_x_sorted))

unique_y_sorted = np.sort(df["y_utm_m"].unique())
pixel_size_y = np.median(np.diff(unique_y_sorted))

pixel_size = float(pixel_size_x)

x_min = float(df["x_utm_m"].min())
x_max = float(df["x_utm_m"].max())
y_min = float(df["y_utm_m"].min())
y_max = float(df["y_utm_m"].max())

print(f"rows={n_rows}")
print(f"cols={n_cols}")
print(f"pixel_size={pixel_size}")

transform = from_origin(
    x_min,
    y_max,
    pixel_size,
    pixel_size
)

CRS_EPSG = "EPSG:32644"

# ===========================================================================
# STEP 6A — Probability Raster
# ===========================================================================

raster_prob = np.full(
    (n_rows, n_cols),
    -9999,
    dtype=np.float32
)

raster_prob[
    df["row"].astype(int).values,
    df["col"].astype(int).values
] = df["prospectivity"].astype(np.float32).values

# ===========================================================================
# STEP 6B — Classified Raster
# ===========================================================================

class_map = {
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5
}

df["class_id"] = df["prospectivity_class"].map(class_map)

raster_class = np.full(
    (n_rows, n_cols),
    -9999,
    dtype=np.int16
)

raster_class[
    df["row"].astype(int).values,
    df["col"].astype(int).values
] = df["class_id"].astype(np.int16).values

# ===========================================================================
# STEP 6C — Diagnostic Plot
# ===========================================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    raster_prob,
    cmap="RdYlGn_r"
)

plt.colorbar(label="Prospectivity")

plt.title("Raw Prediction Grid")

plt.tight_layout()

plt.show()

# ===========================================================================
# STEP 7 — Save Probability GeoTIFF
# ===========================================================================

with rasterio.open(
    "manganese_prospectivity_probability.tif",
    "w",
    driver="GTiff",
    height=n_rows,
    width=n_cols,
    count=1,
    dtype="float32",
    crs=CRS_EPSG,
    transform=transform,
    nodata=-9999,
) as dst:

    dst.write(raster_prob, 1)

    dst.update_tags(
        DESCRIPTION="Manganese Prospectivity Probability",
        MODEL="XGBoost"
    )

print("Saved manganese_prospectivity_probability.tif")
# ===========================================================================
# STEP 7 — Write manganese_prospectivity.tif
# ===========================================================================
with rasterio.open(
    "manganese_prospectivity.tif",
    "w",
    driver="GTiff",
    height=n_rows,
    width=n_cols,
    count=1,
    dtype="float32",
    crs=CRS_EPSG,
    transform=transform,
    nodata=np.nan,
) as dst:
    dst.write(raster_prob, 1)
    dst.update_tags(
        DESCRIPTION="Manganese prospectivity probability (0-1), XGBoost model",
        SOURCE_MODEL="manganese_xgboost.pkl",
    )

print("Wrote manganese_prospectivity.tif")
# ===========================================================================
# STEP 8 — Save Classified GeoTIFF
# ===========================================================================

with rasterio.open(
    "manganese_prospectivity_classified.tif",
    "w",
    driver="GTiff",
    height=n_rows,
    width=n_cols,
    count=1,
    dtype="int16",
    crs=CRS_EPSG,
    transform=transform,
    nodata=-9999,
) as dst:

    dst.write(raster_class, 1)

    dst.update_tags(
        DESCRIPTION="Jenks Classified Manganese Prospectivity"
    )

print("Saved manganese_prospectivity_classified.tif")

# ===========================================================================
# STEP 9 — Verify
# ===========================================================================

with rasterio.open(
    "manganese_prospectivity_probability.tif"
) as src:

    print("\n--- Verification ---")
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Resolution:", src.res)
    print("Width:", src.width)
    print("Height:", src.height)

# ===========================================================================
# STEP 10 — Save PNG
# ===========================================================================

plt.figure(figsize=(12, 10))

img = plt.imshow(
    raster_prob,
    cmap="RdYlGn_r",
    extent=[x_min, x_max, y_min, y_max],
    origin="upper"
)

plt.colorbar(
    img,
    label="Manganese Prospectivity"
)

plt.title(
    "Manganese Prospectivity Map"
)

plt.xlabel("UTM Easting (m)")
plt.ylabel("UTM Northing (m)")

plt.tight_layout()

plt.savefig(
    "manganese_prospectivity.png",
    dpi=300
)

plt.close()

print("Saved manganese_prospectivity.png")

print("\nDONE")