import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("manganese_xgboost_pipeline.pkl")

st.set_page_config(
    page_title="Manganese Prospectivity Prediction",
    layout="wide"
)

st.title("🪨 Manganese Prospectivity Prediction")

st.write(
    "Enter geological and remote sensing features "
    "to predict manganese prospectivity."
)

# ------------------------
# Example Buttons
# ------------------------

col1, col2 = st.columns(2)

with col1:
    load_high = st.button("🟢 Load High Prospectivity Example")

with col2:
    load_low = st.button("🔴 Load Low Prospectivity Example")

# ------------------------
# Default Values
# ------------------------

high_values = {
    "S2_B2_Blue":1200,
    "S2_B3_Green":1400,
    "S2_B4_Red":1500,
    "S2_B5_RedEdge1":1800,
    "S2_B6_RedEdge2":1900,
    "S2_B7_RedEdge3":2000,
    "S2_B8A_NIR":2500,
    "S2_B11_SWIR1":1700,
    "S2_B12_SWIR2":1600,
    "S2_NDVI":0.65,
    "S2_Fe_Oxide_Proxy":1.4,
    "S2_SWIR_Ratio":1.2,
    "Landsat_LST_C":32,
    "S1_VV_dB":-8,
    "S1_VH_dB":-13,
    "S1_VV_VH_Difference_dB":5,
    "distance_to_lineament_m":200,
    "lineament_density_1km":5,
    "distance_to_structure_m":150,
    "structure_density_1km":4,
    "lithology_class_id":2,
    "geomorphology_class_id":3,
    "bhuvan_lulc_class_id":1
}

low_values = {
    "S2_B2_Blue":600,
    "S2_B3_Green":700,
    "S2_B4_Red":800,
    "S2_B5_RedEdge1":900,
    "S2_B6_RedEdge2":950,
    "S2_B7_RedEdge3":1000,
    "S2_B8A_NIR":1100,
    "S2_B11_SWIR1":900,
    "S2_B12_SWIR2":850,
    "S2_NDVI":0.15,
    "S2_Fe_Oxide_Proxy":0.7,
    "S2_SWIR_Ratio":0.8,
    "Landsat_LST_C":25,
    "S1_VV_dB":-15,
    "S1_VH_dB":-20,
    "S1_VV_VH_Difference_dB":5,
    "distance_to_lineament_m":5000,
    "lineament_density_1km":1,
    "distance_to_structure_m":4000,
    "structure_density_1km":1,
    "lithology_class_id":1,
    "geomorphology_class_id":1,
    "bhuvan_lulc_class_id":2
}

if load_high:
    defaults = high_values
elif load_low:
    defaults = low_values
else:
    defaults = high_values

# ------------------------
# Inputs
# ------------------------

st.subheader("Satellite Features")

S2_B2_Blue = st.number_input("S2_B2_Blue", value=float(defaults["S2_B2_Blue"]))
S2_B3_Green = st.number_input("S2_B3_Green", value=float(defaults["S2_B3_Green"]))
S2_B4_Red = st.number_input("S2_B4_Red", value=float(defaults["S2_B4_Red"]))
S2_B5_RedEdge1 = st.number_input("S2_B5_RedEdge1", value=float(defaults["S2_B5_RedEdge1"]))
S2_B6_RedEdge2 = st.number_input("S2_B6_RedEdge2", value=float(defaults["S2_B6_RedEdge2"]))
S2_B7_RedEdge3 = st.number_input("S2_B7_RedEdge3", value=float(defaults["S2_B7_RedEdge3"]))
S2_B8A_NIR = st.number_input("S2_B8A_NIR", value=float(defaults["S2_B8A_NIR"]))
S2_B11_SWIR1 = st.number_input("S2_B11_SWIR1", value=float(defaults["S2_B11_SWIR1"]))
S2_B12_SWIR2 = st.number_input("S2_B12_SWIR2", value=float(defaults["S2_B12_SWIR2"]))

S2_NDVI = st.number_input("S2_NDVI", value=float(defaults["S2_NDVI"]))
S2_Fe_Oxide_Proxy = st.number_input("S2_Fe_Oxide_Proxy", value=float(defaults["S2_Fe_Oxide_Proxy"]))
S2_SWIR_Ratio = st.number_input("S2_SWIR_Ratio", value=float(defaults["S2_SWIR_Ratio"]))

st.subheader("Thermal Features")

Landsat_LST_C = st.number_input(
    "Landsat_LST_C",
    value=float(defaults["Landsat_LST_C"])
)

st.subheader("Radar Features")

S1_VV_dB = st.number_input(
    "S1_VV_dB",
    value=float(defaults["S1_VV_dB"])
)

S1_VH_dB = st.number_input(
    "S1_VH_dB",
    value=float(defaults["S1_VH_dB"])
)

S1_VV_VH_Difference_dB = st.number_input(
    "S1_VV_VH_Difference_dB",
    value=float(defaults["S1_VV_VH_Difference_dB"])
)

st.subheader("Structural Features")

distance_to_lineament_m = st.number_input(
    "distance_to_lineament_m",
    value=float(defaults["distance_to_lineament_m"])
)

lineament_density_1km = st.number_input(
    "lineament_density_1km",
    value=float(defaults["lineament_density_1km"])
)

distance_to_structure_m = st.number_input(
    "distance_to_structure_m",
    value=float(defaults["distance_to_structure_m"])
)

structure_density_1km = st.number_input(
    "structure_density_1km",
    value=float(defaults["structure_density_1km"])
)

st.subheader("Geological Features")

lithology_class_id = st.number_input(
    "lithology_class_id",
    value=int(defaults["lithology_class_id"]),
    step=1
)

geomorphology_class_id = st.number_input(
    "geomorphology_class_id",
    value=int(defaults["geomorphology_class_id"]),
    step=1
)

bhuvan_lulc_class_id = st.number_input(
    "bhuvan_lulc_class_id",
    value=int(defaults["bhuvan_lulc_class_id"]),
    step=1
)

# ------------------------
# Prediction
# ------------------------

if st.button("🚀 Predict Prospectivity"):

    input_df = pd.DataFrame([{
        "S2_B2_Blue": S2_B2_Blue,
        "S2_B3_Green": S2_B3_Green,
        "S2_B4_Red": S2_B4_Red,
        "S2_B5_RedEdge1": S2_B5_RedEdge1,
        "S2_B6_RedEdge2": S2_B6_RedEdge2,
        "S2_B7_RedEdge3": S2_B7_RedEdge3,
        "S2_B8A_NIR": S2_B8A_NIR,
        "S2_B11_SWIR1": S2_B11_SWIR1,
        "S2_B12_SWIR2": S2_B12_SWIR2,
        "S2_NDVI": S2_NDVI,
        "S2_Fe_Oxide_Proxy": S2_Fe_Oxide_Proxy,
        "S2_SWIR_Ratio": S2_SWIR_Ratio,
        "Landsat_LST_C": Landsat_LST_C,
        "S1_VV_dB": S1_VV_dB,
        "S1_VH_dB": S1_VH_dB,
        "S1_VV_VH_Difference_dB": S1_VV_VH_Difference_dB,
        "distance_to_lineament_m": distance_to_lineament_m,
        "lineament_density_1km": lineament_density_1km,
        "distance_to_structure_m": distance_to_structure_m,
        "structure_density_1km": structure_density_1km,
        "lithology_class_id": lithology_class_id,
        "geomorphology_class_id": geomorphology_class_id,
        "bhuvan_lulc_class_id": bhuvan_lulc_class_id
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Probability of Manganese Presence",
        f"{probability:.2%}"
    )

    if prediction == 1:
        st.success("🟢 High Manganese Prospectivity")
    else:
        st.error("🔴 Low Manganese Prospectivity")