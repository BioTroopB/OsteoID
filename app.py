import streamlit as st
import trimesh
import numpy as np
import pickle
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes

st.set_page_config(page_title="OsteoID.ai", layout="centered")
st.title("OsteoID.ai")
st.markdown("**Primate Pectoral Girdle Classifier** — Kevin P. Klier | University at Buffalo BHEML")
st.markdown("Upload any raw .ply file — no landmarking required · Auto-landmarking via ICP")

bone = st.selectbox("Bone type (or Auto-detect)", ["Auto", "clavicle", "scapula", "humerus"])

uploaded_file = st.file_uploader("Upload raw .ply file", type="ply")

if uploaded_file:
    mesh = trimesh.load(uploaded_file)
    verts = np.asarray(mesh.vertices)

    if bone == "Auto":
        if len(verts) < 2000:
            bone = "clavicle"
        elif len(verts) < 5000:
            bone = "scapula"
        else:
            bone = "humerus"

    st.write(f"**Processing as {bone.capitalize()}**")

    # Load template and models
    mean_shape = pickle.load(open(f"models/{bone}/mean_shape_{bone}.pkl", "rb"))
    model_sex = pickle.load(open(f"models/{bone}/model_sex_{bone}.pkl", "rb"))
    model_side = pickle.load(open(f"models/{bone}/model_side_{bone}.pkl", "rb"))
    model_species = pickle.load(open(f"models/{bone}/model_species_{bone}.pkl", "rb"))
    le_species = pickle.load(open(f"models/{bone}/le_species_{bone}.pkl", "rb"))
    pca = pickle.load(open(f"models/{bone}/pca_{bone}.pkl", "rb"))

    # Simple but effective ICP using numpy/scipy (no Open3D needed)
    def simple_icp(source, target, max_iterations=30, tolerance=1e-6):
        src = source.copy()
        for i in range(max_iterations):
            distances = cdist(src, target)
            indices = np.argmin(distances, axis=1)
            corresponding = target[indices]
            # Procrustes alignment
            mtx1, mtx2, disparity = procrustes(corresponding, src)
            src = mtx2
            if disparity < tolerance:
                break
        return src

    # Use downsampled mesh points as source for speed
    sample = verts[np.random.choice(len(verts), size=1000, replace=False)]
    auto_landmarks = simple_icp(sample, mean_shape)

    # Final GPA to training mean shape
    _, aligned_landmarks, _ = procrustes(mean_shape, auto_landmarks)
    features = pca.transform(aligned_landmarks.flatten().reshape(1, -1))

    # Predict
    pred_species = le_species.inverse_transform(model_species.predict(features))[0]
    pred_sex = model_sex.predict(features)[0]
    pred_side = model_side.predict(features)[0]

    conf_species = model_species.predict_proba(features)[0].max() * 100
    conf_sex = model_sex.predict_proba(features)[0].max() * 100
    conf_side = model_side.predict_proba(features)[0].max() * 100

    st.success(f"**Bone**: {bone.capitalize()}")
    st.success(f"**Species**: {pred_species} ({conf_species:.1f}% confidence)")
    st.success(f"**Sex**: {pred_sex} ({conf_sex:.1f}% confidence)")
    st.success(f"**Side**: {pred_side} ({conf_side:.1f}% confidence)")

    # 3D view with auto-landmarks
    fig = go.Figure(data=[
        go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], color='lightgray', opacity=0.6, name='Mesh'),
        go.Scatter3d(x=auto_landmarks[:,0], y=auto_landmarks[:,1], z=auto_landmarks[:,2], mode='markers', marker=dict(size=8, color='red'), name='Auto-landmarks')
    ])
    fig.update_layout(scene_aspectmode='data', height=700)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a .ply file to see it work")

st.markdown("---")
st.markdown("Kevin P. Klier | University at Buffalo BHEML | 2023")
st.markdown("Non-human primates only | 555 specimens | Approved by Dr. Noreen von Cramon-Taubadel")
