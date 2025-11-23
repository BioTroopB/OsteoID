import streamlit as st
import trimesh
import numpy as np
import pickle
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes

st.set_page_config(page_title="OsteoID.ai", layout="centered")
st.title("OsteoID.ai")
st.markdown("**Primate Shoulder Bone Classifier** — Kevin P. Klier")
st.markdown("Upload any raw .ply file — no landmarking required · Auto-landmarking via ICP")

bone = st.selectbox("Bone type (or Auto-detect)", ["Auto", "clavicle", "scapula", "humerus"])

uploaded_file = st.file_uploader("Upload raw .ply file", type="ply")

if uploaded_file is not None:
    # Load mesh properly
    bytes_data = uploaded_file.getvalue()
    mesh = trimesh.load(trimesh.util.wrap_as_stream(bytes_data), file_type='ply')
    verts = np.asarray(mesh.vertices)

    # FIXED AUTO-DETECTION (tuned to real .ply vertex counts from your scans)
    if bone == "Auto":
        n_verts = len(verts)
        if n_verts < 15000:
            bone = "clavicle"
        elif n_verts < 40000:
            bone = "scapula"
        else:
            bone = "humerus"

    st.write(f"**Processing as {bone.capitalize()}** ({len(verts):,} vertices)")

    # Load models and mean shape (ignores placeholders.txt)
    mean_shape = pickle.load(open(f"models/{bone}/mean_shape_{bone}.pkl", "rb"))
    model_sex = pickle.load(open(f"models/{bone}/model_sex_{bone}.pkl", "rb"))
    model_side = pickle.load(open(f"models/{bone}/model_side_{bone}.pkl", "rb"))
    model_species = pickle.load(open(f"models/{bone}/model_species_{bone}.pkl", "rb"))
    le_species = pickle.load(open(f"models/{bone}/le_species_{bone}.pkl", "rb"))
    pca = pickle.load(open(f"models/{bone}/pca_{bone}.pkl", "rb"))

    # FIXED ICP — proper unpacking (disp is scalar)
    def simple_icp(source, target, max_iterations=30, threshold=1e-6):
        s = source.copy()
        prev_disp = np.inf
        for _ in range(max_iterations):
            dists = cdist(s, target)
            correspondences = target[np.argmin(dists, axis=1)]
            # FIXED: _, s, disp — s gets transformed, disp is scalar disparity
            _, s, disp = procrustes(s, correspondences)
            if abs(prev_disp - disp) < threshold:
                break
            prev_disp = disp
        return s

    # Sample points for speed (match landmark counts + oversample for robust ICP)
    landmark_counts = {"clavicle": 7, "scapula": 13, "humerus": 16}
    n_samples = landmark_counts.get(bone, 10) * 10  # Oversample 10x for better alignment
    sample_idx = np.random.choice(len(verts), size=min(n_samples, len(verts)), replace=False)
    sample_points = verts[sample_idx]

    # Run ICP to auto-landmark
    auto_landmarks = simple_icp(sample_points, mean_shape)

    # Final GPA (align to mean)
    _, aligned_landmarks, _ = procrustes(mean_shape, auto_landmarks)

    # PCA + Predict
    features = pca.transform(aligned_landmarks.flatten().reshape(1, -1))

    pred_species = le_species.inverse_transform(model_species.predict(features))[0]
    pred_sex = model_sex.predict(features)[0]
    pred_side = model_side.predict(features)[0]

    conf_species = np.max(model_species.predict_proba(features)) * 100
    conf_sex = np.max(model_sex.predict_proba(features)) * 100
    conf_side = np.max(model_side.predict_proba(features)) * 100

    st.success(f"**Bone**: {bone.capitalize()}")
    st.success(f"**Species**: {pred_species} ({conf_species:.1f}% confidence)")
    st.success(f"**Sex**: {pred_sex} ({conf_sex:.1f}% confidence)")
    st.success(f"**Side**: {pred_side} ({conf_side:.1f}% confidence)")

    # 3D view
    fig = go.Figure(data=[
        go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], color='lightgray', opacity=0.6, name='Mesh'),
        go.Scatter3d(x=auto_landmarks[:,0], y=auto_landmarks[:,1], z=auto_landmarks[:,2], 
                     mode='markers', marker=dict(size=8, color='red'), name='Auto-landmarks')
    ])
    fig.update_layout(scene_aspectmode='data', height=700)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a raw .ply file to see the magic happen")

st.markdown("---")
st.markdown("Kevin P. Klier | 2023")
st.markdown("Based on M.A. research at University at Buffalo under advisement of Dr. Noreen von Cramon-Taubadel")
st.markdown("Non-human primates only")
