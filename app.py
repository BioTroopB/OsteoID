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

    # UPDATED: Pre-center the mesh to match GPA data (fixes alignment bias)
    centroid = np.mean(verts, axis=0)
    verts -= centroid

    # UPDATED AUTO-DETECTION: Higher thresholds for typical scan densities (adjust based on your files)
    if bone == "Auto":
        n_verts = len(verts)
        if n_verts < 60000:  # Clavicles often lower density
            bone = "clavicle"
        elif n_verts < 150000:  # Scapulae medium
            bone = "scapula"
        else:  # Humeri higher
            bone = "humerus"

    st.write(f"**Processing as {bone.capitalize()}** ({len(verts):,} vertices)")

    # Load models and mean shape (ignores placeholders.txt)
    mean_shape = pickle.load(open(f"models/{bone}/mean_shape_{bone}.pkl", "rb"))
    model_sex = pickle.load(open(f"models/{bone}/model_sex_{bone}.pkl", "rb"))
    model_side = pickle.load(open(f"models/{bone}/model_side_{bone}.pkl", "rb"))
    model_species = pickle.load(open(f"models/{bone}/model_species_{bone}.pkl", "rb"))
    le_species = pickle.load(open(f"models/{bone}/le_species_{bone}.pkl", "rb"))
    pca = pickle.load(open(f"models/{bone}/pca_{bone}.pkl", "rb"))

    # FIXED ICP: Align sparse mean_shape (template) to dense sample_points (mesh)
    def simple_icp(source, target, max_iterations=50, threshold=1e-6):  # Increased iterations for better convergence
        s = source.copy()  # Start with sparse template
        prev_disp = np.inf
        for _ in range(max_iterations):
            dists = cdist(s, target)  # Distance from template to mesh points
            indices = np.argmin(dists, axis=1)
            correspondences = target[indices]  # Closest mesh points to template (same size as s)

            # UPDATED FIX: Jitter if low variance or duplicates (prevents norm=0 and improves robustness)
            if np.std(correspondences, axis=0).min() < 1e-5 or len(np.unique(correspondences, axis=0)) < source.shape[0]:
                correspondences += np.random.normal(0, 1e-8, correspondences.shape)  # Tiny noise to ensure variance/uniqueness

            # Align s (template) to correspondences (target points)
            _, s, disp = procrustes(correspondences, s)
            if abs(prev_disp - disp) < threshold:
                break
            prev_disp = disp
        return s

    # Sample points for speed (dense subset of mesh)
    landmark_counts = {"clavicle": 7, "scapula": 13, "humerus": 16}
    n_samples = min(20000, len(verts))  # UPDATED: Denser sample (up to 20k) for better matching on complex bones
    sample_idx = np.random.choice(len(verts), size=n_samples, replace=False)
    sample_points = verts[sample_idx]

    # Run ICP: Source=mean_shape (sparse), target=sample_points (dense)
    auto_landmarks = simple_icp(mean_shape, sample_points)

    # Final GPA (both now same shape)
    _, aligned_landmarks, _ = procrustes(mean_shape, auto_landmarks)

    # PCA + Predict
    features = pca.transform(aligned_landmarks.flatten().reshape(1, -1))

    pred_species = le_species.inverse_transform(model_species.predict(features))[0]
    pred_sex = model_sex.predict(features)[0]
    pred_side = model_side.predict(features)[0]

    conf_species = np.max(model_species.predict_proba(features)) * 100
    conf_sex = np.max(model_sex.predict_proba(features)) * 100
    conf_side = np.max(model_side.predict_proba(features)) * 100

    # UPDATED: Low confidence warning (helps diagnose alignment issues)
    if min(conf_species, conf_sex, conf_side) < 60:
        st.warning("Low confidence across predictions—alignment may be poor. Try recentering/scaling the .ply in MeshLab or Blender before upload.")

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
