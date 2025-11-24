import streamlit as st
import trimesh
import numpy as np
import pickle
import plotly.graph_objects as go
from trimesh.proximity import closest_point
from scipy.spatial import procrustes
import os

# Railway health check (responds instantly to proxy pings)
if st.query_params.get("healthcheck") == "true":
    st.write("OK")
    st.stop()

st.set_page_config(page_title="OsteoID.ai", layout="centered")
st.title("OsteoID.ai")
st.markdown("**Primate Shoulder Bone Classifier — Beta**  \nKevin P. Klier | University at Buffalo")

bone_choice = st.selectbox("Bone type", ["Auto-detect", "clavicle", "scapula", "humerus"])
uploaded_file = st.file_uploader("Upload raw .ply file", type="ply")

if uploaded_file is not None:
    mesh = trimesh.load(trimesh.util.wrap_as_stream(uploaded_file.getvalue()), file_type='ply')
    verts = np.asarray(mesh.vertices)

    if len(verts) < 100:
        st.error("Invalid or empty mesh.")
        st.stop()

    centroid = verts.mean(axis=0)
    verts -= centroid

    if bone_choice == "Auto-detect":
        candidates = {}
        for b in ["clavicle", "scapula", "humerus"]:
            model_path = f"models/{b}"
            if not os.path.exists(model_path):
                continue
            try:
                mean_shape = pickle.load(open(f"{model_path}/mean_shape_{b}.pkl", "rb"))
                pca = pickle.load(open(f"{model_path}/pca_{b}.pkl", "rb"))
                model_sp = pickle.load(open(f"{model_path}/model_species_{b}.pkl", "rb"))
                sample = verts[np.random.choice(len(verts), size=min(10000, len(verts)), replace=False)]
                scaled = sample * (np.linalg.norm(mean_shape) / (np.linalg.norm(sample) + 1e-8))
                feats = pca.transform(scaled[:mean_shape.shape[0]].flatten().reshape(1, -1))
                conf = np.max(model_sp.predict_proba(feats))
                candidates[b] = conf
            except:
                continue
        if candidates:
            bone = max(candidates, key=candidates.get)
            st.info(f"Auto-detected: **{bone.capitalize()}**")
        else:
            st.error("No models found — run train_all_models.py first.")
            st.stop()
    else:
        bone = bone_choice

    model_dir = f"models/{bone}"
    if not os.path.exists(model_dir):
        st.error(f"No model for {bone}. Run train_all_models.py first.")
        st.stop()

    mean_shape    = pickle.load(open(f"{model_dir}/mean_shape_{bone}.pkl", "rb"))
    pca           = pickle.load(open(f"{model_dir}/pca_{bone}.pkl", "rb"))
    model_species = pickle.load(open(f"{model_dir}/model_species_{bone}.pkl", "rb"))
    model_sex     = pickle.load(open(f"{model_dir}/model_sex_{bone}.pkl", "rb"))
    model_side    = pickle.load(open(f"{model_dir}/model_side_{bone}.pkl", "rb"))
    le_species    = pickle.load(open(f"{model_dir}/le_species_{bone}.pkl", "rb"))

    st.write(f"**Processing as {bone.capitalize()}** — {len(verts):,} vertices")

    mesh_cs = np.sqrt(np.sum(verts**2) / len(verts))
    tmpl_cs = np.sqrt(np.sum(mean_shape**2) / len(mean_shape))
    verts = verts * (tmpl_cs / mesh_cs if mesh_cs > 0 else 1.0)

    def align_template(template, mesh_pts, max_iter=40):
        src = template.copy().astype(np.float64)
        for _ in range(max_iter):
            closest, _, _ = closest_point(mesh_pts, src)
            _, src, _ = procrustes(closest, src)
        return src

    sample_pts = verts[np.random.choice(len(verts), size=min(20000, len(verts)), replace=False)]
    aligned_lms = align_template(mean_shape, sample_pts)
    _, aligned_lms, _ = procrustes(mean_shape, aligned_lms)

    feats = pca.transform(aligned_lms.flatten().reshape(1, -1))

    species_pred = model_species.predict(feats)[0]
    species_label = le_species.inverse_transform([species_pred])[0]
    species_proba = model_species.predict_proba(feats)[0]
    conf_sp = np.max(species_proba) * 100

    sex_pred = "Male" if model_sex.predict(feats)[0] == "M" else "Female"
    conf_sex = np.max(model_sex.predict_proba(feats)) * 100

    side_pred = "Left" if model_side.predict(feats)[0] == "L" else "Right"
    conf_side = np.max(model_side.predict_proba(feats)) * 100

    if min(conf_sp, conf_sex, conf_side) < 65:
        st.warning("Low confidence — alignment may be poor. Try re-centering the scan in MeshLab/Blender.")

    st.success(f"**Bone**: {bone.capitalize()}")
    st.success(f"**Species**: {species_label} ({conf_sp:.1f}% confidence)")

    top3_idx = np.argsort(species_proba)[-3:][::-1]
    st.write("**Top 3 species**")
    for i in top3_idx:
        sp = le_species.inverse_transform([i])[0]
        prob = species_proba[i] * 100
        st.write(f"• {sp} — {prob:.1f}%")

    st.success(f"**Sex**: {sex_pred} ({conf_sex:.1f}% confidence)")
    st.success(f"**Side**: {side_pred} ({conf_side:.1f}% confidence)")

    fig = go.Figure(data=[
        go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], color='lightgray', opacity=0.5, name='Bone'),
        go.Scatter3d(x=aligned_lms[:,0], y=aligned_lms[:,1], z=aligned_lms[:,2],
                     mode='markers', marker=dict(size=8, color='red'), name='Auto-landmarks')
    ])
    fig.update_layout(scene_aspectmode='data', height=700)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a raw .ply shoulder bone (clavicle, scapula, or humerus) to begin.")
    st.markdown("*Non-human primates only · No landmarking required*")

st.markdown("---")
st.caption("Based on M.A. research at University at Buffalo under advisement of Dr. Noreen von Cramon-Taubadel")
