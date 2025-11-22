import streamlit as st
import trimesh
import numpy as np
import pickle
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial import procrustes

st.set_page_config(page_title="OsteoID.ai", layout="centered")
st.title("OsteoID.ai")
st.markdown("**Primate Pectoral Girdle Classifier** — Kevin P. Klier | University at Buffalo BHEML")
st.markdown("Upload any raw .ply file — no landmarking required · Auto-landmarking via template registration")

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

    # Open3D point clouds
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(verts)
    target_pcd.estimate_normals()

    template_pcd = o3d.geometry.PointCloud()
    template_pcd.points = o3d.utility.Vector3dVector(mean_shape)
    template_pcd.estimate_normals()

    # Rough alignment
    target_pcd.translate(-target_pcd.get_center())
    template_pcd.translate(-template_pcd.get_center())
    scale = np.max(target_pcd.get_max_bound() - target_pcd.get_min_bound()) / np.max(mean_shape.ptp(axis=0))
    template_pcd.scale(scale, center=(0,0,0))

    # ICP point-to-plane
    icp_result = o3d.pipelines.registration.registration_icp(
        template_pcd, target_pcd, max_correspondence_distance=30.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    template_pcd.transform(icp_result.transformation)
    auto_landmarks = np.asarray(template_pcd.points)

    # Final GPA
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

    # 3D visualization
    fig = go.Figure(data=[
        go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], color='lightgray', opacity=0.6, name='Mesh'),
        go.Scatter3d(x=auto_landmarks[:,0], y=auto_landmarks[:,1], z=auto_landmarks[:,2], mode='markers', marker=dict(size=8, color='red', symbol='diamond'), name='Auto-landmarks')
    ])
    fig.update_layout(scene_aspectmode='data', height=700)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a raw .ply file to see magic happen")

st.markdown("---")
st.markdown("Kevin P. Klier | University at Buffalo BHEML | November 21, 2025")
st.markdown("Non-human primates only | 555 specimens | Approved for public release by Dr. Noreen von Cramon-Taubadel")