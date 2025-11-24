import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from scipy.spatial import procrustes
from imblearn.over_sampling import SMOTE
import pickle
from pathlib import Path

def load_morphofile(filepath):
    names, landmarks = [], []
    current_name = None
    current_lms = []
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(('#', "'#")):
                if current_name and current_lms:
                    landmarks.append(np.array(current_lms))
                    names.append(current_name)
                current_name = s.lstrip("#' ").strip()
                current_lms = []
                continue
            try:
                coords = [float(x) for x in s.split()]
                if len(coords) == 3:
                    current_lms.append(coords)
            except:
                pass
    if current_name and current_lms:
        landmarks.append(np.array(current_lms))
        names.append(current_name)
    print(f"Loaded {len(names)} specimens from {filepath}")
    return names, np.stack(landmarks)

def parse_name(name, bone_keyword):
    parts = name.split('_')
    bone_idx = parts.index(bone_keyword)
    sex = parts[bone_idx - 1]
    side = parts[-1][-1]
    species = '_'.join(parts[:-3])
    return species, sex, side

# ←←← COLAB-FRIENDLY PATHS ←←←
bones = {
    "clavicle": {"file": "/content/MorphoFileClavicle_CLEAN.txt",  "keyword": "clavicle"},
    "scapula":  {" ordin": "/content/MorphoFileScapula_CLEAN.txt",   "keyword": "scapula"},
    "humerus":  {"file": "/content/MorphoFileHumerus_CLEAN.txt",   "keyword": "humerus"}
}

for bone, info in bones.items():
    print(f"\n=== TRAINING {bone.upper()} ===")
    names, landmarks = load_morphofile(info["file"])
    
    species_list, sex_list, side_list = [], [], []
    for n in names:
        sp, sex, side = parse_name(n, info["keyword"])
        species_list.append(sp)
        sex_list.append(sex)
        side_list.append(side)

    mean_shape = landmarks.mean(axis=0)
    aligned = np.zeros_like(landmarks)
    centroid_sizes = []
    for i, lm in enumerate(landmarks):
        current = lm.copy()
        if side_list[i] == 'L':
            current[:, 0] *= -1
        centroid_sizes.append(np.sqrt(np.sum(current**2)))
        _, aligned[i], _ = procrustes(mean_shape, current)
    
    centroid_sizes = np.array(centroid_sizes)
    cs_normalized = (centroid_sizes - centroid_sizes.mean()) / centroid_sizes.std()

    flat = aligned.reshape(len(aligned), -1)
    pca = PCA(n_components=12, random_state=42)
    features_raw = pca.fit_transform(flat)
    features = np.column_stack([features_raw, cs_normalized])

    X_tr, X_te, y_sp_tr, y_sp_te, y_sex_tr, y_sex_te, y_side_tr, y_side_te = train_test_split(
        features, species_list, sex_list, side_list,
        test_size=0.2, random_state=42, stratify=species_list
    )

    le = LabelEncoder()
    y_sp_tr_enc = le.fit_transform(y_sp_tr)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr, y_sp_tr_enc)
    model_species = RandomForestClassifier(n_estimators=1200, class_weight='balanced',
                                          random_state=42, n_jobs=-1)
    model_species.fit(X_res, y_res)
    sp_acc = accuracy_score(le.transform(y_sp_te), model_species.predict(X_te))

    model_sex = RandomForestClassifier(n_estimators=1000, class_weight='balanced',
                                      random_state=42, n_jobs=-1)
    model_sex.fit(X_tr, y_sex_tr)
    sex_acc = accuracy_score(y_sex_te, model_sex.predict(X_te))

    model_side = RandomForestClassifier(n_estimators=800, class_weight='balanced',
                                       random_state=42, n_jobs=-1)
    model_side.fit(X_tr, y_side_tr)
    side_acc = accuracy_score(y_side_te, model_side.predict(X_te))

    cv_sex = cross_val_score(model_sex, features, sex_list, cv=5, scoring='accuracy').mean()
    cv_side = cross_val_score(model_side, features, side_list, cv=5, scoring='accuracy').mean()

    print(f"Holdout → Species: {sp_acc:.1%} | Sex: {sex_acc:.1%} | Side: {side_acc:.1%}")
    print(f"5-fold CV → Sex: {cv_sex:.1%} | Side: {cv_side:.1%}")

    out_dir = Path("models") / bone
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(mean_shape, open(out_dir / f"mean_shape_{bone}.pkl", "wb"))
    pickle.dump(pca, open(out_dir / f"pca_{bone}.pkl", "wb"))
    pickle.dump(model_species, open(out_dir / f"model_species_{bone}.pkl", "wb"))
    pickle.dump(model_sex, open(out_dir / f"model_sex_{bone}.pkl", "wb"))
    pickle.dump(model_side, open(out_dir / f"model_side_{bone}.pkl", "wb"))
    pickle.dump(le, open(out_dir / f"le_species_{bone}.pkl", "wb"))

    print(f"Saved models to {out_dir}\n")

print("ALL DONE — OsteoID.ai is now trained on perfect data.")
print("You can now download the entire 'models' folder and push to GitHub.")
