---
title: OsteoID.ai
emoji: 🦴
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
dockerfile: Dockerfile
pinned: false
---

# OsteoID.ai — AI Primate Shoulder Classifier

**M.A. Project Extension – Kevin P. Klier**  
**STEM-Designated Program (DHS CIP 45.0201)**  
University at Buffalo    
Buffalo Human Evolutionary Morphology Lab (BHEML)

---

## What It Does

Upload any raw 3D shoulder bone (.ply) → AI instantly predicts:  
- **Bone type** (clavicle, humerus, or scapula) — auto-detected  
- **Species** (7 nonhuman primate taxa)  
- **Sex** (M/F)  
- **Side** (L/R)  

Built on **555 3D-scanned nonhuman primate shoulder bones** from my master's project (Homo sapiens excluded per current ethical guidelines).

---

## Data Summary (Non-Human Primates Only)

| Bone     | Specimens | Males | Females | Left | Right |
|----------|-----------|-------|---------|------|-------|
| Clavicle | 185       | 92    | 93      | 69   | 116   |
| Humerus  | 185       | 91    | 94      | 66   | 119   |
| Scapula  | 185       | 101   | 84      | 88   | 97    |
| **Total** | **555**   | **284** | **271** | **223** | **332** |

---

## Accuracy (Holdout Test — November 26, 2025)

| Bone     | Species | Sex   | Side  |
|----------|---------|-------|-------|
| Clavicle | 83.8%   | 64.9% | 89.2% |
| Humerus  | 94.6%   | 64.9% | 81.1% |
| Scapula  | 100.0%  | 64.9% | 91.9% |

> **Note on sex classification**: ~65% accuracy reflects the known low sexual dimorphism in nonhuman primate shoulder girdles — shape alone provides limited signal. This is a biological reality, not a model limitation.

---

## Specimen Collections

Specimens scanned at or loaned from:  
- American Museum of Natural History (AMNH)  
- Neil C. Tappen Collection, Department of Anthropology, University of Minnesota (NCT)  
- Field Museum of Natural History (FMNH)  
- Harvard Museum of Comparative Zoology (MCZ)  
- University at Buffalo Primate Skeletal Collection (UBPSC)  
- Cleveland Museum of Natural History (CMNH)

---

## Committee

- **Noreen von Cramon-Taubadel, Ph.D. (Chair)**  
- **Nicholas J. Holowka, Ph.D.**

---

## Funding & Support

Research conducted at **BHEML**, supported by the **National Science Foundation**.

Data shared with explicit approval from Dr. Noreen von Cramon-Taubadel.

---

## AI Development

- **Code & models**: Kevin P. Klier  
- **AI assistance**: **Grok (xAI)**  

---

## License

- **Code & models**: **MIT License**  
- **Project PDF**: **CC-BY 4.0** — cite Kevin P. Klier if reused

---

*"Bridging biological anthropology and artificial intelligence through geometric morphometrics."*  
— **Kevin P. Klier**  
M.A. Anthropology, University at Buffalo, 2023  

---
