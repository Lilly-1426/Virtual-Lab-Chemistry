# 🧪 Cheminformatics Virtual Laboratory (RDKit-free)

SRM Institute of Science and Technology, Tiruchirappalli

## Aim
To build a foundational understanding of cheminformatics by representing,
visualising and analysing chemical structures with computational tools.

## Learning Objectives
Students will be able to
- Interpret and write SMILES notation.
- Visualise molecular structures in 2D and 3D.
- Calculate molecular descriptors by hand and verify them.
- Understand structure–property relationships and drug-likeness.

## Features
| Page | What it does |
|---|---|
| Molecular Visualization | 2D skeletal drawing, functional groups, 3D viewer |
| Molecular Properties Report | MW, TPSA, HBD/HBA, rotatable bonds, rings, σ/π bonds, drug-likeness, CSV/PDF |
| Structure–Property Analysis | Compare many molecules + bar chart |
| **Problem Solver (new)** | Step-by-step worked solutions for any SMILES |
| **Practice Test (new)** | Randomised, auto-graded calculation questions + MCQ quiz |
| Final Project Report | One PDF for many molecules |

## Files
- `app.py` – Streamlit interface
- `chem_engine.py` – pure-Python SMILES parser and descriptor calculator (replaces RDKit)
- `srm_logo.jpeg` – logo shown on the Home page (keep it next to `app.py`)
- `requirements.txt`

`packages.txt` is **no longer needed** (it only held system libraries for RDKit) – delete it.

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Write rings in aromatic (lower-case) SMILES, e.g. caffeine = `Cn1cnc2c1c(=O)n(C)c(=O)n2C`.
- TPSA is an approximate fragment-based value; LogP is not calculated (it needs a
  full atom-typing library).
- The 3D viewer downloads coordinates from PubChem, so it needs internet access.
