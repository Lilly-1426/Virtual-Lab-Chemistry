"""
Cheminformatics Virtual Laboratory  (RDKit-free version)
SRM Institute of Science and Technology, Tiruchirappalli

All chemistry is done by chem_engine.py (pure Python), so the app deploys on
Streamlit Cloud with nothing more than the packages in requirements.txt.

NEW in this version
  * 🧮 Problem Solver  - step-by-step worked solutions for any SMILES
  * 🎯 Practice Test   - randomised, auto-graded calculation questions
  * Drug-likeness check, functional-group detector, % composition
"""

import os
import random
import urllib.parse
import urllib.request
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from chem_engine import (ATOMIC_WEIGHT, Molecule, calculate_properties,
                         draw_structure, drug_likeness, parse_formula_text,
                         validate)

try:
    import py3Dmol
    HAS_3DMOL = True
except Exception:          # py3Dmol is optional
    HAS_3DMOL = False

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Cheminformatics Virtual Laboratory",
                   page_icon="🧪", layout="wide")

st.markdown("""
<style>
.main-title {text-align:center; font-size:40px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SHARED STATE
# ============================================================
EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}

# Larger bank used by the Practice Test
TEST_BANK = {
    **EXAMPLE_MOLECULES,
    "Methane": "C", "Ethene": "C=C", "Ethyne": "C#C",
    "Acetone": "CC(C)=O", "Toluene": "Cc1ccccc1", "Aniline": "Nc1ccccc1",
    "Benzoic Acid": "OC(=O)c1ccccc1", "Pyridine": "c1ccncc1",
    "Naphthalene": "c1ccc2ccccc2c1", "Glycine": "NCC(=O)O",
    "Acetonitrile": "CC#N", "Chlorobenzene": "Clc1ccccc1",
    "Cyclohexane": "C1CCCCC1", "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Propan-2-ol": "CC(C)O", "Ethyl Acetate": "CCOC(C)=O",
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CCO"
if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Ethanol"


def load_shared_example():
    st.session_state.active_smiles = EXAMPLE_MOLECULES[st.session_state.selected_example]


def molecule_input(prefix):
    """SMILES box + example loader shared by all pages. Returns (smiles, mol)."""
    col1, col2 = st.columns([2, 1])
    with col1:
        smiles = st.text_input("Enter SMILES", key="active_smiles")
    with col2:
        st.selectbox("Choose Example Molecule", list(EXAMPLE_MOLECULES),
                     key="selected_example")
        st.button("Load Example", on_click=load_shared_example,
                  key=f"{prefix}_load_example")
    mol, err = validate(smiles)
    if mol is None:
        st.error(f"❌ Invalid SMILES: {err}")
    else:
        st.success("✅ Valid molecular structure")
        st.session_state.latest_smiles = smiles
    return smiles, mol


def fig_to_png_bytes(fig, dpi=200):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


# ============================================================
# 3D VIEW (coordinates fetched from PubChem - no RDKit needed)
# ============================================================
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_pubchem_sdf(smiles):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
           + urllib.parse.quote(smiles, safe="") + "/SDF?record_type=3d")
    req = urllib.request.Request(url, headers={"User-Agent": "ChemInfoVirtualLab/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


def display_3d(sdf, style, show_surface):
    view = py3Dmol.view(width=800, height=520)
    view.addModel(sdf, "sdf")
    styles = {
        "Ball and Stick": {"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}},
        "Stick": {"stick": {"radius": 0.20}},
        "Space Filling": {"sphere": {"scale": 1.0}},
        "Wireframe": {"line": {"linewidth": 2}},
    }
    view.setStyle(styles[style])
    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.7})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=540)


# ============================================================
# WORKED SOLUTION (used by Problem Solver and Practice Test)
# ============================================================
def worked_solution(mol):
    """Return a list of (title, markdown_text, DataFrame or None) steps."""
    steps = []
    props = calculate_properties(mol)

    # Step 1 - atoms
    rows = []
    for k, a in enumerate(mol.atoms, start=1):
        rows.append({"Atom": k, "Element": a["symbol"],
                     "Aromatic": "yes" if a["aromatic"] else "no",
                     "Heavy-atom neighbours": a["deg"],
                     "Attached H": a["nH"]})
    steps.append((
        "Step 1 – Read the SMILES and list every atom",
        f"`{mol.smiles}` contains **{mol.n_heavy} heavy atoms**. "
        "Each atom's hydrogens are found from its valence "
        "(C = 4, N = 3, O = 2, halogens = 1): "
        "*attached H = valence − bonds already used*.",
        pd.DataFrame(rows)))

    # Step 2 - formula
    counts = mol.element_counts()
    steps.append((
        "Step 2 – Molecular formula",
        f"Add up the atoms of each element (Hill order: C, H, then alphabetical) → "
        f"**{mol.formula()}**",
        pd.DataFrame([{"Element": e, "Count": n} for e, n in
                      sorted(counts.items(), key=lambda x: (x[0] != "C", x[0] != "H", x[0]))])))

    # Step 3 - molecular weight
    rows, total = [], 0.0
    for e, n in sorted(counts.items(), key=lambda x: (x[0] != "C", x[0] != "H", x[0])):
        sub = n * ATOMIC_WEIGHT[e]
        total += sub
        rows.append({"Element": e, "Count": str(n), "Atomic weight": str(ATOMIC_WEIGHT[e]),
                     "Count × weight": f"{sub:.3f}"})
    rows.append({"Element": "TOTAL", "Count": "", "Atomic weight": "",
                 "Count × weight": f"{total:.3f}"})
    steps.append((
        "Step 3 – Molecular weight",
        f"Molecular weight = Σ (number of atoms × atomic weight) = **{round(total, 3)} g/mol**",
        pd.DataFrame(rows)))

    # Step 4 - sigma / pi
    n_bonds, n_h = len(mol.bonds), sum(a["nH"] for a in mol.atoms)
    steps.append((
        "Step 4 – σ and π bonds",
        f"* Every connection is one σ bond: {n_bonds} heavy-atom bonds + {n_h} C/N/O–H bonds "
        f"= **{props['Sigma (σ) Bonds']} σ bonds**\n"
        f"* A double bond has 1 π bond, a triple bond has 2, and every aromatic ring "
        f"contributes its π bonds (benzene = 3) → **{props['Pi (π) Bonds']} π bonds**\n"
        f"* Total bond components = {props['Sigma (σ) Bonds']} + {props['Pi (π) Bonds']} "
        f"= **{props['Total Bonds (σ + π)']}**", None))

    # Step 5 - degree of unsaturation
    c = counts
    hal = sum(c[x] for x in ("F", "Cl", "Br", "I"))
    if mol.net_charge == 0:
        steps.append((
            "Step 5 – Degree of unsaturation (rings + π bonds)",
            f"DoU = (2C + 2 + N − H − X) / 2 = (2×{c['C']} + 2 + {c['N']} − {c['H']} − {hal}) / 2 "
            f"= **{mol.degree_of_unsaturation():g}**\n\n"
            f"Check: rings ({props['Ring Count']}) + π bonds ({props['Pi (π) Bonds']}) "
            f"= {props['Ring Count'] + props['Pi (π) Bonds']}", None))

    # Step 6 - hydrogen bonding & flexibility
    rot = mol.rotatable_bonds()
    steps.append((
        "Step 6 – Hydrogen bonding, flexibility and polarity",
        f"* **H-bond donors** = N/O atoms that carry H → **{props['Hydrogen Bond Donors']}**\n"
        f"* **H-bond acceptors** = number of N and O atoms (Lipinski) → "
        f"**{props['Hydrogen Bond Acceptors']}**\n"
        f"* **Rotatable bonds** = single, non-ring bonds between two non-terminal "
        f"heavy atoms (amide C–N excluded) → **{len(rot)}**\n"
        f"* **Rings** = bonds − atoms + components = {len(mol.bonds)} − {len(mol.atoms)} "
        f"+ {mol._components(list(range(len(mol.atoms))), [(a, b) for a, b, _ in mol.bonds])} "
        f"= **{props['Ring Count']}**\n"
        f"* **TPSA** (sum of polar N/O fragment contributions) ≈ **{props['TPSA (approx.)']} Å²**", None))

    return steps


def render_steps(steps):
    for title, text, df in steps:
        with st.expander(title, expanded=True):
            st.markdown(text)
            if df is not None:
                st.dataframe(df, hide_index=True)


# ============================================================
# PDF REPORT (1 or many molecules)
# ============================================================
def build_pdf(student, reg, records):
    """records: list of (smiles, Molecule). Returns PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=38, rightMargin=38,
                            topMargin=40, bottomMargin=40)
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=19, leading=23,
                           alignment=TA_CENTER, textColor=HexColor("#17365D"))
    head = ParagraphStyle("h", parent=ss["Heading1"], fontSize=13, leading=16,
                          textColor=HexColor("#17365D"), spaceBefore=6, spaceAfter=6)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12,
                          alignment=TA_JUSTIFY)
    small = ParagraphStyle("s", parent=ss["BodyText"], fontSize=8, leading=10)
    hdr = ParagraphStyle("hd", parent=ss["BodyText"], fontSize=8, leading=10,
                         textColor=colors.white, alignment=TA_CENTER)

    story = [
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title),
        Paragraph("MOLECULAR ANALYSIS REPORT", head),
        Paragraph(f"Student: {student or 'Not provided'} &nbsp;&nbsp; "
                  f"Registration No.: {reg or 'Not provided'} &nbsp;&nbsp; "
                  f"Molecules: {len(records)}", body),
        Spacer(1, 0.15 * inch),
    ]

    for idx, (smiles, mol) in enumerate(records, start=1):
        if idx > 1:
            story.append(PageBreak())
        props = calculate_properties(mol)
        story.append(Paragraph(f"Molecule {idx}: {props['Molecular Formula']}", head))

        info = [["Input SMILES", smiles],
                ["Molecular Formula", props["Molecular Formula"]],
                ["Molecular Weight", f"{props['Molecular Weight']} g/mol"]]
        t = Table(info, colWidths=[1.8 * inch, 4.9 * inch])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
                               ("PADDING", (0, 0), (-1, -1), 5)]))
        story += [t, Spacer(1, 0.1 * inch)]

        fig = draw_structure(mol, figsize=(6, 3.6))
        png = fig_to_png_bytes(fig, dpi=170)
        w, h = fig.get_size_inches()
        plt.close(fig)
        story += [Image(png, width=3.8 * inch, height=3.8 * inch * h / w),
                  Spacer(1, 0.1 * inch)]

        rows = [[Paragraph("<b>Property</b>", hdr), Paragraph("<b>Value</b>", hdr)]]
        for k, v in props.items():
            rows.append([Paragraph(str(k), small), Paragraph(str(v), small)])
        pt = Table(rows, colWidths=[3.2 * inch, 3.2 * inch], repeatRows=1)
        pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
                                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                ("PADDING", (0, 0), (-1, -1), 3)]))
        story += [pt, Spacer(1, 0.1 * inch)]

        story.append(Paragraph("Drug-likeness check", head))
        rows = [[Paragraph("<b>Rule</b>", hdr), Paragraph("<b>Value</b>", hdr),
                 Paragraph("<b>Result</b>", hdr)]]
        for rule, val, ok in drug_likeness(props):
            rows.append([Paragraph(rule, small), Paragraph(str(val), small),
                         Paragraph("Pass" if ok else "Fail", small)])
        dt = Table(rows, colWidths=[3.2 * inch, 1.6 * inch, 1.6 * inch])
        dt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
                                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                                ("PADDING", (0, 0), (-1, -1), 3)]))
        story.append(dt)

    story += [Spacer(1, 0.2 * inch),
              Paragraph("Descriptors were calculated with a pure-Python SMILES engine "
                        "(no RDKit). TPSA is an approximate fragment-based value. "
                        "Computed values may differ from experimental measurements.", body)]
    doc.build(story)
    return buf.getvalue()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🧪 Virtual Lab")
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📖 Theory",
    "🧬 Molecular Visualization",
    "📊 Molecular Properties Report",
    "📈 Structure–Property Analysis",
    "🧮 Problem Solver",
    "🎯 Practice Test",
    "📄 Final Project Report",
])
st.sidebar.divider()
st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number")

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "srm_logo.jpeg")
    if os.path.exists(logo_path):
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.image(logo_path)

    st.markdown('<div class="main-title">🧪 Cheminformatics Virtual Laboratory</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; font-size:24px; font-weight:700; color:#1f4e79;">
    SRM Institute of Science and Technology, Tiruchirappalli</div>
    <div style="text-align:center; font-size:18px; font-weight:600; margin-top:5px;">
    Faculty of Engineering and Technology | School of Sciences | Division of Chemistry</div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("## Welcome to the Virtual Laboratory")
    st.write("This interactive laboratory helps students learn how chemical structures "
             "are represented, drawn and analysed by computers.")
    st.markdown("""
    ✅ Enter and validate SMILES notation  
    ✅ Draw 2D structures and view 3D models  
    ✅ Calculate molecular properties and check drug-likeness  
    ✅ Compare several molecules  
    🆕 **Problem Solver** – see every calculation step by step  
    🆕 **Practice Test** – randomised, auto-graded questions  
    ✅ Download CSV / PDF reports
    """)

    st.markdown("### Example Molecules")
    st.dataframe(pd.DataFrame({"Molecule": list(EXAMPLE_MOLECULES),
                               "SMILES": list(EXAMPLE_MOLECULES.values())}),
                 hide_index=True)
    st.divider()
    st.markdown("### 👨‍🏫 Developed by")
    st.markdown("**Dr. R. Subramanian**  \nAssistant Professor  \n"
                "Division of Chemistry, School of Sciences  \n"
                "Faculty of Engineering and Technology  \n"
                "SRM Institute of Science and Technology, Tiruchirappalli")
    st.info("📧 **For queries, contact:** rsmani84@gmail.com")

# ============================================================
# THEORY
# ============================================================
elif page == "📖 Theory":
    st.title("📖 Introduction to Cheminformatics")
    st.markdown("""
## What is Cheminformatics?
Cheminformatics combines **Chemistry ⚗️**, **Computer Science 💻** and **Data Analysis 📊**
to store, process, visualise and analyse chemical information.

## SMILES
**S**implified **M**olecular **I**nput **L**ine **E**ntry **S**ystem – a text string describing a structure.

| Symbol | Meaning | Example |
|---|---|---|
| Capital letters | Aliphatic atoms | `CCO` = ethanol |
| Lower-case letters | Aromatic atoms | `c1ccccc1` = benzene |
| `=` and `#` | Double and triple bonds | `C=C`, `C#N` |
| `( )` | Branches | `CC(=O)O` = acetic acid |
| Digits | Ring closures | `C1CCCCC1` = cyclohexane |
| `[ ]` | Charged / special atoms | `[NH4+]`, `[O-]` |

## How the descriptors are calculated
| Descriptor | Idea |
|---|---|
| Molecular weight | Σ (atoms × atomic weight), hydrogens included |
| σ bonds | One per connection, including C–H bonds |
| π bonds | Double bond = 1, triple = 2, benzene ring = 3 |
| Degree of unsaturation | (2C + 2 + N − H − X) / 2 = rings + π bonds |
| H-bond donors / acceptors | N/O with H / count of N and O |
| Rotatable bonds | Single, non-ring bonds between non-terminal atoms |
| TPSA | Sum of polar N/O fragment contributions (Å²) |
| Fraction Csp³ | sp³ carbons ÷ total carbons |

## Drug-likeness rules
**Lipinski:** MW ≤ 500, H-bond donors ≤ 5, H-bond acceptors ≤ 10.  
**Veber:** TPSA ≤ 140 Å², rotatable bonds ≤ 10.

> 💡 Write ring systems in **aromatic (lower-case) SMILES**, e.g. caffeine as
> `Cn1cnc2c1c(=O)n(C)c(=O)n2C`, so aromatic rings are recognised.
""")

# ============================================================
# VISUALISATION
# ============================================================
elif page == "🧬 Molecular Visualization":
    st.title("🧬 Molecular Visualization Laboratory")
    smiles, mol = molecule_input("vis")

    if mol:
        props = calculate_properties(mol)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("2D Molecular Structure")
            fig = draw_structure(mol)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Schematic skeletal drawing: dashed inner line = aromatic bond; "
                       "carbon atoms are not labelled.")
        with c2:
            st.subheader("Molecular Information")
            st.write("**Formula:**", props["Molecular Formula"])
            st.write("**Molecular weight:**", f"{props['Molecular Weight']} g/mol")
            st.write("**Heavy atoms:**", props["Heavy Atoms"])
            st.write("**Total atoms (incl. H):**", props["Total Atoms (incl. H)"])
            st.write("**σ bonds:**", props["Sigma (σ) Bonds"])
            st.write("**π bonds:**", props["Pi (π) Bonds"])
            groups = mol.functional_groups()
            st.write("**Functional groups:**",
                     ", ".join(f"{k} ×{v}" for k, v in groups.items()) or "None detected")

        st.divider()
        st.subheader("🌐 Interactive 3D Structure")
        if not HAS_3DMOL:
            st.info("Install `py3Dmol` to enable the 3D viewer.")
        else:
            st.caption("3D coordinates are downloaded from PubChem, so this needs an "
                       "internet connection and a compound that exists in PubChem.")
            style = st.selectbox("Select Molecular Model",
                                 ["Ball and Stick", "Stick", "Space Filling", "Wireframe"])
            surface = st.checkbox("Show van der Waals surface")
            if st.button("🚀 Generate 3D Structure"):
                with st.spinner("Fetching 3D coordinates..."):
                    sdf = fetch_pubchem_sdf(mol.smiles)
                if sdf:
                    display_3d(sdf, style, surface)
                else:
                    st.error("Could not obtain 3D coordinates for this molecule "
                             "(no internet access, or the structure is not in PubChem).")

# ============================================================
# PROPERTIES REPORT
# ============================================================
elif page == "📊 Molecular Properties Report":
    st.title("📊 Complete Molecular Properties Report")
    smiles, mol = molecule_input("prop")

    if mol:
        props = calculate_properties(mol)
        st.session_state.latest_properties = props

        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = draw_structure(mol)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            st.subheader("📋 Basic Information")
            st.write("**Input SMILES:**", smiles)
            st.write("**Molecular formula:**", props["Molecular Formula"])
            st.write("**Heavy atoms:**", props["Heavy Atoms"])
            st.write("**σ / π bonds:**",
                     f"{props['Sigma (σ) Bonds']} / {props['Pi (π) Bonds']}")
            comp = mol.percent_composition()
            st.write("**% composition:**",
                     ", ".join(f"{e} {p}%" for e, p in comp.items()))

        st.divider()
        st.subheader("📊 Important Molecular Properties")
        m = st.columns(4)
        m[0].metric("Molecular Weight", f"{props['Molecular Weight']} g/mol")
        m[1].metric("TPSA (approx.)", f"{props['TPSA (approx.)']} Å²")
        m[2].metric("Ring Count", props["Ring Count"])
        m[3].metric("Fraction Csp3", props["Fraction Csp3"])
        m = st.columns(4)
        m[0].metric("H-Bond Donors", props["Hydrogen Bond Donors"])
        m[1].metric("H-Bond Acceptors", props["Hydrogen Bond Acceptors"])
        m[2].metric("Rotatable Bonds", props["Rotatable Bonds"])
        m[3].metric("Degree of Unsaturation", props["Degree of Unsaturation"])

        # ---- NEW: drug-likeness ----
        st.divider()
        st.subheader("💊 Drug-likeness Check (Lipinski + Veber)")
        rules = drug_likeness(props)
        st.dataframe(pd.DataFrame([{"Rule": r, "Value": v, "Result": "✅ Pass" if ok else "❌ Fail"}
                                   for r, v, ok in rules]), hide_index=True)
        passed = sum(ok for _, _, ok in rules)
        if passed == len(rules):
            st.success("All rules satisfied – this molecule looks drug-like.")
        else:
            st.warning(f"{passed}/{len(rules)} rules satisfied – "
                       "this molecule may have poor oral bioavailability.")

        st.divider()
        st.subheader("📑 Complete Property Table")
        df = pd.DataFrame([["Input SMILES", smiles]] + [[k, str(v)] for k, v in props.items()],
                          columns=["Property", "Value"])
        st.dataframe(df, hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "Molecular_Properties_Report.csv", "text/csv")
        if d2.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                pdf = build_pdf(student_name, registration_number, [(smiles, mol)])
            st.download_button("⬇️ Download PDF", pdf, "Molecular_Properties_Report.pdf",
                               "application/pdf")

# ============================================================
# STRUCTURE–PROPERTY ANALYSIS
# ============================================================
elif page == "📈 Structure–Property Analysis":
    st.title("📈 Structure–Property Analysis")
    st.write("Compare several molecules. Enter one per line as **Name, SMILES**.")
    default = "\n".join(f"{n},{s}" for n, s in list(EXAMPLE_MOLECULES.items())[:5])
    text = st.text_area("Enter Molecules", value=default, height=200)

    rows = []
    for line in text.splitlines():
        if "," not in line:
            continue
        name, smi = line.split(",", 1)
        mol = validate(smi)[0]
        if mol:
            p = calculate_properties(mol)
            rows.append({"Molecule": name.strip(), "Formula": p["Molecular Formula"],
                         "MW": p["Molecular Weight"], "TPSA": p["TPSA (approx.)"],
                         "HBD": p["Hydrogen Bond Donors"], "HBA": p["Hydrogen Bond Acceptors"],
                         "Rotatable Bonds": p["Rotatable Bonds"], "Ring Count": p["Ring Count"],
                         "σ": p["Sigma (σ) Bonds"], "π": p["Pi (π) Bonds"]})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        # ---- NEW: quick chart ----
        st.subheader("📊 Compare a property")
        prop = st.selectbox("Property to plot",
                            ["MW", "TPSA", "HBD", "HBA", "Rotatable Bonds", "Ring Count", "σ", "π"])
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar(df["Molecule"], df[prop], color="#2F5597")
        ax.set_ylabel(prop)
        ax.set_title(f"{prop} of the selected molecules")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.download_button("⬇️ Download Comparison Table (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           "structure_property_comparison.csv", "text/csv")
    else:
        st.warning("Please enter at least one valid molecule as: Name, SMILES")

# ============================================================
# NEW: PROBLEM SOLVER
# ============================================================
elif page == "🧮 Problem Solver":
    st.title("🧮 Problem Solver – Step-by-Step Solutions")
    st.write("Enter any SMILES and see **how** each value is obtained, so you can "
             "check your own hand calculations.")
    smiles, mol = molecule_input("solver")

    if mol:
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = draw_structure(mol, figsize=(6, 4))
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            props = calculate_properties(mol)
            st.metric("Formula", props["Molecular Formula"])
            st.metric("Molecular weight", f"{props['Molecular Weight']} g/mol")

        render_steps(worked_solution(mol))

# ============================================================
# NEW: PRACTICE TEST
# ============================================================
elif page == "🎯 Practice Test":
    st.title("🎯 Practice Test")
    tab_calc, tab_mcq = st.tabs(["🧮 Calculation Test (auto-graded)", "📝 Quick MCQ Quiz"])

    # ---------------- calculation test ----------------
    with tab_calc:
        QUESTION_TYPES = {
            "formula": "Write the molecular formula (e.g. C2H6O)",
            "mw": "Calculate the molecular weight in g/mol (±0.5 accepted)",
            "sigma": "How many σ bonds are present? (include C–H, O–H, N–H bonds)",
            "pi": "How many π bonds are present?",
            "hbd": "How many hydrogen-bond donors?",
            "hba": "How many hydrogen-bond acceptors? (count of N + O atoms)",
            "rings": "How many rings are present?",
            "dou": "What is the degree of unsaturation?",
        }

        def make_test(seed, n_questions=6):
            rng = random.Random(seed)
            names = rng.sample(list(TEST_BANK), n_questions)
            types = rng.sample(list(QUESTION_TYPES), n_questions)
            test = []
            for name, qtype in zip(names, types):
                mol = Molecule(TEST_BANK[name])
                p = calculate_properties(mol)
                answer = {"formula": mol.formula(), "mw": p["Molecular Weight"],
                          "sigma": p["Sigma (σ) Bonds"], "pi": p["Pi (π) Bonds"],
                          "hbd": p["Hydrogen Bond Donors"], "hba": p["Hydrogen Bond Acceptors"],
                          "rings": p["Ring Count"], "dou": p["Degree of Unsaturation"]}[qtype]
                test.append({"name": name, "smiles": TEST_BANK[name],
                             "qtype": qtype, "answer": answer})
            return test

        if "test_seed" not in st.session_state:
            st.session_state.test_seed = random.randrange(10 ** 6)
        if st.button("🔄 Generate a New Test"):
            st.session_state.test_seed = random.randrange(10 ** 6)
            st.session_state.pop("test_result", None)

        seed = st.session_state.test_seed
        test = make_test(seed)
        st.caption(f"Test ID: {seed}  •  Use the 🧮 Problem Solver page to check "
                   "your method after submitting.")

        user_answers = []
        for i, q in enumerate(test, start=1):
            st.markdown(f"**Q{i}. {q['name']}** — `{q['smiles']}`")
            ans = st.text_input(QUESTION_TYPES[q["qtype"]], key=f"test_{seed}_{i}")
            user_answers.append(ans)
            st.write("")

        def check(q, text):
            text = (text or "").strip()
            if not text:
                return False
            if q["qtype"] == "formula":
                got = parse_formula_text(text)
                return got is not None and got == parse_formula_text(q["answer"])
            try:
                value = float(text)
            except ValueError:
                return False
            tol = 0.5 if q["qtype"] == "mw" else 1e-6
            return abs(value - float(q["answer"])) <= tol

        if st.button("✅ Submit Test"):
            if not any(a.strip() for a in user_answers):
                st.warning("Please answer at least one question.")
            else:
                results = [check(q, a) for q, a in zip(test, user_answers)]
                st.session_state.test_result = {
                    "seed": seed, "results": results, "answers": user_answers}

        res = st.session_state.get("test_result")
        if res and res["seed"] == seed:
            score = sum(res["results"])
            pct = 100 * score / len(test)
            st.divider()
            st.subheader(f"Score: {score}/{len(test)}  ({pct:.0f}%)")
            if pct >= 80:
                st.balloons()
                st.success("🎉 Excellent performance!")
            elif pct >= 50:
                st.info("👍 Good – review the questions you missed.")
            else:
                st.warning("📚 Please revise the Theory page and try again.")

            rows = []
            for i, (q, ok, a) in enumerate(zip(test, res["results"], res["answers"]), start=1):
                rows.append({"Q": i, "Molecule": q["name"], "SMILES": q["smiles"],
                             "Question": q["qtype"], "Your answer": a,
                             "Correct answer": q["answer"], "Result": "✅" if ok else "❌"})
                if not ok:
                    with st.expander(f"Q{i} – worked solution for {q['name']}"):
                        st.markdown(f"**Correct answer: {q['answer']}**")
                        render_steps(worked_solution(Molecule(q["smiles"])))
            result_df = pd.DataFrame(rows)
            st.dataframe(result_df, hide_index=True)
            out = result_df.copy()
            out.insert(0, "Student", student_name)
            out.insert(1, "Registration No.", registration_number)
            st.download_button("⬇️ Download My Results (CSV)",
                               out.to_csv(index=False).encode("utf-8"),
                               f"practice_test_{seed}.csv", "text/csv")

    # ---------------- MCQ quiz ----------------
    with tab_mcq:
        questions = [
            ("What is the full form of SMILES?",
             ["Simplified Molecular Input Line Entry System",
              "Standard Molecular Information Language Encoding System",
              "Simple Molecular Identification and Labeling System"], 0),
            ("Which SMILES represents ethanol?", ["CCO", "c1ccccc1", "CC(=O)O"], 0),
            ("Which SMILES represents benzene?", ["c1ccccc1", "CCO", "O=C=O"], 0),
            ("In SMILES, parentheses are mainly used to indicate:",
             ["Branches", "Molecular weight", "Atom colour"], 0),
            ("What does TPSA describe?",
             ["Topological Polar Surface Area", "Total Pi Surface Area",
              "Thermal Property Surface Analysis"], 0),
            ("What does HBD mean?",
             ["Hydrogen Bond Donor", "Heavy Bond Descriptor", "Hydrogen Bond Distance"], 0),
            ("A double bond contains how many π bonds?", ["1", "2", "0"], 0),
            ("A triple bond contains how many π bonds?", ["2", "1", "3"], 0),
            ("Lipinski's rule says molecular weight of an oral drug should be:",
             ["≤ 500", "≥ 1000", "exactly 180"], 0),
            ("Lower-case letters in SMILES (c, n, o) indicate:",
             ["Aromatic atoms", "Charged atoms", "Isotopes"], 0),
        ]
        picks = []
        for i, (q, opts, correct) in enumerate(questions):
            shuffled = random.Random(i).sample(opts, len(opts))   # fixed order per question
            picks.append((st.radio(q, shuffled, index=None, key=f"mcq_{i}"), opts[correct]))
        if st.button("Submit MCQ Quiz"):
            if any(p is None for p, _ in picks):
                st.warning("Please answer all questions first.")
            else:
                score = sum(p == c for p, c in picks)
                st.success(f"Your score: {score}/{len(picks)}  ({100 * score / len(picks):.0f}%)")
                for i, ((p, c), (q, _, _)) in enumerate(zip(picks, questions), start=1):
                    if p != c:
                        st.write(f"❌ Q{i}: {q} → correct answer: **{c}**")

# ============================================================
# FINAL REPORT
# ============================================================
elif page == "📄 Final Project Report":
    st.title("📄 Final Student Molecular Analysis Report")
    st.write("Enter one SMILES per line. All valid molecules go into a single PDF.")
    name = st.text_input("Student Name for Report", value=student_name, key="final_name")
    reg = st.text_input("Registration Number for Report", value=registration_number, key="final_reg")
    text = st.text_area("Enter SMILES (one per line)",
                        value=st.session_state.get("latest_smiles", "CCO"), height=150)

    records, bad = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        mol = validate(line)[0]
        (records.append((line, mol)) if mol else bad.append(line))

    if bad:
        st.warning("These SMILES could not be read and will be skipped: " + ", ".join(bad))
    if records:
        st.success(f"{len(records)} valid molecule(s) ready.")
        st.dataframe(pd.DataFrame([
            {"No.": i, "SMILES": s, "Formula": m.formula(),
             "MW (g/mol)": round(m.molecular_weight(), 3)}
            for i, (s, m) in enumerate(records, start=1)]), hide_index=True)
        if st.button("📄 Generate PDF for All Molecules"):
            with st.spinner("Generating PDF..."):
                pdf = build_pdf(name, reg, records)
            st.download_button("⬇️ Download Report PDF", pdf,
                               "Cheminformatics_Report.pdf", "application/pdf")
    else:
        st.info("Enter at least one valid SMILES to generate the report.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("🧪 Cheminformatics Virtual Laboratory | Developed using Python and Streamlit")
