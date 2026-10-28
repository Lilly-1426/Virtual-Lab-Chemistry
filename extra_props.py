
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    QED, Descriptors, rdMolDescriptors, rdFingerprintGenerator,
    Descriptors3D, AllChem
)
from rdkit.Chem.Scaffolds import MurckoScaffold

st.set_page_config(page_title="Molecular Analyzer", page_icon="🧪")
st.title("🧪 Molecular Property Analyzer")
st.caption("Enter a SMILES string to calculate molecular descriptors using RDKit.")

# ---------- Molecule validation ----------
def get_molecule(smiles):
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None


# ---------- Molecular properties ----------
def calculate_properties(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mr = Descriptors.MolMR(mol)
    heavy = mol.GetNumHeavyAtoms()

    # Explicit hydrogens for atom and electron counts
    mh = Chem.AddHs(mol)
    pt = Chem.GetPeriodicTable()

    counts = {}
    for atom in mh.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1

    # Conventional organic DBE formula.
    # Avoid reporting a potentially misleading result for P/S compounds.
    if any(counts.get(x, 0) for x in ("P", "S")):
        dbe = "N/A (P/S-containing molecule)"
    else:
        c = counts.get("C", 0)
        h = counts.get("H", 0)
        n = counts.get("N", 0)
        x = sum(counts.get(e, 0) for e in ("F", "Cl", "Br", "I"))
        value = (2 * c + 2 + n - h - x) / 2
        dbe = int(value) if value.is_integer() else value

    # Hybridization of heavy atoms
    hybrid = {"SP": 0, "SP2": 0, "SP3": 0}
    for atom in mol.GetAtoms():
        key = str(atom.GetHybridization())
        if key in hybrid:
            hybrid[key] += 1

    # Potential tetrahedral chiral centres, including unassigned centres
    centres = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True
    )

    # Only definite E/Z assignments
    ez = sum(
        1 for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() in (
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOZ,
        )
    )

    formal_charge = Chem.GetFormalCharge(mol)
    valence_electrons = sum(
        pt.GetNOuterElecs(atom.GetAtomicNum())
        for atom in mh.GetAtoms()
    ) - formal_charge

    hetero = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in (1, 6)
    )

    # Average-mass elemental composition
    total_mass = sum(
        pt.GetAtomicWeight(atom.GetAtomicNum())
        for atom in mh.GetAtoms()
    )

    composition = {}
    for symbol, number in counts.items():
        atomic_number = pt.GetAtomicNumber(symbol)
        atomic_weight = pt.GetAtomicWeight(atomic_number)
        composition[symbol] = round(
            100 * number * atomic_weight / total_mass, 2
        )

    violations = sum((
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ))
    veber = rot <= 10 and tpsa <= 140
    ghose = (
        160 <= mw <= 480
        and -0.4 <= logp <= 5.6
        and 40 <= mr <= 130
        and 20 <= mh.GetNumAtoms() <= 70
    )

    try:
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchi, inchikey = "Unavailable", "Unavailable"

    scaffold = (
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        or "None (acyclic)"
    )

    return {
        "Canonical SMILES": Chem.MolToSmiles(mol),
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(mw, 3),
        "LogP": round(logp, 3),
        "TPSA (A²)": round(tpsa, 3),
        "H-Bond Donors": hbd,
        "H-Bond Acceptors": hba,
        "Rotatable Bonds": rot,
        "Molar Refractivity": round(mr, 3),
        "Heavy Atoms": heavy,
        "QED": round(QED.qed(mol), 3),
        "Lipinski Violations": violations,
        "Lipinski Rule": "PASS" if violations <= 1 else "FAIL",
        "Veber Rule": "PASS" if veber else "FAIL",
        "Ghose Filter": "PASS" if ghose else "FAIL",
        "Degree of Unsaturation": dbe,
        "sp Atoms": hybrid["SP"],
        "sp2 Atoms": hybrid["SP2"],
        "sp3 Atoms": hybrid["SP3"],
        "Tetrahedral Stereocentres": len(centres),
        "Unassigned Chiral Centres": sum(
            label == "?" for _, label in centres
        ),
        "Specified E/Z Double Bonds": ez,
        "Formal Charge": formal_charge,
        "Total Valence Electrons": valence_electrons,
        "Heteroatoms": hetero,
        "Heteroatom % of Heavy Atoms": round(
            100 * hetero / heavy, 2
        ) if heavy else 0,
        "Labute ASA (A²)": round(
            rdMolDescriptors.CalcLabuteASA(mol), 3
        ),
        "Bertz Complexity": round(Descriptors.BertzCT(mol), 2),
        "Balaban J": round(Descriptors.BalabanJ(mol), 3),
        "Kappa 1": round(rdMolDescriptors.CalcKappa1(mol), 3),
        "Wiener Index": int(Chem.GetDistanceMatrix(mol).sum() / 2),
        "Murcko Scaffold": scaffold,
        "InChI": inchi,
        "InChIKey": inchikey,
        "Elemental Composition (% mass)": composition,
    }


# ---------- Morgan fingerprint similarity ----------
def similarity(mol1, mol2):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    )
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)
    return round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)


# ---------- 3D descriptors ----------
def calculate_3d(mol):
    mol3d = Chem.AddHs(Chem.Mol(mol))
    status = AllChem.EmbedMolecule(mol3d, randomSeed=42)

    if status != 0:
        raise ValueError("Could not generate a 3D conformer.")

    # Use MMFF when available; otherwise try UFF
    if AllChem.MMFFHasAllMoleculeParams(mol3d):
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=300)
    elif AllChem.UFFHasAllMoleculeParams(mol3d):
        AllChem.UFFOptimizeMolecule(mol3d, maxIters=300)
    else:
        raise ValueError("No supported force-field parameters for this molecule.")

    return {
        "Radius of Gyration (A)": round(
            Descriptors3D.RadiusOfGyration(mol3d), 3
        ),
        "Asphericity": round(Descriptors3D.Asphericity(mol3d), 3),
        "Eccentricity": round(Descriptors3D.Eccentricity(mol3d), 3),
        "PMI 1": round(Descriptors3D.PMI1(mol3d), 2),
        "PMI 2": round(Descriptors3D.PMI2(mol3d), 2),
        "PMI 3": round(Descriptors3D.PMI3(mol3d), 2),
    }


# ---------- Streamlit interface ----------
smiles = st.text_input(
    "Molecule SMILES",
    value="CC(=O)Oc1ccccc1C(=O)O",
    help="Default example: aspirin"
)

mol = get_molecule(smiles)

if mol is None:
    st.error("Invalid SMILES. Please enter a valid molecular SMILES string.")
else:
    try:
        props = calculate_properties(mol)
        composition = props.pop("Elemental Composition (% mass)")

        st.subheader("Drug-likeness")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QED", props["QED"])
        c2.metric("Lipinski", props["Lipinski Rule"])
        c3.metric("Veber", props["Veber Rule"])
        c4.metric("Ghose", props["Ghose Filter"])

        st.subheader("Molecular structure")
        st.code(props["Canonical SMILES"])
        st.caption("2D structure")
        st.image(
            Chem.Draw.MolToImage(mol, size=(450, 300)),
            use_container_width=False
        )

        st.subheader("All molecular properties")
        st.dataframe(
            [{"Property": key, "Value": str(value)}
             for key, value in props.items()],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Elemental composition (% by mass)")
        st.bar_chart(composition)

        st.subheader("Molecule similarity")
        other_smiles = st.text_input(
            "Second molecule SMILES",
            value="CC(=O)Oc1ccccc1C(=O)O",
            key="second_smiles"
        )
        other_mol = get_molecule(other_smiles)

        if other_mol is None:
            st.warning("Enter a valid second SMILES to calculate similarity.")
        else:
            st.metric(
                "Morgan Tanimoto similarity",
                similarity(mol, other_mol)
            )

        st.subheader("3D shape descriptors")
        if st.button("Calculate 3D descriptors"):
            try:
                st.json(calculate_3d(mol))
            except Exception as error:
                st.error(f"3D calculation failed: {error}")

    except Exception as error:
        st.error(f"Property calculation failed: {error}")
