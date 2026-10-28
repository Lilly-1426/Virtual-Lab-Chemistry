# ============================================================
# EXTRA PROPERTIES  -- paste right after calculate_properties()
# Nothing here duplicates what the lab already calculates.
# ============================================================
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def calculate_extra_properties(mol):
    """Drug-likeness, electronic, stereo, topology and identifier values."""

    mw    = Descriptors.MolWt(mol)
    logp  = Descriptors.MolLogP(mol)
    tpsa  = rdMolDescriptors.CalcTPSA(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    rot   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mr    = Descriptors.MolMR(mol)
    heavy = mol.GetNumHeavyAtoms()

    mh = Chem.AddHs(mol)
    pt = Chem.GetPeriodicTable()

    # ---- Drug-likeness rules ----
    lipinski_viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber_pass    = (rot <= 10) and (tpsa <= 140)
    ghose_pass    = (160 <= mw <= 480 and -0.4 <= logp <= 5.6
                     and 40 <= mr <= 130 and 20 <= mh.GetNumAtoms() <= 70)

    # ---- Element counts (incl. H) ----
    counts = {}
    for a in mh.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1

    # ---- Degree of unsaturation = (2C + 2 + N - H - X) / 2 ----
    C = counts.get("C", 0)
    H = counts.get("H", 0)
    N = counts.get("N", 0) + counts.get("P", 0)
    X = sum(counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    dou = (2 * C + 2 + N - H - X) / 2
    dou = int(dou) if float(dou).is_integer() else dou

    # ---- Hybridization of heavy atoms ----
    hyb = {"SP": 0, "SP2": 0, "SP3": 0}
    for a in mol.GetAtoms():
        h = str(a.GetHybridization())
        if h in hyb:
            hyb[h] += 1

    # ---- Stereochemistry ----
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    ez = sum(1 for b in mol.GetBonds()
             if b.GetBondType() == Chem.BondType.DOUBLE
             and b.GetStereo() != Chem.BondStereo.STEREONONE)

    # ---- Electrons / charge ----
    formal_charge = Chem.GetFormalCharge(mol)
    valence_e = sum(pt.GetNOuterElecs(a.GetAtomicNum()) for a in mh.GetAtoms()) \
                - formal_charge
    hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))

    # ---- Elemental composition, % by mass ----
    total_mass = sum(pt.GetAtomicWeight(a.GetAtomicNum()) for a in mh.GetAtoms())
    composition = {
        sym: round(100 * n * pt.GetAtomicWeight(pt.GetAtomicNumFromSymbol(sym))
                   / total_mass, 2)
        for sym, n in counts.items()
    }

    # ---- Identifiers ----
    try:
        inchi, inchikey = Chem.MolToInchi(mol), Chem.MolToInchiKey(mol)
    except Exception:
        inchi, inchikey = "N/A", "N/A"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "None (acyclic)"

    # ---- Topological indices ----
    dist = Chem.GetDistanceMatrix(mol)
    wiener = int(dist.sum() / 2)

    return {
        "Canonical SMILES": Chem.MolToSmiles(mol),
        "InChI": inchi,
        "InChIKey": inchikey,
        "Murcko Scaffold": scaffold,

        "QED Drug-likeness (0-1)": round(QED.qed(mol), 3),
        "Lipinski Violations": lipinski_viol,
        "Lipinski Rule of Five": "PASS" if lipinski_viol <= 1 else "FAIL",
        "Veber Rule": "PASS" if veber_pass else "FAIL",
        "Ghose Filter": "PASS" if ghose_pass else "FAIL",

        "Degree of Unsaturation": dou,
        "sp Atoms": hyb["SP"],
        "sp2 Atoms": hyb["SP2"],
        "sp3 Atoms": hyb["SP3"],

        "Stereocentres": len(centres),
        "Specified E/Z Double Bonds": ez,
        "Formal Charge": formal_charge,
        "Total Valence Electrons": valence_e,
        "Heteroatoms": hetero,
        "Heteroatom % of Heavy Atoms": round(100 * hetero / heavy, 2) if heavy else 0,

        "Labute ASA (A^2)": round(rdMolDescriptors.CalcLabuteASA(mol), 3),
        "Bertz Complexity": round(Descriptors.BertzCT(mol), 2),
        "Balaban J": round(Descriptors.BalabanJ(mol), 3),
        "Kappa 1 (shape)": round(rdMolDescriptors.CalcKappa1(mol), 3),
        "Wiener Index": wiener,

        "Elemental Composition (% mass)": composition,
    }


def tanimoto_similarity(mol_a, mol_b):
    """Morgan (ECFP4-like) Tanimoto similarity: 0 = unrelated, 1 = identical."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return round(DataStructs.TanimotoSimilarity(
        gen.GetFingerprint(mol_a), gen.GetFingerprint(mol_b)), 3)


def calculate_3d_shape(mol3d):
    """3D shape values. Pass the molecule returned by generate_3d_molecule()."""
    from rdkit.Chem import Descriptors3D
    return {
        "Radius of Gyration (A)": round(Descriptors3D.RadiusOfGyration(mol3d), 3),
        "Asphericity": round(Descriptors3D.Asphericity(mol3d), 3),
        "Eccentricity": round(Descriptors3D.Eccentricity(mol3d), 3),
        "PMI 1": round(Descriptors3D.PMI1(mol3d), 2),
        "PMI 2": round(Descriptors3D.PMI2(mol3d), 2),
        "PMI 3": round(Descriptors3D.PMI3(mol3d), 2),
    }


# ============================================================
# NEW PAGE  -- paste as a new  elif  block before the FOOTER
# (also add "🔬 Advanced Descriptors" to the sidebar radio list)
# ============================================================
#
# elif page == "🔬 Advanced Descriptors":
#
#     st.title("🔬 Advanced Descriptors")
#
#     smiles = st.text_input("SMILES", value=st.session_state.active_smiles,
#                            key="adv_smiles")
#     mol = get_molecule(smiles)
#
#     if mol is None:
#         st.error("Invalid SMILES.")
#     else:
#         extra = calculate_extra_properties(mol)
#         comp = extra.pop("Elemental Composition (% mass)")
#
#         st.subheader("Drug-likeness")
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("QED", extra["QED Drug-likeness (0-1)"])
#         c2.metric("Lipinski", extra["Lipinski Rule of Five"])
#         c3.metric("Veber", extra["Veber Rule"])
#         c4.metric("Ghose", extra["Ghose Filter"])
#
#         st.subheader("All values")
#         st.dataframe(
#             pd.DataFrame({"Property": list(extra.keys()),
#                           "Value": [str(v) for v in extra.values()]}),
#             use_container_width=True, hide_index=True)
#
#         st.subheader("Elemental composition (% by mass)")
#         st.bar_chart(pd.Series(comp))
#
#         st.subheader("Similarity between two molecules")
#         other = st.text_input("Second SMILES", value="CC(=O)Oc1ccccc1C(=O)O",
#                               key="adv_smiles2")
#         mol2 = get_molecule(other)
#         if mol2 is not None:
#             st.metric("Tanimoto similarity", tanimoto_similarity(mol, mol2))
#
#         if st.button("Calculate 3D shape descriptors"):
#             mol3d = generate_3d_molecule(mol)
#             if mol3d is None:
#                 st.error("Could not generate 3D structure.")
#             else:
#                 st.json(calculate_3d_shape(mol3d))
