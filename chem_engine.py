"""
chem_engine.py
--------------
A small, pure-Python cheminformatics engine (NO RDKit).

It parses SMILES strings (organic subset, bracket atoms, branches, ring
closures, aromatic atoms, charges) and calculates the descriptors used in the
Cheminformatics Virtual Laboratory.

Only numpy is needed (for the 2D layout used by the drawing function).
"""

import re
from collections import Counter, deque

import numpy as np

# ------------------------------------------------------------------
# ATOMIC DATA
# ------------------------------------------------------------------
ATOMIC_WEIGHT = {
    "H": 1.008, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Na": 22.990, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "K": 39.098, "As": 74.922, "Se": 78.971,
    "Br": 79.904, "I": 126.904,
}

EXACT_MASS = {
    "H": 1.007825, "Li": 7.016003, "B": 11.009305, "C": 12.000000,
    "N": 14.003074, "O": 15.994915, "F": 18.998403, "Na": 22.989770,
    "Si": 27.976927, "P": 30.973762, "S": 31.972071, "Cl": 34.968853,
    "K": 38.963707, "As": 74.921595, "Se": 79.916522, "Br": 78.918338,
    "I": 126.904473,
}

# Allowed valences for the "organic subset" (atoms written without brackets)
VALENCES = {
    "B": [3], "C": [4], "N": [3, 5], "O": [2], "P": [3, 5],
    "S": [2, 4, 6], "F": [1], "Cl": [1], "Br": [1], "I": [1],
}

AROMATIC_SYMBOLS = {
    "b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S",
    "se": "Se", "as": "As",
}

HALOGENS = {"F", "Cl", "Br", "I"}

BRACKET_RE = re.compile(
    r"^(\d+)?([A-Za-z][a-z]?)(@{1,2})?(?:H(\d*))?([+-]+|[+-]\d+)?(?::\d+)?$"
)


class SmilesError(ValueError):
    """Raised when a SMILES string cannot be interpreted."""


# ------------------------------------------------------------------
# MOLECULE CLASS
# ------------------------------------------------------------------
class Molecule:
    """A molecule built from a SMILES string."""

    def __init__(self, smiles):
        self.smiles = (smiles or "").strip()
        self.atoms = []   # dict per atom
        self.bonds = []   # (atom_i, atom_j, order)  order: 1, 2, 3, 4 or 1.5
        self._parse()
        self._assign_hydrogens()
        self._build_graph()

    # ---------------- parsing ----------------
    def _parse(self):
        s = self.smiles
        if not s:
            raise SmilesError("Please enter a SMILES string.")

        atoms, bonds = self.atoms, self.bonds
        stack, rings = [], {}
        prev, pending = None, None
        i, n = 0, len(s)

        def add_atom(symbol, aromatic, bracket, h=0, charge=0):
            nonlocal prev, pending
            idx = len(atoms)
            atoms.append({
                "symbol": symbol, "aromatic": aromatic,
                "bracket": bracket, "nH": h, "charge": charge,
            })
            if prev is not None:
                if pending is not None:
                    order = pending
                elif aromatic and atoms[prev]["aromatic"]:
                    order = 1.5
                else:
                    order = 1
                bonds.append((prev, idx, order))
            pending = None
            prev = idx

        while i < n:
            ch = s[i]

            if ch == "[":
                j = s.find("]", i)
                if j < 0:
                    raise SmilesError("Missing closing ']' in SMILES.")
                inner = s[i + 1:j]
                m = BRACKET_RE.match(inner)
                if not m:
                    raise SmilesError(f"Cannot read bracket atom [{inner}].")
                _, sym, _, hcount, charge = m.groups()
                if sym[0].islower():
                    if sym not in AROMATIC_SYMBOLS:
                        raise SmilesError(f"Unknown aromatic atom [{sym}].")
                    aromatic, symbol = True, AROMATIC_SYMBOLS[sym]
                else:
                    aromatic, symbol = False, sym
                if symbol not in ATOMIC_WEIGHT:
                    raise SmilesError(f"Element '{symbol}' is not supported.")
                h = 0 if hcount is None else (int(hcount) if hcount else 1)
                q = 0
                if charge:
                    if charge[1:].isdigit():
                        q = int(charge[1:]) * (1 if charge[0] == "+" else -1)
                    else:
                        q = len(charge) * (1 if charge[0] == "+" else -1)
                add_atom(symbol, aromatic, True, h, q)
                i = j + 1
                continue

            two = s[i:i + 2]
            if two in ("Cl", "Br"):
                add_atom(two, False, False)
                i += 2
            elif ch in "BCNOPSFI":
                add_atom(ch, False, False)
                i += 1
            elif ch in "bcnops":
                add_atom(AROMATIC_SYMBOLS[ch], True, False)
                i += 1
            elif ch == "(":
                if prev is None:
                    raise SmilesError("A branch '(' must follow an atom.")
                stack.append(prev)
                i += 1
            elif ch == ")":
                if not stack:
                    raise SmilesError("Unmatched ')' in SMILES.")
                prev = stack.pop()
                i += 1
            elif ch in "-=#:$":
                pending = {"-": 1, "=": 2, "#": 3, ":": 1.5, "$": 4}[ch]
                i += 1
            elif ch in "/\\":
                i += 1          # stereo bond marks: treated as single bonds
            elif ch == ".":
                prev, pending = None, None
                i += 1
            elif ch.isdigit() or ch == "%":
                if ch == "%":
                    label, i = s[i + 1:i + 3], i + 3
                    if len(label) < 2 or not label.isdigit():
                        raise SmilesError("Invalid ring label after '%'.")
                else:
                    label, i = ch, i + 1
                if prev is None:
                    raise SmilesError("Ring-closure digit must follow an atom.")
                if label in rings:
                    other, order0 = rings.pop(label)
                    if other == prev:
                        raise SmilesError("An atom cannot close a ring with itself.")
                    order = pending if pending is not None else order0
                    if order is None:
                        both = atoms[other]["aromatic"] and atoms[prev]["aromatic"]
                        order = 1.5 if both else 1
                    bonds.append((other, prev, order))
                else:
                    rings[label] = (prev, pending)
                pending = None
            else:
                raise SmilesError(f"Unexpected character '{ch}' in SMILES.")

        if stack:
            raise SmilesError("Unclosed '(' in SMILES.")
        if rings:
            raise SmilesError("Unclosed ring: ring digit " + ", ".join(rings) + " has no partner.")
        if pending is not None:
            raise SmilesError("SMILES ends with a bond symbol.")
        if not atoms:
            raise SmilesError("No atoms found in SMILES.")

    # ---------------- hydrogens & aromatic pi electrons ----------------
    def _assign_hydrogens(self):
        adj = [[] for _ in self.atoms]
        for a, b, o in self.bonds:
            adj[a].append((b, o))
            adj[b].append((a, o))

        for k, at in enumerate(self.atoms):
            sym, deg = at["symbol"], len(adj[k])
            at["deg"] = deg
            exo_double = any(o == 2 for _, o in adj[k])

            # Does this aromatic atom donate one atom to a pi double bond?
            at["pi_arom"] = False
            if at["aromatic"] and not exo_double:
                if sym in ("C", "B"):
                    at["pi_arom"] = True
                elif sym in ("N", "P"):
                    at["pi_arom"] = at["charge"] > 0 or (deg + at["nH"]) == 2

            if at["bracket"]:
                continue          # explicit hydrogens already known

            if sym not in VALENCES:
                raise SmilesError(f"Atom '{sym}' must be written in brackets.")
            bsum = sum(1 if o == 1.5 else o for _, o in adj[k])
            if at["pi_arom"]:
                bsum += 1
            if bsum > max(VALENCES[sym]):
                raise SmilesError(
                    f"Atom {k + 1} ({sym}) has too many bonds for its valence.")
            at["nH"] = 0
            for v in VALENCES[sym]:
                if bsum <= v:
                    at["nH"] = int(v - bsum)
                    break

    def _build_graph(self):
        self.adj = [[] for _ in self.atoms]
        for bi, (a, b, o) in enumerate(self.bonds):
            self.adj[a].append((b, o, bi))
            self.adj[b].append((a, o, bi))

    # ---------------- basic counts ----------------
    @property
    def n_heavy(self):
        return sum(1 for a in self.atoms if a["symbol"] != "H")

    @property
    def n_hydrogens(self):
        return sum(a["nH"] for a in self.atoms) + sum(
            1 for a in self.atoms if a["symbol"] == "H")

    @property
    def n_atoms_total(self):
        return len(self.atoms) + sum(a["nH"] for a in self.atoms)

    @property
    def net_charge(self):
        return sum(a["charge"] for a in self.atoms)

    def element_counts(self):
        c = Counter()
        for a in self.atoms:
            c[a["symbol"]] += 1
            if a["nH"]:
                c["H"] += a["nH"]
        return c

    def formula(self):
        """Molecular formula in Hill notation."""
        c = self.element_counts()
        order = []
        if "C" in c:
            order.append("C")
            if "H" in c:
                order.append("H")
            order += sorted(e for e in c if e not in ("C", "H"))
        else:
            order = sorted(c)
        text = "".join(e + (str(c[e]) if c[e] > 1 else "") for e in order)
        q = self.net_charge
        if q:
            text += ("+" if q > 0 else "-") + (str(abs(q)) if abs(q) > 1 else "")
        return text

    def molecular_weight(self):
        return sum(ATOMIC_WEIGHT[e] * n for e, n in self.element_counts().items())

    def exact_mass(self):
        return sum(EXACT_MASS[e] * n for e, n in self.element_counts().items())

    # ---------------- bonds ----------------
    def pi_bond_count(self):
        pi = 0
        for _, _, o in self.bonds:
            if o == 2:
                pi += 1
            elif o == 3:
                pi += 2
            elif o == 4:
                pi += 3
        pi += sum(1 for a in self.atoms if a["pi_arom"]) // 2
        return pi

    def sigma_bond_count(self):
        # one sigma bond per connection, including bonds to hydrogen
        return len(self.bonds) + sum(a["nH"] for a in self.atoms)

    # ---------------- rings ----------------
    def _components(self, nodes, edges):
        parent = {n: n for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in edges:
            parent[find(a)] = find(b)
        return len({find(n) for n in nodes})

    def ring_count(self):
        """Cyclomatic number = bonds - atoms + connected components."""
        nodes = list(range(len(self.atoms)))
        edges = [(a, b) for a, b, _ in self.bonds]
        return len(edges) - len(nodes) + self._components(nodes, edges)

    def aromatic_ring_count(self):
        """Counts rings formed purely by aromatic bonds (lower-case SMILES)."""
        edges = [(a, b) for a, b, o in self.bonds if o == 1.5]
        if not edges:
            return 0
        nodes = sorted({x for e in edges for x in e})
        return len(edges) - len(nodes) + self._components(nodes, edges)

    def is_ring_bond(self, bidx):
        a, b, _ = self.bonds[bidx]
        seen, dq = {a}, deque([a])
        while dq:
            u = dq.popleft()
            for v, _, bi in self.adj[u]:
                if bi == bidx or v in seen:
                    continue
                if v == b:
                    return True
                seen.add(v)
                dq.append(v)
        return False

    # ---------------- chemistry helpers ----------------
    def _carbonyl_oxygens(self, k):
        """Oxygens double-bonded to atom k."""
        return [v for v, o, _ in self.adj[k]
                if o == 2 and self.atoms[v]["symbol"] == "O"]

    def is_carbonyl_carbon(self, k):
        return self.atoms[k]["symbol"] == "C" and bool(self._carbonyl_oxygens(k))

    def hbond_donors(self):
        return sum(1 for a in self.atoms if a["symbol"] in ("N", "O") and a["nH"] > 0)

    def hbond_acceptors(self):
        # Original Lipinski definition: count of N and O atoms
        return sum(1 for a in self.atoms if a["symbol"] in ("N", "O"))

    def rotatable_bonds(self):
        """Single, non-ring bonds between non-terminal heavy atoms
        (amide C-N bonds and bonds next to triple bonds are excluded)."""
        found = []
        for bi, (a, b, o) in enumerate(self.bonds):
            if o != 1:
                continue
            A, B = self.atoms[a], self.atoms[b]
            if A["symbol"] == "H" or B["symbol"] == "H":
                continue
            if A["deg"] < 2 or B["deg"] < 2:
                continue
            if self.is_ring_bond(bi):
                continue
            if any(oo == 3 for _, oo, _ in self.adj[a] + self.adj[b]):
                continue
            pair = {A["symbol"], B["symbol"]}
            if pair == {"C", "N"}:
                c_idx = a if A["symbol"] == "C" else b
                if self.is_carbonyl_carbon(c_idx):
                    continue   # amide bond
            found.append((a, b))
        return found

    def tpsa(self):
        """Approximate topological polar surface area (Ertl-type fragment
        values for N and O atoms)."""
        total = 0.0
        for k, at in enumerate(self.atoms):
            sym, h, deg, q = at["symbol"], at["nH"], at["deg"], at["charge"]
            orders = [o for _, o, _ in self.adj[k]]
            if sym == "O":
                if q < 0:
                    total += 23.06
                elif h >= 1:
                    total += 20.23
                elif 2 in orders:
                    total += 17.07
                elif at["aromatic"]:
                    total += 13.14
                elif deg == 2:
                    total += 9.23
            elif sym == "N":
                if at["aromatic"]:
                    total += 15.79 if h >= 1 else (4.93 if deg == 3 else 12.89)
                elif q > 0 and any(
                        self.atoms[v]["symbol"] == "O" and o == 2
                        for v, o, _ in self.adj[k]):
                    total += 11.68                       # nitro-type N
                elif 3 in orders:
                    total += 23.79                       # nitrile
                elif 2 in orders:
                    total += 23.85 if h >= 1 else 12.36  # imine
                elif q == 0:
                    total += {2: 26.02, 1: 12.03}.get(h, 3.24)
        return round(total, 2)

    def fraction_csp3(self):
        carbons = [k for k, a in enumerate(self.atoms) if a["symbol"] == "C"]
        if not carbons:
            return 0.0
        sp3 = 0
        for k in carbons:
            if self.atoms[k]["aromatic"]:
                continue
            if all(o == 1 for _, o, _ in self.adj[k]):
                sp3 += 1
        return round(sp3 / len(carbons), 3)

    def degree_of_unsaturation(self):
        c = self.element_counts()
        halogens = sum(c[x] for x in HALOGENS)
        return (2 * c["C"] + 2 + c["N"] + c["P"] - c["H"] - halogens) / 2

    def percent_composition(self):
        mw = self.molecular_weight()
        return {e: round(100 * ATOMIC_WEIGHT[e] * n / mw, 2)
                for e, n in self.element_counts().items()}

    def functional_groups(self):
        """Simple rule-based functional-group detector."""
        g = Counter()
        for k, at in enumerate(self.atoms):
            sym = at["symbol"]
            nbrs = [(v, o) for v, o, _ in self.adj[k]]
            if sym in HALOGENS:
                g["Halogen (" + sym + ")"] += 1
            if sym == "C":
                if self.is_carbonyl_carbon(k):
                    others = [(v, o) for v, o in nbrs if o == 1]
                    acid = ester = amide = False
                    for v, _ in others:
                        s2 = self.atoms[v]["symbol"]
                        if s2 == "O" and (self.atoms[v]["nH"] > 0 or self.atoms[v]["charge"] < 0):
                            acid = True
                        elif s2 == "O" and self.atoms[v]["deg"] == 2:
                            ester = True
                        elif s2 == "N":
                            amide = True
                    if acid:
                        g["Carboxylic acid"] += 1
                    elif ester:
                        g["Ester"] += 1
                    elif amide:
                        g["Amide"] += 1
                    else:
                        g["Carbonyl (aldehyde / ketone)"] += 1
                if any(o == 3 and self.atoms[v]["symbol"] == "N" for v, o in nbrs):
                    g["Nitrile"] += 1
            elif sym == "O" and not at["aromatic"]:
                c_nbrs = [v for v, _ in nbrs]
                if at["nH"] == 1 and at["deg"] == 1 and not any(
                        self.is_carbonyl_carbon(v) for v in c_nbrs):
                    g["Hydroxyl (alcohol / phenol)"] += 1
                elif (at["nH"] == 0 and at["deg"] == 2 and all(
                        self.atoms[v]["symbol"] == "C" and not self.is_carbonyl_carbon(v)
                        for v in c_nbrs)):
                    g["Ether"] += 1
            elif sym == "N" and not at["aromatic"] and at["charge"] == 0:
                if all(o == 1 for _, o in nbrs) and not any(
                        self.is_carbonyl_carbon(v) for v, _ in nbrs):
                    g["Amine"] += 1
        n_ar = self.aromatic_ring_count()
        if n_ar:
            g["Aromatic ring"] += n_ar
        return dict(g)


# ------------------------------------------------------------------
# PUBLIC HELPERS
# ------------------------------------------------------------------
def get_molecule(smiles):
    """Return a Molecule, or None if the SMILES is invalid."""
    try:
        return Molecule(smiles)
    except SmilesError:
        return None


def validate(smiles):
    """Return (Molecule or None, error message or '')."""
    try:
        return Molecule(smiles), ""
    except SmilesError as exc:
        return None, str(exc)


def calculate_properties(mol):
    return {
        "Molecular Formula": mol.formula(),
        "Molecular Weight": round(mol.molecular_weight(), 3),
        "Exact Molecular Weight": round(mol.exact_mass(), 4),
        "TPSA (approx.)": mol.tpsa(),
        "Hydrogen Bond Donors": mol.hbond_donors(),
        "Hydrogen Bond Acceptors": mol.hbond_acceptors(),
        "Rotatable Bonds": len(mol.rotatable_bonds()),
        "Ring Count": mol.ring_count(),
        "Aromatic Rings": mol.aromatic_ring_count(),
        "Heavy Atoms": mol.n_heavy,
        "Total Atoms (incl. H)": mol.n_atoms_total,
        "Heavy Atom Bonds": len(mol.bonds),
        "Sigma (σ) Bonds": mol.sigma_bond_count(),
        "Pi (π) Bonds": mol.pi_bond_count(),
        "Total Bonds (σ + π)": mol.sigma_bond_count() + mol.pi_bond_count(),
        "Degree of Unsaturation": (
            mol.degree_of_unsaturation() if mol.net_charge == 0 else "n/a (charged)"),
        "Fraction Csp3": mol.fraction_csp3(),
        "Net Charge": mol.net_charge,
    }


def drug_likeness(props):
    """Lipinski (MW, HBD, HBA) and Veber (TPSA, rotatable bonds) checks."""
    rules = [
        ("Molecular weight ≤ 500", props["Molecular Weight"], props["Molecular Weight"] <= 500),
        ("H-bond donors ≤ 5", props["Hydrogen Bond Donors"], props["Hydrogen Bond Donors"] <= 5),
        ("H-bond acceptors ≤ 10", props["Hydrogen Bond Acceptors"], props["Hydrogen Bond Acceptors"] <= 10),
        ("TPSA ≤ 140 Å² (Veber)", props["TPSA (approx.)"], props["TPSA (approx.)"] <= 140),
        ("Rotatable bonds ≤ 10 (Veber)", props["Rotatable Bonds"], props["Rotatable Bonds"] <= 10),
    ]
    return rules


def parse_formula_text(text):
    """Turn a typed formula such as 'C9H8O4' into a Counter (or None)."""
    text = (text or "").replace(" ", "").replace("₀", "0")
    if not text or not re.fullmatch(r"([A-Z][a-z]?\d*)+", text):
        return None
    c = Counter()
    for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", text):
        c[el] += int(num) if num else 1
    return c


# ------------------------------------------------------------------
# 2D STRUCTURE DRAWING (matplotlib, no RDKit)
# ------------------------------------------------------------------
ATOM_COLORS = {
    "C": "#222222", "N": "#2F5BFF", "O": "#E53935", "S": "#B8860B",
    "P": "#E67E22", "F": "#2E9E4F", "Cl": "#2E9E4F", "Br": "#8E3B2F",
    "I": "#8E44AD", "H": "#555555",
}


def _layout_2d(mol, tries=10, iters=500):
    """Force-directed 2D layout; the best of several random starts is kept."""
    n = len(mol.atoms)
    if n == 1:
        return np.zeros((1, 2))
    best, best_e = None, np.inf
    for seed in range(tries):
        rng = np.random.default_rng(seed)
        pos = rng.uniform(-1, 1, (n, 2)) * np.sqrt(n) * 0.6
        for it in range(iters):
            delta = pos[:, None, :] - pos[None, :, :]
            dist = np.linalg.norm(delta, axis=2) + np.eye(n)
            force = (delta / dist[..., None] * (0.9 / dist ** 2)[..., None]).sum(axis=1)
            for a, b, _ in mol.bonds:
                d = pos[b] - pos[a]
                length = np.linalg.norm(d) + 1e-9
                pull = (length - 1.0) * 2.5 * d / length
                force[a] += pull
                force[b] -= pull
            cool = 0.12 * (1 - it / iters) + 0.01
            pos += np.clip(force * cool, -0.25, 0.25)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + np.eye(n)
        energy = sum((np.linalg.norm(pos[a] - pos[b]) - 1.0) ** 2 for a, b, _ in mol.bonds)
        energy += np.sum(np.triu(0.3 / dist, 1))
        if energy < best_e:
            best, best_e = pos.copy(), energy
    return best


def _atom_label(at):
    sym, h, q = at["symbol"], at["nH"], at["charge"]
    text = sym
    if h == 1:
        text += "H"
    elif h > 1:
        text += f"H{h}"
    if q:
        text += ("+" if q > 0 else "−") + (str(abs(q)) if abs(q) > 1 else "")
    return text


def draw_structure(mol, title=None, figsize=(7, 5)):
    """Return a matplotlib Figure with a skeletal 2D structure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = _layout_2d(mol)
    # centre and scale
    pos = pos - pos.mean(axis=0)

    labeled = set()
    for k, at in enumerate(mol.atoms):
        if at["symbol"] != "C" or at["charge"] or mol.n_heavy == 1 or at["bracket"]:
            labeled.add(k)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    def shorten(k, p_from, p_to, amount=0.28):
        """Shorten a bond end if the atom carries a text label."""
        if k not in labeled:
            return p_from
        d = p_to - p_from
        length = np.linalg.norm(d) + 1e-9
        return p_from + d / length * amount

    for a, b, o in mol.bonds:
        p1, p2 = pos[a].copy(), pos[b].copy()
        s1, s2 = shorten(a, p1, p2), shorten(b, p2, p1)
        d = s2 - s1
        length = np.linalg.norm(d) + 1e-9
        perp = np.array([-d[1], d[0]]) / length
        if o == 1:
            offsets, styles = [0], ["-"]
        elif o == 2:
            offsets, styles = [-0.06, 0.06], ["-", "-"]
        elif o == 3:
            offsets, styles = [-0.1, 0, 0.1], ["-", "-", "-"]
        elif o == 1.5:
            offsets, styles = [-0.05, 0.08], ["-", "--"]
        else:
            offsets, styles = [0], ["-"]
        for off, ls in zip(offsets, styles):
            inner = 0.12 if (o == 1.5 and ls == "--") else 0.0
            q1, q2 = s1 + d * inner, s2 - d * inner
            ax.plot([q1[0] + perp[0] * off, q2[0] + perp[0] * off],
                    [q1[1] + perp[1] * off, q2[1] + perp[1] * off],
                    ls, color="#333333", lw=2, solid_capstyle="round", zorder=1)

    for k in labeled:
        at = mol.atoms[k]
        ax.text(pos[k][0], pos[k][1], _atom_label(at), ha="center", va="center",
                fontsize=14, fontweight="bold",
                color=ATOM_COLORS.get(at["symbol"], "#444444"), zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    pad = 0.9
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
