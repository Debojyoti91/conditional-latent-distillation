import pytest
from rdkit import Chem
from latent_materials.aems import mutations as aem
from latent_materials.surfactants import descriptors as surf
from latent_materials.aems import descriptors as aem_desc


@pytest.mark.parametrize(
    "mutation", [aem.add_quatN, aem.add_phosphonium, aem.add_imidazolium]
)
def test_charged_mutations_are_sanitizable_and_pass_gate(mutation):
    product = mutation("CCC")
    mol = Chem.MolFromSmiles(product)
    assert mol is not None
    assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 1
    assert aem.structure_pass(product)


def test_aem_rejects_neutral_and_peroxide_candidates():
    assert not aem.structure_pass("CCCC")
    assert not aem.structure_pass("CCOOCC[N+](C)(C)C")


def test_canonicalization_and_historical_regex_descriptor_semantics():
    assert surf.canon_smiles("OCC") == surf.canon_smiles("CCO")
    assert surf.longest_chain("CCCCOCC") == 4
    assert surf.count_EO("CCOCCO") == 2
    assert surf.count_PO("CC(C)O") == 1
    assert surf.canon_smiles("invalid") is None


def test_aem_graph_chain_and_charge_descriptors():
    mol = Chem.MolFromSmiles("CCCC[N+](C)(C)C")
    assert aem_desc.formal_charge(mol) == 1
    assert aem_desc.longest_carbon_chain_graph(mol) == 4
    assert aem_desc.cationic_site_count(mol) == 1


def test_surfactant_repairs_preserve_valid_heads_and_reject_peroxides():
    from latent_materials.surfactants import mutations

    parent = "CCCCCCCCCCCCOCCOS(=O)(=O)[O-]"
    for operation in [
        mutations.safe_add_EO_before_sulfate,
        mutations.safe_chain_delta_before_first_O,
    ]:
        product = operation(parent)
        assert mutations.valid_canon(product) is not None
        assert mutations.structure_pass(product)
    assert not mutations.structure_pass(mutations.add_EO(parent))
    assert not mutations.structure_pass(mutations.remove_EO(parent))
