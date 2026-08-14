from modelmap.collapse import collapse_repeats
from modelmap.schema import Node


def _node(nid, parent, order, cls="Block", shapes=None):
    return Node(
        id=nid, kind="container" if shapes is None else "linear", cls=cls,
        parent=parent, depth=nid.count(".") + 1 if nid else 0, order=order,
        params=0, weight_shapes=shapes,
    )


def _layer(i, dim):
    return [
        _node(f"m.layers.{i}", "m.layers", i),
        _node(f"m.layers.{i}.fc", f"m.layers.{i}", 0, cls="Linear", shapes={"weight": [dim, dim]}),
    ]


def test_identical_siblings_collapse_to_representative():
    nodes = [_node("", None, 0), _node("m", "", 0), _node("m.layers", "m", 0)]
    for i in range(4):
        nodes += _layer(i, dim=64)
    pruned, repeats = collapse_repeats(nodes)

    assert len(repeats) == 1
    r = repeats[0]
    assert r.parent == "m.layers"
    assert r.representative == "m.layers.0"
    assert r.count == 4
    assert r.members == ["0", "1", "2", "3"]
    ids = {n.id for n in pruned}
    assert "m.layers.0.fc" in ids
    assert "m.layers.1" not in ids and "m.layers.3.fc" not in ids


def test_structurally_different_sibling_breaks_the_run():
    nodes = [_node("", None, 0), _node("m", "", 0), _node("m.layers", "m", 0)]
    for i in range(3):
        nodes += _layer(i, dim=64)
    nodes += _layer(3, dim=128)  # different weight shape → different signature
    pruned, repeats = collapse_repeats(nodes)

    assert len(repeats) == 1 and repeats[0].count == 3
    assert {n.id for n in pruned} >= {"m.layers.3", "m.layers.3.fc"}


def test_short_runs_stay_expanded():
    nodes = [_node("", None, 0), _node("m", "", 0), _node("m.layers", "m", 0)]
    for i in range(2):
        nodes += _layer(i, dim=64)
    pruned, repeats = collapse_repeats(nodes)
    assert repeats == []
    assert len(pruned) == len(nodes)
