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


def test_interleaved_designs_collapse_into_two_non_contiguous_repeats():
    # DeepSeek-V4 style: A A B A B A B A B … — no contiguous run reaches 3,
    # but both designs repeat many times
    nodes = [_node("", None, 0), _node("m", "", 0), _node("m.layers", "m", 0)]
    pattern = [64, 64, 128, 64, 128, 64, 128, 64, 128, 64]
    for i, dim in enumerate(pattern):
        nodes += _layer(i, dim=dim)
    pruned, repeats = collapse_repeats(nodes)

    assert [(r.representative, r.count) for r in repeats] == [("m.layers.0", 6), ("m.layers.2", 4)]
    assert repeats[0].members == ["0", "1", "3", "5", "7", "9"]
    assert repeats[1].members == ["2", "4", "6", "8"]
    ids = {n.id for n in pruned}
    assert {"m.layers.0", "m.layers.2"} <= ids
    assert not ({"m.layers.1", "m.layers.3", "m.layers.4.fc"} & ids)


def test_contiguous_runs_win_and_leftovers_regroup():
    # A A A  B  A A A  B B B  → two A runs + one B run + a lone B (too few to group)
    nodes = [_node("", None, 0), _node("m", "", 0), _node("m.layers", "m", 0)]
    pattern = [64, 64, 64, 128, 64, 64, 64, 128, 128, 128]
    for i, dim in enumerate(pattern):
        nodes += _layer(i, dim=dim)
    _, repeats = collapse_repeats(nodes)
    assert [(r.representative, r.count) for r in repeats] == [("m.layers.0", 3), ("m.layers.4", 3), ("m.layers.7", 3)]


def test_named_siblings_never_collapse():
    # four identical norms (contiguous or not) and query/key/value are roles,
    # not a stack — only numbered siblings fold
    nodes = [_node("", None, 0), _node("blk", "", 0)]
    order = 0
    for name, shape in (("norm_a", [8]), ("norm_b", [8]), ("norm_c", [8]), ("norm_d", [8]), ("query", [8, 8]), ("key", [8, 8]), ("value", [8, 8])):
        nodes.append(_node(f"blk.{name}", "blk", order, cls="Norm" if "norm" in name else "Linear", shapes={"weight": shape}))
        order += 1
    pruned, repeats = collapse_repeats(nodes)
    assert repeats == [] and len(pruned) == len(nodes)


def test_repeats_inside_collapsed_blocks_are_dropped():
    # every layer holds an experts list; only the representative layer's run survives
    nodes = [_node("", None, 0), _node("m", "", 0), _node("m.layers", "m", 0)]
    for i in range(3):
        nodes += [_node(f"m.layers.{i}", "m.layers", i), _node(f"m.layers.{i}.experts", f"m.layers.{i}", 0)]
        for e in range(4):
            nodes.append(_node(f"m.layers.{i}.experts.{e}", f"m.layers.{i}.experts", e, cls="Expert", shapes={"w": [4, 4]}))
    pruned, repeats = collapse_repeats(nodes)
    assert [(r.parent, r.count) for r in repeats] == [("m.layers", 3), ("m.layers.0.experts", 4)]
    assert all(n.id.startswith(("m.layers.0", "m.layers", "m", "")) for n in pruned)
