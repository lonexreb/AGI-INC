from halo.obs.obs_summarizer import extract_actionable_nodes


def test_extract_actionable_nodes_keyword_recall_flat_nodes():
    top_k = 5
    nodes = []

    for i in range(20):
        nodes.append({"role": "button", "browsergym_id": f"b{i}", "name": f"Button {i}"})

    nodes.append({"role": "button", "browsergym_id": "checkout_bid", "name": "Checkout"})

    for i in range(21, 40):
        nodes.append({"role": "button", "browsergym_id": f"b{i}", "name": f"Button {i}"})

    axtree_object = {"nodes": nodes}
    actionable = extract_actionable_nodes(axtree_object, top_k=top_k)

    assert len(actionable) == top_k
    assert "checkout_bid" in {n["bid"] for n in actionable}


def test_extract_actionable_nodes_keyword_recall_nested_tree():
    top_k = 5

    axtree_object = {
        "role": "root",
        "children": [
            {"role": "button", "bid": "b0", "name": "Button 0"},
            {"role": "button", "bid": "b1", "name": "Button 1"},
            {"role": "button", "bid": "b2", "name": "Button 2"},
            {"role": "button", "bid": "b3", "name": "Button 3"},
            {"role": "button", "bid": "b4", "name": "Button 4"},
            {
                "role": "group",
                "children": [
                    {"role": "button", "bid": "checkout_bid", "name": "Checkout"}
                ],
            },
        ],
    }

    actionable = extract_actionable_nodes(axtree_object, top_k=top_k)

    assert len(actionable) == top_k
    assert "checkout_bid" in {n["bid"] for n in actionable}
