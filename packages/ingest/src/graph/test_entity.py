"""Graph entity behavior tests."""

from src.graph.entity import GraphNode, NodeType


def test_node_name_unwraps_provenance() -> None:
    node = GraphNode(
        id="company:canonical",
        node_type=NodeType.COMPANY,
        data={"name": {"value": "Canonical", "source": "radar", "confidence": 0.5}},
    )

    assert node.name == "Canonical"
