from typing import Hashable, NamedTuple, List

Node = Hashable


class Position(NamedTuple):
    x: int
    y: int


class Edge(NamedTuple):
    source: Node
    target: Node
    weight: float
    info: Hashable


class Graph:
    def __init__(self):
        self._nodes = set()
        self._edges = {}

    def is_valid_node(self, node) -> bool:
        return node in self._nodes

    def add_node(self, node: Node):
        self._nodes.add(node)
        self._edges[node] = []

    def add_edge(self, node1: Node, node2: Node, weight: float=1., info=None):
        assert self.is_valid_node(node1)
        assert self.is_valid_node(node2)

        self._edges[node1].append(Edge(node1, node2, weight, info))

    def get_nodes(self) -> List[Node]:
        return list(self._nodes)

    def get_edges(self, node: Node) -> List[Edge]:
        return self._edges[node]

    def get_neighbours(self, node):
        return [edge.target for edge in self.get_edges(node)]