import abc
from collections import deque
import numpy as np
from typing import Hashable, NamedTuple, List, Any, Callable, Tuple

Node = Hashable


class Position(NamedTuple):
    x: int
    y: int


class Edge(NamedTuple):
    source: Node
    target: Node
    weight: float
    info: Hashable


class AbstractGraph(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_edges(self, node: Node) -> List[Edge]:
        raise NotImplementedError()

    def get_neighbours(self, node: Node) -> List[Node]:
        return [edge.target for edge in self.get_edges(node)]


class Graph(AbstractGraph):
    def __init__(self):
        self._nodes = set()
        self._edges = {}

    def is_valid_node(self, node) -> bool:
        return node in self._nodes

    def add_node(self, node: Node):
        if node not in self._nodes:
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

    def get_neighbours(self, node: Node) -> List[Node]:
        return [edge.target for edge in self.get_edges(node)]


class Agenda(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def next_item(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def add_item(self, item: Any):
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def items(self):
        raise NotImplementedError()

    def __len__(self):
        return 0


class QueueAgenda(Agenda):
    def __init__(self):
        self._queue = deque()

    def next_item(self):
        return self._queue.popleft()

    def add_item(self, item: Any):
        self._queue.append(item)

    def clear(self):
        self._queue.clear()

    def items(self):
        return list(self._queue)

    def remove(self, item):
        self._queue.remove(item)

    def __len__(self):
        return len(self._queue)


class StackAgenda(QueueAgenda):
    def next_item(self):
        return self._queue.pop()


class SearchNode(NamedTuple):
    state: Node
    cost: float
    path: Tuple


def search(graph: AbstractGraph, start: Node, agenda: Agenda, goal_fn: Callable):
    agenda.add_item(SearchNode(start, 0., ()))
    visited = set()
    while len(agenda) > 0:
        next_node = agenda.next_item()
        if goal_fn(next_node):
                return next_node
        visited.add(next_node.state)
        edges = graph.get_edges(next_node.state)
        for edge in edges:
            if edge.target not in visited:
                agenda.add_item(SearchNode(edge.target,
                                           next_node.cost + edge.weight,
                                           next_node.path + (next_node.state,)))
    return None


def bfs(graph: AbstractGraph, start: Node, goal_fn: Callable):
    return search(graph, start, QueueAgenda(), goal_fn)


def dfs(graph: AbstractGraph, start: Node, goal_fn: Callable):
    return search(graph, start, StackAgenda(), goal_fn)


def dijkstra(graph: AbstractGraph, start: Node, goal_fn: Callable):
    agenda = QueueAgenda()  # basic list impl rather than PQ

    start_node = SearchNode(start, 0., ())
    agenda.add_item(start_node)
    dists = {start: 0.}
    visited = set()

    while len(agenda) > 0:
        min_dist = np.inf
        next_node = None

        for n in agenda.items():
            if dists[n.state] < min_dist and n.state not in visited:
                min_dist = dists[n.state]
                next_node = n
        agenda.remove(next_node)
        visited.add(next_node.state)

        if goal_fn(next_node.state):
            return next_node

        edges = graph.get_edges(next_node.state)

        for edge in edges:
            node_dist = dists[edge.target] if edge.target in dists else np.inf
            tot_dist = edge.weight + min_dist
            if tot_dist < node_dist:
                node = SearchNode(
                    edge.target,
                    tot_dist,
                    next_node.path + (next_node.state,))
                dists[edge.target] = tot_dist
                agenda.add_item(node)
    return None

