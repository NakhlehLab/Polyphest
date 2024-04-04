import collections
import re
import networkx as nx
import numpy as np
import treeswift
import PhyNetPy
from multiset import Multiset, FrozenMultiset
from abc import ABC, abstractmethod
from PhyNetPy.Graph import DAG
from PhyNetPy.Node import Node
from typing import List, Set, Dict, Counter, Union

from polyphest.multree_builder import Cluster


def find_unresolved_node(tree: DAG):
    unresolved = []
    for node in tree.nodes:
        if not tree.out_degrees[node] == 0:
            if tree.out_degrees[node] > 2:
                unresolved.append(node)
    return unresolved


def remove_binary_nodes(net: DAG):
    """Modified based on DAG.prune_excess_nodes()"""

    def prune(net: DAG) -> bool:
        root = net.root()[0]
        q = collections.deque([root])
        net_updated = False

        while q:
            cur = q.pop()  # pop right for bfs

            for neighbor in net.get_children(cur):
                current_node: Node = neighbor
                previous_node: Node = cur
                node_removed = False

                # There could be a chain of nodes with in/out degree = 1. Resolve the whole chain before moving on to search more nodes
                while net.in_degree(current_node) == net.out_degree(current_node) == 1:
                    net.remove_edge([previous_node, current_node])

                    previous_node = current_node
                    temp = net.get_children(current_node)[0]
                    net.remove_node(current_node)
                    current_node = temp
                    node_removed = True

                # We need to connect cur to its new successor
                if node_removed:
                    net.remove_edge([previous_node, current_node])
                    net.add_edges([cur, current_node])
                    # current_node.set_parent([cur])
                    net_updated = True

                # Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(current_node)

        return net_updated

    while True:
        update = prune(net)
        if not update:
            break


def remove_floaters(net: DAG):
    net.remove_floaters()  # this only removes nodes in graph.nodes and with in/out degree = 0
    # some nodes may have in/out degree = 0 but are not in graph.nodes, so we need to remove them manually
    nodes = list(net.in_degrees.keys())
    for nd in nodes:
        if nd not in net.nodes:
            del net.in_degrees[nd]

    nodes = list(net.out_degrees.keys())
    for nd in nodes:
        if nd not in net.nodes:
            del net.out_degrees[nd]


def convert_tree_to_dag(tree):
    graph = DAG()
    tnode_to_nnode = dict()
    edges = []
    for node in tree.traverse_preorder():
        name = str(node.cluster) if hasattr(node, 'cluster') else node.label
        nd = Node(name=name)
        if hasattr(node, 'cluster'):
            nd.add_attribute('cluster', node.cluster)
        tnode_to_nnode[node] = nd

    for node in tree.traverse_postorder():
        if node.parent:
            edges.append([tnode_to_nnode[node.parent], tnode_to_nnode[node]])

    graph.add_nodes(list(tnode_to_nnode.values()))
    graph.add_edges(edges, as_list=True)
    for nd in graph.nodes:
        if nd.get_name() is None or nd.get_name() == "":
            graph.add_uid_node(nd)
    return graph


def clade_to_networkx(network: DAG, node: Node):
    q = collections.deque([node])
    nodes, edges = [node], []

    while len(q) > 0:
        cur = q.pop()

        for neighbor in network.get_children(cur):
            nodes.append(neighbor)
            edges.append((cur, neighbor))
            q.append(neighbor)

    graph = nx.DiGraph()
    for node in nodes:
        if network.out_degree(node) == 0:
            if not node.attribute_value_if_exists('species'):
                raise RuntimeError(f"Node {node} has no attribute 'species'.")
            graph.add_node(node, species=node.attribute_value_if_exists('species'))
        else:
            graph.add_node(node, species='')
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def get_leaves_under_node(network: DAG, node: Node) -> FrozenMultiset:
    q = collections.deque([node])
    leaves = Multiset()

    while len(q) > 0:
        cur = q.popleft()

        if network.out_degree(cur) == 0:
            leaves.add(cur.attribute_value_if_exists('species'))

        for neighbor in network.get_children(cur):
            q.append(neighbor)

    return FrozenMultiset([l for l in leaves])


def init_node_height(network: DAG):
    def update_parent_height(node, height):
        for parent in node.get_parent(return_all=True):
            parent_height = parent.attribute_value_if_exists('height')
            if parent_height is None or parent_height < height:
                parent.add_attribute("height", height)
                update_parent_height(parent, height + 1)

    for leaf in network.get_leaves():
        leaf.add_attribute('height', 0)
        update_parent_height(leaf, 1)


def insert_node_on_edge(tail: Node, head: Node, network: DAG):
    hybrid = Node()
    network.add_uid_node(hybrid)
    network.remove_edge([tail, head])
    network.add_edges([[hybrid, head], [tail, hybrid]], as_list=True)
    head.remove_parent(tail)
    head.add_parent(hybrid)
    return hybrid


def remove_clade(network: DAG, node: Node):
    q = collections.deque([node])
    nodes = [node]
    edges = []
    while q:
        cur = q.pop()
        for neighbor in network.get_children(cur):
            nodes.append(neighbor)
            edges.append([cur, neighbor])
            q.append(neighbor)

    for e in edges:
        network.remove_edge(e)
    for n in nodes:
        try:
            network.remove_node(n)
        except KeyError:
            pass
            # print(f"Node {n.get_name()} not found")



def merge_clades(u: Node, v: Node, network: DAG):
    """
    Remove clade v and make v's parent the parent of u.
    """
    parent = v.get_parent()
    network.remove_edge([parent, v])
    network.add_edges([parent, u])
    u.add_parent(parent)
    if len(u.get_parent(return_all=True)) > 1:
        u.set_is_reticulation(True)

    remove_clade(network, v)


def add_node_name(graph: DAG, node, node_name=None):
    if node_name is None:
        node_name = "UID_" + str(graph.UID)
        graph.UID += 1

    graph.update_node_name(node, node_name)


def print_clusters(clusters):
    string = ""
    if isinstance(clusters, collections.Counter):
        for cluster, count in clusters.most_common():
            if len(cluster) == 1:
                continue
            string += f"{str(cluster)}:{count}\t"
    elif isinstance(clusters, collections.defaultdict):
        string = ""
        cluster_list = []
        for cluster, mul_support in clusters.items():
            for mul, support in mul_support.items():
                cluster_list.append((cluster, mul, support))
        cluster_list = sorted(cluster_list, key=lambda x: (x[2], x[1], len(x[0])), reverse=True)
        for cluster, mul, support in cluster_list:
            string += f"{mul}x{str(cluster)}:{support}\t"
    return string


def taxon_map_func(gene_label):
    text = re.sub(r"\d+([a-z]+)[A,B]", r"\1", gene_label)  # gene label to species label
    # text = re.sub(r'(:\d+\.\d+[eE]?[-+]?\d+)', '', text)  # remove branch lengths
    return text


def convert_to_lexicographic_tree(tree: treeswift, taxon_map_func=lambda x: x):
    node_to_leaves = dict()
    for leaf in tree.traverse_leaves():
        species = taxon_map_func(leaf.get_label())
        leaf.set_label(species)
        node_to_leaves[leaf] = species

    for node in tree.traverse_postorder(leaves=False, internal=True):
        children = []
        for child in node.children:
            min_leaf = node_to_leaves.get(child)
            i = 0
            while i < len(children):
                if min_leaf < node_to_leaves[children[i]]:
                    break
                i += 1
            children.insert(i, child)

        leaves = ""
        for child in children:
            if leaves == "":
                leaves = node_to_leaves[child]
            else:
                leaves += "/" + node_to_leaves[child]

        node.children = children
        node_to_leaves[node] = leaves

    return


def summarize_gene_trees(gene_trees: List[treeswift.Tree], taxon_map_func):
    expression_to_tree = collections.Counter()
    for tree in gene_trees:
        for node in tree.traverse_postorder():
            # node.set_edge_length(None)
            if hasattr(node, 'edge_params'):
                del node.edge_params

        convert_to_lexicographic_tree(tree, taxon_map_func)
        expression_to_tree[tree.newick()] += 1

    unique_trees = [(treeswift.read_tree_newick(newick), count) for newick, count in expression_to_tree.items()]
    return unique_trees


AmbiguousCluster = Dict[FrozenMultiset, Counter[int]]
CoreCluster = Counter[FrozenMultiset]
UnifiedCluster = Counter[Cluster]
ClusterType = Union[AmbiguousCluster, CoreCluster, UnifiedCluster]


class FilteringStrategy(ABC):
    @abstractmethod
    def calculate_threshold(self, frequencies: List[int]) -> float:
        pass

    def filter_clusters(self, clusters: ClusterType) -> ClusterType:
        frequencies = self.get_all_frequencies(clusters)
        threshold = self.calculate_threshold(frequencies)

        if isinstance(clusters, Counter):
            filtered_clusters = collections.Counter(
                {cluster: count for cluster, count in clusters.items() if count >= threshold})
        elif isinstance(clusters, collections.defaultdict):
            filtered_clusters = collections.defaultdict(collections.Counter)
            for cluster, mul_support in clusters.items():
                for mul, support in mul_support.items():
                    if mul * support >= threshold:
                        filtered_clusters[cluster][mul] = support

        return filtered_clusters

    def get_all_frequencies(self, clusters: ClusterType) -> List[int]:
        frequencies = []
        if isinstance(clusters, Counter):  # Core cluster
            frequencies.extend(clusters.values())
        elif isinstance(clusters, collections.defaultdict):  # Ambiguous cluster
            for counter in clusters.values():
                frequencies.extend([mul * support for mul, support in counter.items()])
        return frequencies


class ConstantFiltering(FilteringStrategy):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def calculate_threshold(self, frequencies: List[int]) -> float:
        return self.threshold


class PercentileFiltering(FilteringStrategy):
    def __init__(self, percentile: float, base_threshold: float):
        self.percentile = percentile
        self.base_threshold = base_threshold

    def calculate_threshold(self, frequencies: List[int]) -> float:
        return max(np.percentile(frequencies, self.percentile), self.base_threshold)


class MedianFiltering(FilteringStrategy):
    def __init__(self, scale=1):
        self.scale = scale

    def calculate_threshold(self, frequencies: List[int]) -> float:
        return np.median(frequencies) * self.scale


class StdDevFiltering(FilteringStrategy):
    def __init__(self, num_std_dev=1, from_mean=True):
        self.num_std_dev = num_std_dev
        self.from_mean = from_mean

    def calculate_threshold(self, frequencies: List[int]) -> float:
        if self.from_mean:
            center = np.mean(frequencies)
        else:
            center = np.median(frequencies)
        std_dev = np.std(frequencies)
        return center - (self.num_std_dev * std_dev)
