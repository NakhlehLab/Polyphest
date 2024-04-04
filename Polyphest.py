import argparse
import sys
import os
import glob
import re
import time
import treeswift
import pandas as pd
from typing import Dict, List, Set
from multiset import FrozenMultiset

from polyphest import folding, utils, multree_builder


def infer_network(
    gene_trees: List[treeswift.Tree],
    filter_strategy: utils.FilteringStrategy,
    use_near_isomorphic: bool,
    isomorphic_threshold: float,
    use_clusters: bool,
    root_cluster: FrozenMultiset = None,
) -> Dict[str, str]:
    weighted_gene_trees = utils.summarize_gene_trees(gene_trees, lambda x: x)
    mul_tree, clusters = multree_builder.generate_multree(
        weighted_gene_trees, filter_strategy, root_cluster=root_cluster
    )
    mul_tree_newick = re.sub(r"(UID_\d+)", "", mul_tree.newick())
    tree = treeswift.read_tree(mul_tree.newick(), schema="newick")
    dag = utils.convert_tree_to_dag(tree)
    for leaf in dag.get_leaves():
        leaf.add_attribute("species", leaf.get_name())

    folding.convert_multree_to_network(
        dag,
        use_near_isomorphic,
        use_clusters,
        clusters=clusters,
        threshold=isomorphic_threshold,
    )
    for nd in dag.nodes:
        if nd.get_name() is None:
            utils.add_node_name(dag, nd)

    network_newick = dag.newick().replace("#UID_", "#H")
    network_newick = re.sub(r"(UID_\d+)", "", network_newick)
    return {"multree": mul_tree_newick, "network": network_newick}


def label_tree_nodes(tree: treeswift.Tree, label_prefix: str = "I"):
    existing_labels = set()
    count = 0
    for node in tree.traverse_postorder():
        if node.is_leaf():
            continue
        if node.label is not None and node.label != "":
            existing_labels.add(node.label)
        else:
            while f"{label_prefix}{count}" in existing_labels:
                count += 1
            node.label = f"{label_prefix}{count}"
            existing_labels.add(node.label)


def load_gene_trees(file: str, root_cluster: FrozenMultiset):
    gene_trees = treeswift.read_tree(file, schema="newick")
    for t in gene_trees:
        label_tree_nodes(t, label_prefix="IND")
        leaves = [tip.get_label() for tip in t.traverse_leaves()]
        if not root_cluster.issuperset(set(leaves)):
            raise RuntimeError(
                f"Gene tree {t} contains leaves that are not in the consensus multiset.\n{set(leaves).difference(root_cluster)}"
            )

    return gene_trees


def read_consensus_multiset(file):
    with open(file, "r") as file:
        labels = [line.strip() for line in file]

    return FrozenMultiset(labels)


def main(
    output_dir: str,
    gene_tree_file: str,
    consensus_multiset_file: str,
    filter_strategy: utils.FilteringStrategy,
    use_near_isomorphic: bool,
    isomorphic_threshold: float,
    use_clusters=True,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root_cluster = read_consensus_multiset(consensus_multiset_file)
    gene_trees = load_gene_trees(gene_tree_file, root_cluster)
    start_time = time.perf_counter()
    result = infer_network(
        gene_trees,
        filter_strategy,
        use_near_isomorphic,
        isomorphic_threshold,
        use_clusters,
        root_cluster,
    )
    end_time = time.perf_counter()
    name = os.path.basename(gene_tree_file).rsplit(".", 1)[0] + "-polyphest.txt"
    save_file = os.path.join(output_dir, name)
    with open(save_file, "w") as file:
        file.write(f"multree: {result['multree']}\n")
        file.write(f"network: {result['network']}\n")
        file.write(f"Time taken: {end_time - start_time}\n")


def create_filter_strategy(strategy_name, percentile=None, constant_value=None):
    """
    Creates and returns an instance of the specified filtering strategy.

    :param strategy_name: The name of the filtering strategy (e.g., "constant", "percentile").
    :param percentile: The percentile value for the PercentileFiltering strategy.
    :param constant_value: The constant value for the ConstantFiltering strategy.
    :return: An instance of the specified filtering strategy.
    """
    if strategy_name == "constant":
        if constant_value is None:
            raise ValueError("Filter threshold must be provided for the constant filtering strategy.")
        return utils.ConstantFiltering(threshold=constant_value)
    elif strategy_name == "percentile":
        if percentile is None:
            raise ValueError("Percentile must be provided for the percentile filtering strategy.")
        base_threshold = 0.2  # Default or derived from elsewhere if needed
        return utils.PercentileFiltering(percentile=percentile, base_threshold=base_threshold)
    elif strategy_name == "median":
        return utils.MedianFiltering()
    elif strategy_name == "stddev":

        return utils.StdDevFiltering()
    else:
        raise ValueError(f"Unknown filter strategy: {strategy_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Polyphest: Polyploid Phylogeny Estimation.")

    # Core arguments
    parser.add_argument(
        "--output_dir", default=os.getcwd(), help="Output directory for saving results."
    )
    parser.add_argument(
        "--gene_tree_file", required=True, help="File path for the input gene tree."
    )
    parser.add_argument(
        "--consensus_multiset_file",
        required=True,
        help="File path for the consensus multiset.",
    )
    parser.add_argument(
        "--filter_strategy",
        required=True,
        choices=["constant", "percentile", "median", "stddev"],
        help="Filtering strategy to use.",
    )
    parser.add_argument(
        "--use_near_isomorphic",
        type=bool,
        default=False,
        help="Whether to use near isomorphic comparison.",
    )
    parser.add_argument(
        "--isomorphic_threshold",
        type=float,
        default=0.2,
        help="Threshold for isomorphic comparison.",
    )


    # Strategy-specific arguments
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Percentile for 'percentile' filtering. Required if --filter_strategy is 'percentile'.",
    )
    parser.add_argument(
        "--constant_value",
        type=float,
        default=None,
        help="Constant value for 'constant' filtering. Required if --filter_strategy is 'constant'.",
    )

    args = parser.parse_args()

    # Conditional requirement checks
    if args.filter_strategy == "percentile" and args.percentile is None:
        parser.error("--percentile is required when --filter_strategy is 'percentile'.")
    elif args.filter_strategy == "constant" and args.constant_value is None:
        parser.error(
            "--constant_value is required when --filter_strategy is 'constant'."
        )

    return args

if __name__ == "__main__":
    args = parse_args()

    # Create the filter strategy instance
    filter_strategy = create_filter_strategy(
        strategy_name=args.filter_strategy,
        percentile=args.percentile,
        constant_value=args.constant_value
    )

    # Call the main function
    main(
        output_dir=args.output_dir,
        gene_tree_file=args.gene_tree_file,
        consensus_multiset_file=args.consensus_multiset_file,
        filter_strategy=filter_strategy,
        use_near_isomorphic=args.use_near_isomorphic,
        isomorphic_threshold=args.isomorphic_threshold,
    )

