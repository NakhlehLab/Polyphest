# Polyphest: Fast Polyploid Phylogeny Estimation
## Installation
```
git clone https://github.com/NakhlehLab/Polyphest.git
cd Polyphest

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```
## Usage
```
python Polyphest.py --gene_tree_file PATH_TO_GENE_TREE --consensus_multiset_file PATH_TO_MULTISET --filter_strategy STRATEGY [OPTIONS]
```
### Required Arguments
- `--gene_tree_file`: Path to the input gene tree file. The gene tree should be multi-labeled.
- `--consensus_multiset_file`: Path to the consensus multiset file. This file should list the leaf labels of the species MUL-tree, with each label on a new line. Leaf labels in the gene tree must correspond to elements within this consensus multiset
- `--filter_strategy`: The filtering strategy to be used. Options include `constant`, `percentile`, `median`, and `stddev`.

### Optional Arguments
- `--output_dir`: The directory where the results will be saved. If not specified, the current working directory will be used by default.
- `--use_near_isomorphic`: Enables the use of near isomorphic comparison. Set to `False` by default.
- `--isomorphic_threshold`: Sets the threshold for isomorphic comparison. The default value is `0.2`.
- `--percentile`: Specifies the percentile for percentile filtering. This is required if `--filter_strategy` is set to `percentile`.
- `--constant_value`: Specifies the constant value for constant filtering. This is required if `--filter_strategy` is set to `constant`.

> ### Filtering Strategies
> Polyphest supports four filtering strategies for removing clusters with low frequencies:
> - **Constant Filtering**: Applies a fixed threshold to filter clusters.
> - **Percentile Filtering**: Dynamically sets the threshold based on a specified percentile of the frequency distribution.
> - **Median Filtering**: Uses the median of frequencies multiplied by a scale factor as the threshold.
> - **Standard Deviation (StdDev) Filtering**: Sets the threshold based on the standard deviation from the mean or median of frequencies. 

## Examples
Run Polyphest using the percentile filtering strategy:
```
python Polyphest.py --gene_tree_file ./example/gene_multree.newick --consensus_multiset_file ./example/consensus_multiset.txt --filter_strategy percentile --percentile 75 --use_near_isomorphic True --isomorphic_threshold 0.2
```

Run Polyphest using the constant filtering strategy with a specific threshold value:
```
python Polyphest.py --gene_tree_file ./example/gene_multree.newick --consensus_multiset_file ./example/consensus_multiset.txt --filter_strategy constant --constant_value 0.2 --use_near_isomorphic True --isomorphic_threshold 0.2
```


