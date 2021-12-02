from functools import lru_cache

from julia.api import Julia

Julia(compiled_modules=False)

from .utils import check_data_format


@lru_cache()
def get_jl_decision_tree(env_path: str):
    from julia import Pkg

    Pkg.activate(env_path)
    from julia import DecisionTree

    return DecisionTree


class graph_rf_model:
    def __init__(
        self,
        DecisionTree,
        adj,
        n_trees=10,
        max_depth=-1,
        min_samples_leaf=5,
        min_samples_split=2,
        min_purity_increase=0.0,
    ):
        self.DecisionTree = DecisionTree
        self.adj = adj
        self.n_trees = round(n_trees)
        self.max_depth = round(max_depth)
        self.min_samples_leaf = round(min_samples_leaf)
        self.min_samples_split = round(min_samples_split)
        self.min_purity_increase = min_purity_increase

    def fit(
        self,
        features,
        labels,
    ):
        self.model = self.DecisionTree.build_forest(
            check_data_format(labels),
            check_data_format(features),
            -1,  # n_subfeatures
            self.n_trees,
            0.7,  # partial_sampling
            self.max_depth,
            self.min_samples_leaf,
            self.min_samples_split,
            self.min_purity_increase,
            adj=self.adj,
        )

    def predict(self, features):
        assert hasattr(self, "model")
        return self.DecisionTree.apply_forest(self.model, check_data_format(features))


class julia_rf_model:
    def __init__(
        self,
        DecisionTree,
        n_trees=10,
        max_depth=-1,
        min_samples_leaf=5,
        min_samples_split=2,
        min_purity_increase=0.0,
    ):
        self.DecisionTree = DecisionTree
        self.n_trees = round(n_trees)
        self.max_depth = round(max_depth)
        self.min_samples_leaf = round(min_samples_leaf)
        self.min_samples_split = round(min_samples_split)
        self.min_purity_increase = min_purity_increase

    def fit(
        self,
        features,
        labels,
    ):
        self.model = self.DecisionTree.build_forest(
            check_data_format(labels),
            check_data_format(features),
            -1,  # n_subfeatures
            self.n_trees,
            0.7,  # partial_sampling
            self.max_depth,
            self.min_samples_leaf,
            self.min_samples_split,
            self.min_purity_increase,
        )

    def predict(self, features):
        assert hasattr(self, "model")
        return self.DecisionTree.apply_forest(self.model, check_data_format(features))
