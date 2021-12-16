from functools import lru_cache

from julia.api import Julia

Julia(compiled_modules=False)

from .utils import check_data_format


@lru_cache()
def get_jl_modules(env_path: str):
    from julia import Pkg

    Pkg.activate(env_path)
    from julia import DecisionTree
    from julia import JLD

    return DecisionTree, JLD


class base_model:
    def __init__(
        self,
        DecisionTree,
        JLD,
        features,
        labels,
    ):
        self.DecisionTree = DecisionTree
        self.JLD = JLD
        self.features = check_data_format(features)
        self.labels = check_data_format(labels)

    def predict(self, features):
        assert hasattr(self, "model")
        return self.DecisionTree.apply_forest(self.model, check_data_format(features))

    def save_model(self, path: str, save_features_and_labels: bool = False):
        objects_dict = {
            "model": self.model,
        }
        if save_features_and_labels:
            objects_dict = {
                **objects_dict,
                "features": self.features,
                "labels": self.labels,
            }
        self.JLD.save(path, objects_dict)


class graph_rf_model(base_model):
    def __init__(
        self,
        DecisionTree,
        JLD,
        features,
        labels,
        adj,
        n_trees=10,
        max_depth=-1,
        min_samples_leaf=5,
        min_samples_split=2,
        min_purity_increase=0.0,
        jump_probability=0.0,
    ):
        super().__init__(
            DecisionTree,
            JLD,
            features,
            labels,
        )
        self.adj = adj
        self.n_trees = round(n_trees)
        self.max_depth = round(max_depth)
        self.min_samples_leaf = round(min_samples_leaf)
        self.min_samples_split = round(min_samples_split)
        self.min_purity_increase = min_purity_increase
        self.jump_probability = jump_probability

    def fit(self):
        self.model = self.DecisionTree.build_forest(
            self.labels,
            self.features,
            -1,  # n_subfeatures
            self.n_trees,
            0.7,  # partial_sampling
            self.max_depth,
            self.min_samples_leaf,
            self.min_samples_split,
            self.min_purity_increase,
            self.jump_probability,
            sparse_adj=self.adj,
        )


class julia_rf_model(base_model):
    def __init__(
        self,
        DecisionTree,
        JLD,
        features,
        labels,
        n_trees=10,
        max_depth=-1,
        min_samples_leaf=5,
        min_samples_split=2,
        min_purity_increase=0.0,
    ):
        super().__init__(
            DecisionTree,
            JLD,
            features,
            labels,
        )
        self.n_trees = round(n_trees)
        self.max_depth = round(max_depth)
        self.min_samples_leaf = round(min_samples_leaf)
        self.min_samples_split = round(min_samples_split)
        self.min_purity_increase = min_purity_increase

    def fit(self):
        self.model = self.DecisionTree.build_forest(
            self.labels,
            self.features,
            -1,  # n_subfeatures
            self.n_trees,
            0.7,  # partial_sampling
            self.max_depth,
            self.min_samples_leaf,
            self.min_samples_split,
            self.min_purity_increase,
        )
