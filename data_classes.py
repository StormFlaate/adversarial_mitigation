from dataclasses import dataclass, field
import xgboost as xgb

@dataclass
class Metric:
    mean: list
    l1: list
    l2: list
    linf: list


@dataclass
class ActivationResults:
    benign_feature_maps: Metric
    adv_feature_maps: Metric


@dataclass
class ProcessResults:
    before_activation: ActivationResults
    after_activation: ActivationResults
    fooling_rate: float
    benign_dense_layers: list = field(default_factory=list)
    adv_dense_layers: list = field(default_factory=list)


@dataclass
class XGBoostClassifierResults:
    model: xgb.XGBClassifier
    accuracy: float
    tp: float
    tn: float
    fp: float
    fn: float


@dataclass
class Metrics:
    feature_map_mean: float
    feature_map_l1: float
    feature_map_l2: float
    feature_map_linf: float
    activations_mean: float
    activations_l1: float
    activations_l2: float
    activations_linf: float
    dense_layers: float