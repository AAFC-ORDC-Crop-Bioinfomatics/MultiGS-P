# omniGS/models/registry.py

from .brr import BRR
from .svr_rbf import SVR_RBF   
from .rr_blup import RR_BLUP
from .gblup import GBLUP
from .bayesa import BayesA 
from .bayesb import BayesB
from .bayesc import BayesC 
from .bayesr import BayesR
from .rkhs import RKHS
from .elasticnet import ElasticNet
from .lasso import LASSO
from .svr_linear import SVR_LINEAR
from .svr_poly import SVR_POLY
from .krr_linear import KRR_LINEAR
from .krr_rbf import KRR_RBF
from .randomforest import RandomForest
from .xgboost import XGBOOST
from .lightgbm import LIGHTGBM
from .mlp import MLP
from .cnn import CNN
from .mlp_sklearn import MLP_SKLEARN
from .stacking import STACKING

# ---------------------------------------------------------
# Model Registry
# ---------------------------------------------------------
# Maps model names (as they appear in config["enabled_models"])
# to their corresponding Python classes.

MODEL_REGISTRY = {
    "BRR": BRR,
    "SVR_RBF": SVR_RBF,
    "RR_BLUP": RR_BLUP,
    "GBLUP": GBLUP,
    "BAYESA": BayesA,
    "BAYESB": BayesB,
    "BAYESC": BayesC,
    "BAYESR": BayesR,
    "RKHS": RKHS,
    "ELASTICNET": ElasticNet,
    "LASSO": LASSO,
    "SVR_LINEAR": SVR_LINEAR,
    "SVR_POLY": SVR_POLY,
    "KRR_LINEAR": KRR_LINEAR,
    "KRR_RBF": KRR_RBF, 
    "RANDOMFOREST": RandomForest,
    "XGBOOST": XGBOOST,
    "LIGHTGBM": LIGHTGBM,
    "MLP": MLP,
    "CNN": CNN,
    "MLP_SKLEARN": MLP_SKLEARN,
    "STACKING": STACKING, 
}


def get_model_class(name: str):
    """
    Retrieve a model class from the registry by name.
    
    Args:
        name (str): Model name, must match keys in MODEL_REGISTRY.
    
    Returns:
        class: The corresponding model class.
    
    Raises:
        ValueError: If the model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]

