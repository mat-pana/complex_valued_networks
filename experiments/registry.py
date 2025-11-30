# models/registry.py
from typing import Callable, Dict, Any
from functools import partial
from networks.RealRNN import RealRNN
from networks.RealFeedForward import RealFeedForward
from networks.ComplexFeedForward import ComplexFeedForward
from networks.DualBranchFusion import DualBranchFusion

REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn

    return deco


@register("RealRNN")
def build_real_rnn(**kw):
    return RealRNN(**kw)


@register("RealFeedForward")
def build_real_ff(**kw):
    return RealFeedForward(**kw)


@register("ComplexFeedForward")
def build_complex_ff(**kw):
    return ComplexFeedForward(**kw)


@register("DualBranchFusion")
def build_dual_branch_fusion(rnn_branch: dict, cv_branch: dict, **head_kw):
    rnn = REGISTRY[rnn_branch["name"]](**rnn_branch["params"])
    cv = REGISTRY[cv_branch["name"]](**cv_branch["params"])
    return DualBranchFusion(rnn_branch=rnn, cv_branch=cv, **head_kw)


def build(config: dict):
    name = config["name"]
    params = config.get("params", {})
    return REGISTRY[name](**params)
