# %%
from collections import namedtuple
from . import metric_stat, metric_MLu, metric_privacy

import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

Metrics = namedtuple(
    "Metrics",
    [
        "KL",
        "GoF",
        "MMD",
        "CW",
        "alpha_precision", 
        "beta_recall",
        "base_cls", 
        "syn_cls",
        "model_selection", 
        "feature_selection",
        "Kanon_base",
        "Kanon_syn",
        "KMap",
        "DCR_RS",
        "DCR_RR",
        "DCR_SS",
        "AD",
    ]
)
#%%
def evaluate(syndata, train, test, target, continuous_features, categorical_features, device="cpu"):
    
    print("\n1. Statistical Fidelity\n")
    
    print("(marginal) KL-Divergence...\n")
    KL = metric_stat.KLDivergence(
        train, syndata, continuous_features)
    
    print("(marginal) Goodness Of Fit...\n")
    GoF = metric_stat.GoodnessOfFit(
        train, syndata, continuous_features)
    
    print("(joint) MMD...\n")
    MMD = metric_stat.MaximumMeanDiscrepancy(
        train, syndata, continuous_features, categorical_features)
    
    print("(joint) Cramer-Wold Distance...\n")
    CW = metric_stat.CramerWoldDistance(
        train, syndata, continuous_features, categorical_features, device)
    
    print("(joint) alpha-precision, beta-recall...\n")
    alpha_precision, beta_recall = metric_stat.naive_alpha_precision_beta_recall(
        train, syndata, continuous_features, categorical_features)
    
    print("\n2. Machine Learning Utility\n")
    
    print("Classification downstream task...\n")
    base_cls, syn_cls, model_selection, feature_selection = metric_MLu.MLu_cls(
        train, test, syndata, target, continuous_features, categorical_features)
    
    print("\n3. Privacy Preservability\n")
    
    print("K-anonimity...\n")
    Kanon_base, Kanon_syn = metric_privacy.kAnonymization(
        train, syndata, continuous_features)
    
    print("K-Map...\n")
    KMap = metric_privacy.kMap(
        train, syndata, continuous_features)
    
    print("Distance to Closest Record...\n")
    DCR_RS, DCR_RR, DCR_SS = metric_privacy.DCR_metric(
        train, syndata, continuous_features, categorical_features)
    
    print("Attribute Disclosure...\n")
    AD = metric_privacy.AttributeDisclosure(
        train, syndata, continuous_features, categorical_features)
    
    return Metrics(
        KL, GoF, MMD, CW, alpha_precision, beta_recall,
        base_cls, syn_cls, model_selection, feature_selection,
        Kanon_base, Kanon_syn, KMap, DCR_RS, DCR_RR, DCR_SS, AD
    )
#%%