#%%
"""
Reference:
[1] Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark
- https://github.com/HLasse/data-centric-synthetic-data
"""
#%%
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import spearmanr
#%%
def MLu_cls(train, test, syndata, target, continuous_features, categorical_features):
    ### pre-processing
    continuous = continuous_features
    categorical = [x for x in categorical_features if x != target]
    
    train_ = train.copy()
    test_ = test.copy()
    syndata_ = syndata.copy()
    # continuous: standardization
    scaler = StandardScaler().fit(train_[continuous])
    train_[continuous] = scaler.transform(train_[continuous])
    test_[continuous] = scaler.transform(test_[continuous])
    syndata_[continuous] = scaler.transform(syndata_[continuous])
    # categorical: one-hot encoding
    if len(categorical):
        scaler = OneHotEncoder(drop="first", handle_unknown='ignore').fit(train_[categorical])
        train_ = np.concatenate([
            train_[continuous].values,
            scaler.transform(train_[categorical]).toarray(),
            train_[[target]].values
        ], axis=1)
        test_ = np.concatenate([
            test_[continuous].values,
            scaler.transform(test_[categorical]).toarray(),
            test_[[target]].values
        ], axis=1)
        syndata_ = np.concatenate([
            syndata_[continuous].values,
            scaler.transform(syndata_[categorical]).toarray(),
            syndata_[[target]].values
        ], axis=1)
    else:
        train_ = train_.values
        test_ = test_.values
        syndata_ = syndata_.values
    
    """Baseline"""
    performance = []
    feature_importance = []
    print(f"(Baseline) Classification: Accuracy...")
    for name, clf in [
        ('logit', LogisticRegression(tol=0.001, random_state=42, n_jobs=-1, max_iter=1000)),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('RBF-SVM', SVC(random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('GradBoost', GradientBoostingClassifier(random_state=42)),
        ('AdaBoost', AdaBoostClassifier(random_state=42)),
    ]:
        clf.fit(train_[:, :-1], train_[:, -1])
        pred = clf.predict(test_[:, :-1])
        acc = accuracy_score(test_[:, -1], pred)
        if name in ["RandomForest", "GradBoost", "AdaBoost"]: 
            feature_importance.append(clf.feature_importances_)
        print(f"[{name}] ACC: {acc:.3f}")
        performance.append(acc)

    base_performance = performance
    base_cls = np.mean(performance)
    base_feature_importance = feature_importance
    
    """Synthetic"""
    if len(np.unique(syndata_[:, -1])) == 0:
        return (
            base_cls, 0., 0., 0.
        )
    else:
        performance = []
        feature_importance = []
        print(f"(Synthetic) Classification: Accuracy...")
        for name, clf in [
            ('logit', LogisticRegression(tol=0.001, random_state=42, n_jobs=-1, max_iter=1000)),
            ('KNN', KNeighborsClassifier(n_jobs=-1)),
            ('RBF-SVM', SVC(random_state=42)),
            ('RandomForest', RandomForestClassifier(random_state=42, n_jobs=-1)),
            ('GradBoost', GradientBoostingClassifier(random_state=42)),
            ('AdaBoost', AdaBoostClassifier(random_state=42)),
        ]:
            clf.fit(syndata_[:, :-1], syndata_[:, -1])
            pred = clf.predict(test_[:, :-1])
            acc = accuracy_score(test_[:, -1], pred)
            if name in ["RandomForest", "GradBoost", "AdaBoost"]: 
                feature_importance.append(clf.feature_importances_)
            print(f"[{name}] ACC: {acc:.3f}")
            performance.append(acc)
                
        syn_cls = np.mean(performance)
        model_selection = spearmanr(base_performance, performance).statistic
        feature_selection = []
        for f1, f2 in zip(base_feature_importance, feature_importance):
            feature_selection.append(spearmanr(f1, f2).statistic)
        feature_selection = np.mean(feature_selection)
        
        return (
            base_cls, syn_cls, model_selection, feature_selection
        )
#%%