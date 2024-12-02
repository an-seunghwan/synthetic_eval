# Synthetic-Eval

**Synthetic-Eval** is a package for the comprehensive evaluation of synthetic tabular datasets.

### 1. Installation
Install using pip:
```
pip install synthetic-eval
```

### 2. Supported Metrics
- **Statistical Fidelity**
  1. KL-Divergence (`KL`)
  2. Goodness-of-Fit (Kolmogorov-Smirnov test & Chi-Squared test) (`GoF`)
  3. Maximum Mean Discrepancy (`MMD`)
  4. Cramer-Wold Distance (`CW`)
  5. (naive) $\alpha$-precision & $\beta$-recall (`alpha_precision`, `beta_recall`)
- **Machine Learning Utility** (classification task) 
  1. Accuracy (`base_cls`, `syn_cls`)
  2. Model Selection Performance (`model_selection`)
  3. Feature Selection Performance (`feature_selection`)
- **Privacy Preservation**
  1. $k$-Anonymization (`Kanon_base`, `Kanon_syn`)
  2. $k$-Map (`KMap`)
  3. Distance to Closest Record (`DCR_RS`, `DCR_RR`, `DCR_SS`)
  4. Attribute Disclosure (`AD`)

### 3. Usage
```python
from synthetic_eval import evaluation
evaluation.evaluate # function for evaluating synthetic data quality
```
- See [example.ipynb](example.ipynb) for detailed example and its results with `loan` dataset.
  - Link for download `loan` dataset: [https://www.kaggle.com/datasets/teertha/personal-loan-modeling](https://www.kaggle.com/datasets/teertha/personal-loan-modeling)

#### Example
- *Please ensure that the target column for the machine learning utility is the last column of the dataset.*
```python
"""import libraries"""
import pandas as pd
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""specify column types"""
data = pd.read_csv('./loan.csv') 
# len(data) # 5,000

"""specify column types"""
data = pd.read_csv('./loan.csv') 
# len(data) # 5,000

"""specify column types"""
continuous_features = [
    'Age',
    'Experience',
    'Income', 
    'CCAvg',
    'Mortgage',
]
categorical_features = [
    'Family',
    'Securities Account',
    'CD Account',
    'Online',
    'CreditCard',
    'Personal Loan', 
]
target = 'Personal Loan' # machine learning utility target column

### the target column should be the last column
data = data[continuous_features + [x for x in categorical_features if x != target] + [target]] 

"""training, test, synthetic datasets"""
data[categorical_features] = data[categorical_features].apply(
        lambda col: col.astype('category').cat.codes) 

train = data.iloc[:2000]
test = data.iloc[2000:4000]
syndata = data.iloc[4000:]

"""load Synthetic-Eval"""
from synthetic_eval import evaluation
results = evaluation.evaluate(
    syndata, train, test, 
    target, continuous_features, categorical_features, device
)

"""print results"""
for x, y in results._asdict().items():
    print(f"{x}: {y:.3f}")
```

### 3. References
  - https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics
  - https://github.com/HLasse/data-centric-synthetic-data