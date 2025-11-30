import os
import pandas as pd

from const import DATA

from eda            import EDA
from feature_impact import FeatureImpactAnalysis

'''

(1) inspect distributions of each feature
(2) compare feature distributions for failure vs non-failure
(3) check correlations
(4) look for obvious thresholds (e.g., failures occur mostly above 310 K process temp)
(5) identify outliers or data-quality issues

'''

'''
(1)

Because these metrics let you understand the shape, center, and spread of each feature's 
distribution in a way that is simple, interpretable, and statistically meaningful. They 
form the minimal set needed to answer whether failures occur under different operating 
conditions.

'''

# Preprocess: 
# -- Get normal and failures
# -- Convert string features to numeric values 
# ---------------------------------------------------------------------
def get_preprocessed_data(raw_data, features):

    # Convert columns to numeric once, before splitting
    features_numeric = raw_data
    for col in features:
        features_numeric[col] = (
            features_numeric[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    # Split into failures and non-failures
    failures = features_numeric[features_numeric["Target"] == 1]    
    normal   = features_numeric[features_numeric["Target"] == 0]

    return failures, normal, features_numeric 

# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    
    # Read raw data
    ext = ".csv"
    files = [f for f in os.listdir(DATA) if f.endswith(ext)]
    data_filepath = os.path.join(DATA, files[0])
    raw_data = pd.read_csv(data_filepath, sep=";")
    
    # Features you want to compare
    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    failures, normal, features_numeric = get_preprocessed_data(raw_data, features)

    eda = EDA()
    eda.get_descriptive_analysis(failures, normal, features_numeric)
    eda.get_plots(failures, normal, features)

    feature_impact_ana = FeatureImpactAnalysis()
    rf, X_test, y_test = feature_impact_ana.random_forrest(features_numeric, features)
    feature_impact_ana.permutation_importance(rf, X_test, y_test, features)

