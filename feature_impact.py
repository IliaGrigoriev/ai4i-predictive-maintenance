import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import classification_report

class FeatureImpactAnalysis:

    # Constructor
    def __init__(self):
        pass

    # Random Forrest
    # -------------------------------------------------------------------------------
    def random_forrest(self, features_numeric, features):

        X = features_numeric[features]
        y = features_numeric["Target"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Training
        rf = RandomForestClassifier(
            n_estimators = 200,
            random_state = 42,
            n_jobs =-1
        )
        rf.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = rf.predict(X_test)
        
        # Generate report as dictionary
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        # Convert to DataFrame
        report_df = pd.DataFrame(report_dict).T
        # Round all numeric values to 2 decimals
        report_df = report_df.round(2)
        # Save as CSV (semicolon-separated)
        report_df.to_csv("classification_report.csv", sep=";", index=True)

        print(report_df)

        # 6. Feature impact
        importances = rf.feature_importances_
        fi = (
            pd.DataFrame({"feature": features, "impact": importances})
            .sort_values("impact", ascending=False)
            .reset_index(drop=True)
        )
        print(fi)

        fi.to_csv("feature_impact.csv", index = False, sep = ";")

        # All data for reasoning
        features_numeric["rf_pred"]         = rf.predict(X)
        features_numeric["rf_prob_failure"] = rf.predict_proba(X)[:, 1]

        # Optional: save full dataset with predictions
        features_numeric.to_csv("data_with_rf_predictions.csv", index=False, sep=";")

        return rf, X_test, y_test

    # Permutation importance
    # --------------------------------------------------------------------------------
    def permutation_importance(self, rf, X_test, y_test, features):
        from sklearn.inspection import permutation_importance

        # (1) Compute permutation importance on the test set
        result = permutation_importance(
            rf,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        # (2) Build a sorted DataFrame
        pi = (
            pd.DataFrame({
                "feature": features,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        print(pi)

        # 3. Optional: save for slides
        pi.to_csv("permutation_importance.csv", index=False, sep=";")

