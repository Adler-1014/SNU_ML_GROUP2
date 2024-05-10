# Assuming CustomModel class is already defined and imported
model_types = ['random_forest', 'linear_regression', 'svr', 'xgboost', 'gradient_boosting', 'decision_tree', 'knn', 'elastic_net']

# Dictionary to store the results
results = {}

# Iterate over each model type, fit, predict, and display results
for model_type in model_types:
    print(f"Results for {model_type}:")
    model = CustomModel(model_type=model_type)
    model.fit(X_train, Y_train)
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    
    # Storing results
    results[model_type] = {'Training Score': train_score, 'Test Score': test_score}
    
    # Output scores
    print(f"Training Score: {train_score:.4f}")
    print(f"Test Score: {test_score:.4f}")
    
    # Feature Importance - Only if applicable
    if model_type not in ['svr', 'knn']:  # Assuming SVR and KNN don't support feature importance in your implementation
        print("Feature Importances:")
        model.get_feature_importance(X_train.columns)
    print("\n" + "-"*60 + "\n")

# Optionally, convert results to a DataFrame for better visualization or further analysis
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)
