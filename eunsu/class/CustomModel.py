from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# 모델들을 클래스화 했습니다. 사용하는데 편하지 않을까요? 자주쓰는 모델들은 대부분 불러왔습니다.
# 지원하는 모델은 def initaliz~ 코드를 확인해주시면됩니당
class CustomModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='linear_regression'):
        """
        Initialize the CustomModel with a specified model type.
        :param model_type: str, the type of model to use ('random_forest', 'linear_regression', 'svr', 'xgboost', 'gradient_boosting', 'decision_tree', 'knn', 'elastic_net')
        """
        self.model_type = model_type
        self.model = None
        
    # 전처리 부분도 함수를 선언해놓긴 했습니다. 그러나 각자 전처리 방법이 다를 것 같아
    # train_X 데이터를 그냥 사용하는 식으로 작성했습니다.
    def preprocess(self, X):
        """
        Handle all preprocessing steps like feature encoding, normalization, etc.
        Modify this method as needed when preprocessing changes.
        :param X: DataFrame, features to be preprocessed
        :return: DataFrame, preprocessed features
        """
        return X  
    
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        :param X: DataFrame, training features
        :param y: Series, training target
        """
        X_preprocessed = self.preprocess(X)
        # Model initialization based on the model_type
        self.initialize_model()
        self.model.fit(X_preprocessed, y)
    
    def predict(self, X):
        """
        Predict using the model.
        :param X: DataFrame, features for prediction
        :return: array, predictions
        """
        X_preprocessed = self.preprocess(X)
        return self.model.predict(X_preprocessed)
    
    def score(self, X, y):
        """
        Returns the R-squared score of the prediction.
        :param X: DataFrame, test features
        :param y: Series, true values
        :return: float, R-squared score
        """
        predictions = self.predict(X)
        return r2_score(y, predictions)

    def initialize_model(self):
        """
        Initialize the specific type of model based on model_type attribute.
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression()
        elif self.model_type == 'svr':
            self.model = SVR(kernel='rbf')
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(objective='reg:squarederror')
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeRegressor(random_state=42)
        elif self.model_type == 'knn':
            self.model = KNeighborsRegressor()
        elif self.model_type == 'elastic_net':
            self.model = ElasticNet(random_state=42)
        else:
            raise ValueError("Unsupported model type")

    # 중요변수를 출력하는 부분들을 추가했습니다. 다만 모든 모델이 지원하지는 않습니다.
    # 지원하지 않는 모델의 경우에는 경고문을 출력할 수 있게 작성했습니다.
    # 그리고 중요 변수를 그래프 형태로 볼 수 있도록 작성했습니다.
    def get_feature_importance(self, feature_names):
        """
        Retrieve and plot feature importance from the model if available.
        :param feature_names: list, names of the features used for model training
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            print("Feature importance not supported for this model type.")
            return
        
        # Create a bar chart of the feature importances
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()


# Example usage:
# feature_names = X_train.columns.tolist()
# model = CustomModel(model_type='xgboost')
# model.fit(X_train, Y_train)
# print("Training score:", model.score(X_train, Y_train))
# print("Test score:", model.score(X_test, Y_test))
# print("Feature Importances:",model.get_feature_importance(feature_names))