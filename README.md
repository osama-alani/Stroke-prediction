# Stroke-prediction

![image](https://github.com/osama-alani/Stroke-prediction/assets/133378136/902e8de9-8132-496a-a603-1e8f6af1b112)

### A stroke is a medical consition where there is a blockage in the blood supplement to the brain, or when a brain vessel burst. Many factors regarding one's lifestyle and health contribute to increasing or decreasing the chances of getting a stroke. Some of those factors have greater wieght and effect than others, and it is our objective in this project to train a model to predict the chances of having a stroke as these parameters differ from case to case.


## 1. Specification of dependencies

We used:-

pandas library to read the data

seaborn, altair, plotly and pyplot for Visualization

scikit-learn, xgboost for prediction and modeling

And we have added a code inside the model to install these libraries in case you did not install it before.

## 2. Training Code
```
#spliting data
X = train.loc[:,train.columns != 'stroke']
y = train['stroke']
train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X, y, test_size=0.33, random_state=42)

#modeling
import xgboost as xgb

# Instantiate the model with tuned hyperparameters
model = xgb.XGBClassifier(
    seed=42,
    learning_rate=0.15,  # Adjust the learning rate 
    n_estimators=100,  # Increase the number of estimators
    max_depth=5,  # Adjust the maximum depth of each tree
    subsample=0.8,  # Adjust the subsample ratio
    colsample_bytree=0.8,  # Adjust the column subsample ratio
    alpha = 0.4, #Adjust the complexity of a tree
    gamma = 0.4, #Adjust the complexity of a tree
    min_child_weight = 5 #sets the weight limit for a tree node to split

```

## 3. Prediction Code
```
# Fit the model with data
model.fit(
    train_set_x, train_set_y,
    eval_set=[(train_set_x, train_set_y), (test_set_x, test_set_y)],
    verbose=False,
    early_stopping_rounds=10 
)

best_n_rounds = model.best_iteration

y_pred = model.predict(test_set_x, ntree_limit=best_n_rounds)
```
