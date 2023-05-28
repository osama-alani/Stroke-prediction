# Stroke-prediction

![image](https://github.com/osama-alani/Stroke-prediction/assets/133378136/902e8de9-8132-496a-a603-1e8f6af1b112)

## A stroke is a medical consition where there is a blockage in the blood supplement to the brain, or when a brain vessel burst. Many factors regarding one's lifestyle and health contribute to increasing or decreasing the chances of getting a stroke. Some of those factors have greater wieght and effect than others, and it is our objective in this project to train a model to predict the chances of having a stroke as these parameters differ from case to case.


### Note: a detailed walkthrough of the code could be found in the code's notebook "main.ipynb" provided in the reposetry. 

 1. The libraries used in the code of the model are: numpy, pandas, altair, seaborn, scikit-learn, xgboost, gradio, plotly, and matplotlib. All can be installed by uncommenting the first cell in the notebook main.ipynb and running it
```
! pip install numpy pandas altair seaborn scikit-learn xgboost gradio plotly matplotlib
```

2. Two data sets were provided to the model: "train1.csv" and "train2.csv" (which later in the code were merged in one data set "train"). And one data set to test the model "test.csv". All the data could be found inside the "data" file in the repostry.
3. The model's predictions of the test file was saved in the file "submission.csv" which also could be found in the data folder.
4. The model's prediction's were uploaded to Kaggle's "Binary Classification with a Tabular Stroke Prediction Dataset" competition, and a score of 0.89358 was acquired.
