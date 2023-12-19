import pandas as pd
import numpy as np
import seaborn as sns
from seaborn import heatmap
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt


cricket_df = pd.read_csv('data/yusufscricketdata.csv')
print(f"Dataset successflly imported of Shape: {cricket_df.shape}")

cricket_df.head()
cricket_df.describe()
cricket_df.info()
cricket_df.nunique()
cricket_df.dtypes


sns.displot(cricket_df['wickets'], kde=False,bins=10)
plt.title('Wicket hits')

plt.show()

cricket_df.columns

irrelevant = ['mid','date','venue','batsman','bowler','striker','non-striker']
print(f"before removing columns: {cricket_df.shape}")
cricket_df = cricket_df.drop(irrelevant,axis=1)
print(f"after remvoving irrelevant columns: {cricket_df.shape}")
cricket_df.head()

const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
              'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
              'Delhi Daredevils', 'Sunrisers Hyderabad']

print(f'Before Removing Inconsistent Teams : {cricket_df.shape}')
cricket_df = cricket_df[(cricket_df['bat_team'].isin(const_teams)) & (cricket_df['bowl_team'].isin(const_teams))]
print(f'After Removing Irrelevant Columns : {cricket_df.shape}')
print(f"Consistent Teams : \n{cricket_df['bat_team'].unique()}")
cricket_df.head()

print(f'before removing overs {cricket_df.shape}')
cricket_df = cricket_df[cricket_df('overs')>= 5.0]
print(f'after removing overs: {cricket_df.shape}')
cricket_df.head()

heatmap(data=cricket_df.corr(),annot=True)

le = LabelEncoder()
for col in ['bat team','bowl team']:
    cricket_df[col] = le.fit_transform(cricket_df[col])
cricket_df.head()

ColumnTransformer(['encoder',OneHotEncoder(),[0,1]],remainder='passthrough')
cricket_df = np.array(ColumnTransformer.fit_transform(cricket_df))


cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
              'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
              'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
              'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
              'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
              'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
       'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(cricket_df, columns=cols)
df.head()

features = df.drop(['total'],axis=1)
labels = df['total']


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)
print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")


#ML Algorithms
models = dict()

tree = DecisionTreeRegressor()
tree.fit(train_features,train_labels)

train_score_tree = str(tree.score(train_features,train_labels)*100)
test_score_tree = str(tree.score(test_features,test_labels)*100)
print(f"Train Score: {train_score_tree[:5]}%nTest Score:{test_score_tree[:5]}%")
models['tree'] = test_score_tree


print("--Decision Tree regressor model evaluation--")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, tree.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, tree.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, tree.predict(test_features)))))


linreg = LinearRegression()
linreg.fit(train_features,train_labels)


train_score_linreg = str(linreg.score(train_features,train_labels)*100)
test_score_linreg = str(linreg.score(test_features,test_labels)*100)
print(f"Train Score:{train_score_linreg[:5]}%nTest Score:{test_score_linreg[:5]}%")
models['linreg'] = test_score_linreg


print("---- Linear Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, linreg.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, linreg.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, linreg.predict(test_features)))))


forest = RandomForestRegressor()
forest.fit(train_features,train_labels)

train_score_forest = str(forest.score(train_features,train_labels)*100)
test_score_forest = str(forest.score(test_features,test_labels)*100)
print(f"train score is: {train_score_forest[:5]}%nTest Score is:{test_score_forest[:5]}%")
models['forest'] = test_score_forest


print("---- Random Forest Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, forest.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, forest.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, forest.predict(test_features)))))


svm = SVR()
svm.fit(train_features,train_labels)


train_score_svm = str(svm.score(train_features, train_labels)*100)
test_score_svm = str(svm.score(test_features, test_labels)*100)
print(f'Train Score : {train_score_svm[:5]}%\nTest Score : {test_score_svm[:5]}%')
models["svm"] = test_score_svm 


print("---- Support Vector Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, svm.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, svm.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, svm.predict(test_features)))))


#predicting best ML Model above
model_names = list(models.keys())
accuracy = list(map(float,models.values()))
plt.bar(model_names,accuracy)


#predictions
def score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model=forest):
  prediction_array = []
  # Batting Team
  if batting_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
  elif batting_team == 'Delhi Daredevils':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
  elif batting_team == 'Kings XI Punjab':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
  elif batting_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
  elif batting_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
  elif batting_team == 'Rajasthan Royals':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
  elif batting_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
  elif batting_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
  # Bowling Team
  if bowling_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
  elif bowling_team == 'Delhi Daredevils':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
  elif bowling_team == 'Kings XI Punjab':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
  elif bowling_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
  elif bowling_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
  elif bowling_team == 'Rajasthan Royals':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
  elif bowling_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
  elif bowling_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
  prediction_array = prediction_array + [runs, wickets, overs, runs_last_5, wickets_last_5]
  prediction_array = np.array([prediction_array])
  pred = model.predict(prediction_array)
  return int(round(pred[0]))




#test 1
batting_team='Delhi Daredevils'
bowling_team='Chennai Super Kings'
score = score_predict(batting_team, bowling_team, overs=10.2, runs=68, wickets=3, runs_last_5=29, wickets_last_5=1)
print(f'Predicted Score : {score} || Actual Score : 147')



#test 2
batting_team="Kolkata Knight Riders"
bowling_team="Chennai Super Kings"
score = score_predict(batting_team, bowling_team, overs=18.0, runs=150, wickets=4, runs_last_5=57, wickets_last_5=1)
print(f'Predicted Score : {score} || Actual Score : 172')


#test 3
batting_team='Delhi Daredevils'
bowling_team='Mumbai Indians'
score = score_predict(batting_team, bowling_team, overs=18.0, runs=96, wickets=8, runs_last_5=18, wickets_last_5=4)
print(f'Predicted Score : {score} || Actual Score : 110')