import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
# CSV File Load & Basic validations
csv_path = Path(__file__).parent/"decisiontree.csv"
df= pd.read_csv(csv_path)
df = df.dropna()
le_dict ={}

for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        le_dict[column]= le


X= df.drop(['Day', 'Play'], axis=1)
y= df['Play']

model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X,y)

plt.figure(figsize=(20,12))
plot_tree(model, feature_names = X.columns, class_names = le_dict['Play'].classes_, filled=True,
          fontsize=12)
plt.show()


#Prediction For the User Input 
def predict_play(weather, temperature, humidity, wind):
    w =le_dict['Weather'].transform([weather])[0]
    t = le_dict['Temperature'].transform([temperature])[0]
    h = le_dict['Humidity'].transform([humidity])[0]
    wi = le_dict['Wind'].transform([wind])[0]

    user_input = pd.DataFrame([[w,t,h,wi]], columns=X.columns)
    prediction = model.predict(user_input)[0]
    return le_dict['Play'].inverse_transform([prediction])[0]

print(predict_play("Sunny", "Cool","Normal", "Weak"))
print(predict_play("Rain", "Mild","High", "Strong"))