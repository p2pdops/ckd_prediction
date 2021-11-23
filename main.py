from flask import Flask, render_template, request
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')
df = pd.read_csv('train.csv')
df.drop('id', axis=1, inplace=True)

ind_col = [col for col in df.columns if col != 'class']
dep_col = 'class'

X = df[ind_col]
y = df[dep_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

# 15.8,
# 53.0,
# 6800.0,
# 6.1,
# 0,
# 0,
# 0,
# 0,
# 0,
# 0,
# 1

test_data = {
    'age': [58.0],
    'blood_pressure': [80.0],
    'specific_gravity': [1.025, ],
    'albumin': [0.0, ],
    'sugar': [0.0, ],
    'red_blood_cells': [1, ],
    'pus_cell': [1, ],
    'pus_cell_clumps': [0.0, ],
    'bacteria': [0.0],
    'blood_glucose_random': [131.0],
    'blood_urea': [18.0],
    'serum_creatinine': [1.1],
    'sodium': [141.0],
    'potassium': [3.5],
    'haemoglobin': [15.8, ],
    'packed_cell_volume': [53.0, ],
    'white_blood_cell_count': [6800.0, ],
    'red_blood_cell_count': [6.1, ],
    'hypertension': [0],
    'diabetes_mellitus': [0],
    'coronary_artery_disease': [0],
    'appetite': [0],
    'peda_edema': [0],
    'anemia': [0],
}

test_df = pd.DataFrame(test_data)

# print(df.columns)
# print(test_df.columns)

print(etc.predict(test_df))

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/check', methods=['POST', 'OPTIONS'])
def check_api():
    values = request.form.to_dict()
    print("********* debug line :" + str(values))
    values = {k: [float(v)] for k, v in values.items()}
    print("********* debug line :" + str(values))

    test = pd.DataFrame(values)

    return str(etc.predict(test))


app.run(host='localhost', port=5000, debug=True)
