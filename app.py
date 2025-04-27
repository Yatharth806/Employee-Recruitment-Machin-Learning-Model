from flask import Flask, render_template, request
import  pickle
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)
rfc = pickle.load(open('models/model.pkl' , 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
df = pd.read_csv('notebook/HR_comma_sep.csv')
#=====================Dashboard function =================================
def reading_cleaning(df):
    df.drop_duplicates(inplace=True)
    cols = df.columns.tolist()
    df.columns = [x.lower() for x in  cols]
    return df

df = reading_cleaning(df)


def employee_important_info(df):
    average_satisfaction = np.round(df['satisfaction_level'].mean(),3)
    department_satisfaction = np.round(df.groupby('department')['satisfaction_level'].mean(),2)
    salary_satisfaction = np.round(df.groupby('salary')['satisfaction_level'].mean(),3)
    left_employees = len(df[df['left'] == 1])
    stayed_employees = len(df[df['left'] == 0])

    return average_satisfaction, department_satisfaction, salary_satisfaction, left_employees, stayed_employees


#Pie plot for employee different feature
def plots(df,col):
    values  =  df[col].unique()
    plt.figure(figsize=(15,8))
    explode = [0.1 if len(values) > 1 else 0]*len(values)
    plt.pie(df[col].value_counts(),explode = explode,startangle = 40 ,autopct ='%1.1f%%',shadow= True)
    labels = [f'{value} ({col})'  for value in values]
    plt.legend(labels = labels , loc ='upper right')
    plt.title(f"Distribution of {col}")
    plt.savefig('static/' + col +'.png')
    plt.close()


def distribution(df,col):
    values  =  df[col].unique()
    plt.figure(figsize=(15,8))
    sns.countplot(x=df[col],hue='left' ,palette ='Set1',data=df)
    labels = [f'{value} ({col})'  for value in values]
    plt.legend(labels = labels , loc ='upper right')
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=90)
    plt.savefig('static/' + col + '_distribution.png')
    plt.close()


def comparison(df ,x,y):
    plt.figure(figsize=(15,8))
    sns.barplot(x=x,y=y,hue='left',data = df)
    plt.title(f'{x} vs {y}')
    plt.savefig('static/' +  'comparison.png')
    plt.close()

def corr_with_left(df):
    df_encoded = pd.get_dummies(df)
    correlations = df_encoded.corr()['left'].sort_values()[:-1]
    colors = ['skyblue' if corr>=0 else 'salmon' for corr in correlations]
    plt.figure(figsize=(10,8))
    correlations.plot(kind='barh', color=colors)
    # Add title and labels
    plt.title('Correlation with Left')
    plt.xlabel('Correlation')
    plt.ylabel('Features')
    plt.savefig('static/' + 'correlation.png')
    plt.close()

def histogram(df, col):
    fig, axes = plt.subplots(1, 2, figsize=(35, 20))  # Create a grid of 1 row and 2 columns

    # Plot the first histogram
    sns.histplot(data=df, x=col, hue='left', bins=20, ax=axes[0])
    axes[0].set_title(f"Histogram of {col}")

    # Plot the second histogram
    sns.kdeplot(data=df, x='satisfaction_level', y='last_evaluation', hue='left', shade=True, ax=axes[1])
    axes[1].set_title("Kernel Density Estimation")

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig('static/' + 'satisfaction_level_histogram.png')
    plt.close()

#=====================prediction function====================================================
def prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
    data = {
    'sl_no': [sl_no],
    'gender': [gender],
    'ssc_p': [ssc_p],
    'hsc_p': [hsc_p],
    'degree_p': [degree_p],
    'workex': [workex],
    'etest_p': [etest_p],
    'specialisation': [specialisation],
    'mba_p': [mba_p]
    }
    data = pd.DataFrame(data)
    data['gender'] = data['gender'].map({'Male':1,"Female":0})
    data['workex'] = data['workex'].map({"Yes":1,"No":0})
    data['specialisation'] = data['specialisation'].map({"Mkt&HR":1,"Mkt&Fin":0})
    scaled_df = scaler.transform(data)
    result = rfc.predict(scaled_df).reshape(1, -1)
    return result[0]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/job')
def job():
    return render_template('job.html')

@app.route('/ana')
def ana():
    average_satisfaction, department_satisfaction,salary_satisfaction, left_employees, stayed_employees = employee_important_info(df)
    department_satisfaction = department_satisfaction.to_dict()
    salary_satisfaction     = salary_satisfaction.to_dict()
    plots(df ,'left')
    plots(df , 'salary')
    plots(df , 'number_project')
    plots(df, 'department')
    distribution(df,'salary')
    distribution(df,'department')
    comparison(df, 'department', 'satisfaction_level')
    corr_with_left(df)
    histogram(df, 'satisfaction_level')
    return render_template('ana.html' ,df = df.head() ,average_satisfaction=average_satisfaction,
                           department_satisfaction =department_satisfaction ,salary_satisfaction=salary_satisfaction ,
                           left_employees=left_employees ,stayed_employees =stayed_employees  )

@app.route('/placement' , methods=['POST' ,'GET'])
def pred():
    if request.method=='POST':
        sl_no = request.form['sl_no']
        gender = request.form['gender']
        ssc_p =  request.form['ssc_p']
        hsc_p =  request.form['hsc_p']
        degree_p =  request.form['degree_p']
        workex =   request.form['workex']
        etest_p =   request.form['etest_p']
        specialisation =  request.form['specialisation']
        mba_p    =   request.form['mba_p']
        result = prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p)
        if result == 1:
            pred = 'Placed'
            rec = 'We recomend you that this is the best candidate for your business'
            return render_template('job.html' ,result=pred , rec=rec)
        else:
            pred = 'Not Placed'
            rec = 'We recomend you thst this is not the bset candidate for your business'
            return render_template('job.html', result = pred , rec =rec)


if __name__  == "__main__":
    app.run(debug=True)


