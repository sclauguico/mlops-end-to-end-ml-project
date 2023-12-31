## End-to-End Machine Learning Project

### Anaconda CMD

```python
cd C:\Users\sclau>cd mlops-end-to-end-ml-project
code .
```

### GitHub Repo

1. Create a repository: **[mlops-end-to-end-ml-project](https://github.com/sclauguico/mlops-end-to-end-ml-project)**

### VSCode Terminal

```python
create -p venv python==3.8 -y
conda activate venv/
```

### VSCode Explorer

1.

```python
git init
```

1. Create a [README.md](http://README.md) under the mlops project (not the venv)
2.

```python
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/sclauguico/mlops-end-to-end-ml-project.git
	git remote -v
	git config --global user.name "Sandy Lauguico"
	git config --global user.email sclauguico@gmail.com
git push -u origin main
```

### GitHub Repo

1. Add a new file .gitignore
2. Choose .gitignore template: Python
3. Commit changes: Create .gitignore

### VSCode Terminal

```python
cls
git pull
```

### VSCode Explorer

1. Create a new file under mlops: [setup.py](http://setup.py) and requirements.txt

   ### Building the ML application on [setup.py](http://setup.py) as a package

### VSCode

setup.py

```python
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e . "
def get_requirements(file_path:str)->List[str]:
    '''
      This function will return the list of requirements.
    '''

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [requirements=req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

# metadata of the entire project
setup (
		name='mlops-end-to-end-project',
		version='0.0.1',
		author='Sandy',
		author_email='sclauguico@gmail.com',
		packages=find_packages(),
		#install_requires=['pandas','numpy','seaborn'],
    install_requires=get_requirements('requirements.txt')
)
```

requirements.txt

```python
pandas
numpy
seaborn
-e .
```

### this will automatically trigger setup.py

### VSCode Explorer

1. Create a new folder under mlops: src
2. Under src, create a file: **init**.py

### VSCode Terminal

```python
pip install -r requirements.txt

git add .
git status
git commit -m "setup and requirements"
git push -u origin main
```

### VSCode Explorer

1. Create a folder under src: components
2. Under components create a file: **init**.py
3. Under components create a file: data_ingestion.py
4. Under components create a file: data_transformation.py
5. Under components create a file: model_trainer.py
6. Under src create a folder: pipeline
7. Under pipeline create a file: train_pipeline.py
8. Under pipeline create a file: predict_pipeline.py
9. Under pipeling, create a file: **init**.py
10. Under src create the following files:
    1. logger.py
    2. exception.py
    3. utils.py

### VSCode

### exception.py

```python
import sys
import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by Zero.")
        raise CustomException(e, sys)
```

### logger.py

```python
import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__=="__main__":
    logging.info("Logging has started")
```

### utils.py

```python

```

### VSCode Terminal

```python
pip install -r requirements.txt

git add .
git status
git commit -m "logging and exception"
git push -u origin main
```

### VSCode

1. Create a folder: notebook
2. Create a folder: data
3. Create a new Python environment
   1. Ctrl + shift + p
   2. Create Python Environment
   3. Select venv
4. Choose venv for the kernel
5. Under notebook create a file: eda-student-performance.ipynb

   ```python
   ## Student Performance Indicator

   #### Life cycle of Machine learning Project

   - Understanding the Problem Statement
   - Data Collection
   - Data Checks to perform
   - Exploratory data analysis
   - Data Pre-Processing
   - Model Training
   - Choose best model
   ### 1) Problem statement
   - This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.

   ### 2) Data Collection
   - Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
   - The data consists of 8 column and 1000 rows.
   ### 2.1 Import Data and Required Packages
   ####  Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.
   import numpy as np
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   %matplotlib inline
   import warnings
   warnings.filterwarnings('ignore')
   #### Import the CSV Data as Pandas DataFrame
   df = pd.read_csv('data/stud.csv')
   #### Show Top 5 Records
   df.head()
   #### Shape of the dataset
   df.shape

   ### 2.2 Dataset information
   - gender : sex of students  -> (Male/female)
   - race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
   - parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
   - lunch : having lunch before test (standard or free/reduced)
   - test preparation course : complete or not complete before test
   - math score
   - reading score
   - writing score
   ### 3. Data Checks to perform

   - Check Missing values
   - Check Duplicates
   - Check data type
   - Check the number of unique values of each column
   - Check statistics of data set
   - Check various categories present in the different categorical column
   ### 3.1 Check Missing values
   df.isna().sum()
   #### There are no missing values in the data set
   ### 3.2 Check Duplicates
   df.duplicated().sum()
   #### There are no duplicates  values in the data set
   ### 3.3 Check data types
   # Check Null and Dtypes
   df.info()
   ### 3.4 Checking the number of unique values of each column
   df.nunique()
   ### 3.5 Check statistics of data set
   df.describe()
   #### Insight
   - From above description of numerical data, all means are very close to each other - between 66 and 68.05;
   - All standard deviations are also close - between 14.6 and 15.19;
   - While there is a minimum score  0 for math, for writing minimum is much higher = 10 and for reading myet higher = 17
   ### 3.7 Exploring Data
   df.head()
   print("Categories in 'gender' variable:     ",end=" " )
   print(df['gender'].unique())

   print("Categories in 'race_ethnicity' variable:  ",end=" ")
   print(df['race_ethnicity'].unique())

   print("Categories in'parental level of education' variable:",end=" " )
   print(df['parental_level_of_education'].unique())

   print("Categories in 'lunch' variable:     ",end=" " )
   print(df['lunch'].unique())

   print("Categories in 'test preparation course' variable:     ",end=" " )
   print(df['test_preparation_course'].unique())
   # define numerical & categorical columns
   numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
   categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

   # print columns
   print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
   print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
   df.head(2)

   ### 3.8 Adding columns for "Total Score" and "Average"
   df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
   df['average'] = df['total score']/3
   df.head()
   reading_full = df[df['reading_score'] == 100]['average'].count()
   writing_full = df[df['writing_score'] == 100]['average'].count()
   math_full = df[df['math_score'] == 100]['average'].count()

   print(f'Number of students with full marks in Maths: {math_full}')
   print(f'Number of students with full marks in Writing: {writing_full}')
   print(f'Number of students with full marks in Reading: {reading_full}')
   reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
   writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
   math_less_20 = df[df['math_score'] <= 20]['average'].count()

   print(f'Number of students with less than 20 marks in Maths: {math_less_20}')
   print(f'Number of students with less than 20 marks in Writing: {writing_less_20}')
   print(f'Number of students with less than 20 marks in Reading: {reading_less_20}')
   #####  Insights
    - From above values we get students have performed the worst in Maths
    - Best performance is in reading section
   ### 4. Exploring Data ( Visualization )
   #### 4.1 Visualize average score distribution to make some conclusion.
   - Histogram
   - Kernel Distribution Function (KDE)
   #### 4.1.1 Histogram & KDE
   fig, axs = plt.subplots(1, 2, figsize=(15, 7))
   plt.subplot(121)
   sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
   plt.subplot(122)
   sns.histplot(data=df,x='average',kde=True,hue='gender')
   plt.show()
   fig, axs = plt.subplots(1, 2, figsize=(15, 7))
   plt.subplot(121)
   sns.histplot(data=df,x='total score',bins=30,kde=True,color='g')
   plt.subplot(122)
   sns.histplot(data=df,x='total score',kde=True,hue='gender')
   plt.show()
   #####  Insights
   - Female students tend to perform well then male students.
   plt.subplots(1,3,figsize=(25,6))
   plt.subplot(141)
   sns.histplot(data=df,x='average',kde=True,hue='lunch')
   plt.subplot(142)
   sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='lunch')
   plt.subplot(143)
   sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='lunch')
   plt.show()
   #####  Insights
   - Standard lunch helps perform well in exams.
   - Standard lunch helps perform well in exams be it a male or a female.
   plt.subplots(1,3,figsize=(25,6))
   plt.subplot(141)
   ax =sns.histplot(data=df,x='average',kde=True,hue='parental level of education')
   plt.subplot(142)
   ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='parental level of education')
   plt.subplot(143)
   ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='parental level of education')
   plt.show()
   #####  Insights
   - In general parent's education don't help student perform well in exam.
   - 2nd plot shows that parent's whose education is of associate's degree or master's degree their male child tend to perform well in exam
   - 3rd plot we can see there is no effect of parent's education on female students.
   plt.subplots(1,3,figsize=(25,6))
   plt.subplot(141)
   ax =sns.histplot(data=df,x='average',kde=True,hue='race/ethnicity')
   plt.subplot(142)
   ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='race/ethnicity')
   plt.subplot(143)
   ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='race/ethnicity')
   plt.show()
   #####  Insights
   - Students of group A and group B tends to perform poorly in exam.
   - Students of group A and group B tends to perform poorly in exam irrespective of whether they are male or female
   #### 4.2 Maximumum score of students in all three subjects

   plt.figure(figsize=(18,8))
   plt.subplot(1, 4, 1)
   plt.title('MATH SCORES')
   sns.violinplot(y='math score',data=df,color='red',linewidth=3)
   plt.subplot(1, 4, 2)
   plt.title('READING SCORES')
   sns.violinplot(y='reading score',data=df,color='green',linewidth=3)
   plt.subplot(1, 4, 3)
   plt.title('WRITING SCORES')
   sns.violinplot(y='writing score',data=df,color='blue',linewidth=3)
   plt.show()
   #### Insights
   - From the above three plots its clearly visible that most of the students score in between 60-80 in Maths whereas in reading and writing most of them score from 50-80
   #### 4.3 Multivariate analysis using pieplot
   plt.rcParams['figure.figsize'] = (30, 12)

   plt.subplot(1, 5, 1)
   size = df['gender'].value_counts()
   labels = 'Female', 'Male'
   color = ['red','green']

   plt.pie(size, colors = color, labels = labels,autopct = '.%2f%%')
   plt.title('Gender', fontsize = 20)
   plt.axis('off')

   plt.subplot(1, 5, 2)
   size = df['race/ethnicity'].value_counts()
   labels = 'Group C', 'Group D','Group B','Group E','Group A'
   color = ['red', 'green', 'blue', 'cyan','orange']

   plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
   plt.title('Race/Ethnicity', fontsize = 20)
   plt.axis('off')

   plt.subplot(1, 5, 3)
   size = df['lunch'].value_counts()
   labels = 'Standard', 'Free'
   color = ['red','green']

   plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
   plt.title('Lunch', fontsize = 20)
   plt.axis('off')

   plt.subplot(1, 5, 4)
   size = df['test preparation course'].value_counts()
   labels = 'None', 'Completed'
   color = ['red','green']

   plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
   plt.title('Test Course', fontsize = 20)
   plt.axis('off')

   plt.subplot(1, 5, 5)
   size = df['parental level of education'].value_counts()
   labels = 'Some College', "Associate's Degree",'High School','Some High School',"Bachelor's Degree","Master's Degree"
   color = ['red', 'green', 'blue', 'cyan','orange','grey']

   plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
   plt.title('Parental Education', fontsize = 20)
   plt.axis('off')

   plt.tight_layout()
   plt.grid()

   plt.show()
   #####  Insights
   - Number of Male and Female students is almost equal
   - Number students are greatest in Group C
   - Number of students who have standard lunch are greater
   - Number of students who have not enrolled in any test preparation course is greater
   - Number of students whose parental education is "Some College" is greater followed closely by "Associate's Degree"
   #### 4.4 Feature Wise Visualization
   #### 4.4.1 GENDER COLUMN
   - How is distribution of Gender ?
   - Is gender has any impact on student's performance ?
   #### UNIVARIATE ANALYSIS ( How is distribution of Gender ? )
   f,ax=plt.subplots(1,2,figsize=(20,10))
   sns.countplot(x=df['gender'],data=df,palette ='bright',ax=ax[0],saturation=0.95)
   for container in ax[0].containers:
       ax[0].bar_label(container,color='black',size=20)

   plt.pie(x=df['gender'].value_counts(),labels=['Male','Female'],explode=[0,0.1],autopct='%1.1f%%',shadow=True,colors=['#ff4d4d','#ff8000'])
   plt.show()
   #### Insights
   - Gender has balanced data with female students are 518 (48%) and male students are 482 (52%)
   #### BIVARIATE ANALYSIS ( Is gender has any impact on student's performance ? )
   gender_group = df.groupby('gender').mean()
   gender_group
   plt.figure(figsize=(10, 8))

   X = ['Total Average','Math Average']

   female_scores = [gender_group['average'][0], gender_group['math score'][0]]
   male_scores = [gender_group['average'][1], gender_group['math score'][1]]

   X_axis = np.arange(len(X))

   plt.bar(X_axis - 0.2, male_scores, 0.4, label = 'Male')
   plt.bar(X_axis + 0.2, female_scores, 0.4, label = 'Female')

   plt.xticks(X_axis, X)
   plt.ylabel("Marks")
   plt.title("Total average v/s Math average marks of both the genders", fontweight='bold')
   plt.legend()
   plt.show()
   #### Insights
   - On an average females have a better overall score than men.
   - whereas males have scored higher in Maths.
   #### 4.4.2 RACE/EHNICITY COLUMN
   - How is Group wise distribution ?
   - Is Race/Ehnicity has any impact on student's performance ?
   #### UNIVARIATE ANALYSIS ( How is Group wise distribution ?)
   f,ax=plt.subplots(1,2,figsize=(20,10))
   sns.countplot(x=df['race/ethnicity'],data=df,palette = 'bright',ax=ax[0],saturation=0.95)
   for container in ax[0].containers:
       ax[0].bar_label(container,color='black',size=20)

   plt.pie(x = df['race/ethnicity'].value_counts(),labels=df['race/ethnicity'].value_counts().index,explode=[0.1,0,0,0,0],autopct='%1.1f%%',shadow=True)
   plt.show()
   #### Insights
   - Most of the student belonging from group C /group D.
   - Lowest number of students belong to groupA.
   #### BIVARIATE ANALYSIS ( Is Race/Ehnicity has any impact on student's performance ? )
   Group_data2=df.groupby('race/ethnicity')
   f,ax=plt.subplots(1,3,figsize=(20,8))
   sns.barplot(x=Group_data2['math score'].mean().index,y=Group_data2['math score'].mean().values,palette = 'mako',ax=ax[0])
   ax[0].set_title('Math score',color='#005ce6',size=20)

   for container in ax[0].containers:
       ax[0].bar_label(container,color='black',size=15)

   sns.barplot(x=Group_data2['reading score'].mean().index,y=Group_data2['reading score'].mean().values,palette = 'flare',ax=ax[1])
   ax[1].set_title('Reading score',color='#005ce6',size=20)

   for container in ax[1].containers:
       ax[1].bar_label(container,color='black',size=15)

   sns.barplot(x=Group_data2['writing score'].mean().index,y=Group_data2['writing score'].mean().values,palette = 'coolwarm',ax=ax[2])
   ax[2].set_title('Writing score',color='#005ce6',size=20)

   for container in ax[2].containers:
       ax[2].bar_label(container,color='black',size=15)
   #### Insights
   - Group E students have scored the highest marks.
   - Group A students have scored the lowest marks.
   - Students from a lower Socioeconomic status have a lower avg in all course subjects
   #### 4.4.3 PARENTAL LEVEL OF EDUCATION COLUMN
   - What is educational background of student's parent ?
   - Is parental education has any impact on student's performance ?
   #### UNIVARIATE ANALYSIS ( What is educational background of student's parent ? )
   plt.rcParams['figure.figsize'] = (15, 9)
   plt.style.use('fivethirtyeight')
   sns.countplot(df['parental level of education'], palette = 'Blues')
   plt.title('Comparison of Parental Education', fontweight = 30, fontsize = 20)
   plt.xlabel('Degree')
   plt.ylabel('count')
   plt.show()
   #### Insights
   - Largest number of parents are from some college.
   #### BIVARIATE ANALYSIS ( Is parental education has any impact on student's performance ? )
   df.groupby('parental level of education').agg('mean').plot(kind='barh',figsize=(10,10))
   plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
   plt.show()
   #### Insights
   - The score of student whose parents possess master and bachelor level education are higher than others.
   #### 4.4.4 LUNCH COLUMN
   - Which type of lunch is most common amoung students ?
   - What is the effect of lunch type on test results?

   #### UNIVARIATE ANALYSIS ( Which type of lunch is most common amoung students ? )
   plt.rcParams['figure.figsize'] = (15, 9)
   plt.style.use('seaborn-talk')
   sns.countplot(df['lunch'], palette = 'PuBu')
   plt.title('Comparison of different types of lunch', fontweight = 30, fontsize = 20)
   plt.xlabel('types of lunch')
   plt.ylabel('count')
   plt.show()
   #### Insights
   - Students being served Standard lunch was more than free lunch
   #### BIVARIATE ANALYSIS (  Is lunch type intake has any impact on student's performance ? )
   f,ax=plt.subplots(1,2,figsize=(20,8))
   sns.countplot(x=df['parental level of education'],data=df,palette = 'bright',hue='test preparation course',saturation=0.95,ax=ax[0])
   ax[0].set_title('Students vs test preparation course ',color='black',size=25)
   for container in ax[0].containers:
       ax[0].bar_label(container,color='black',size=20)

   sns.countplot(x=df['parental level of education'],data=df,palette = 'bright',hue='lunch',saturation=0.95,ax=ax[1])
   for container in ax[1].containers:
       ax[1].bar_label(container,color='black',size=20)
   #### Insights
   - Students who get Standard Lunch tend to perform better than students who got free/reduced lunch
   #### 4.4.5 TEST PREPARATION COURSE COLUMN
   - Which type of lunch is most common amoung students ?
   - Is Test prepration course has any impact on student's performance ?
   #### BIVARIATE ANALYSIS ( Is Test prepration course has any impact on student's performance ? )
   plt.figure(figsize=(12,6))
   plt.subplot(2,2,1)
   sns.barplot (x=df['lunch'], y=df['math score'], hue=df['test preparation course'])
   plt.subplot(2,2,2)
   sns.barplot (x=df['lunch'], y=df['reading score'], hue=df['test preparation course'])
   plt.subplot(2,2,3)
   sns.barplot (x=df['lunch'], y=df['writing score'], hue=df['test preparation course'])
   #### Insights
   - Students who have completed the Test Prepration Course have scores higher in all three categories than those who haven't taken the course
   #### 4.4.6 CHECKING OUTLIERS
   plt.subplots(1,4,figsize=(16,5))
   plt.subplot(141)
   sns.boxplot(df['math score'],color='skyblue')
   plt.subplot(142)
   sns.boxplot(df['reading score'],color='hotpink')
   plt.subplot(143)
   sns.boxplot(df['writing score'],color='yellow')
   plt.subplot(144)
   sns.boxplot(df['average'],color='lightgreen')
   plt.show()
   #### 4.4.7 MUTIVARIATE ANALYSIS USING PAIRPLOT
   sns.pairplot(df,hue = 'gender')
   plt.show()
   #### Insights
   - From the above plot it is clear that all the scores increase linearly with each other.
   ### 5. Conclusions
   - Student's Performance is related with lunch, race, parental level education
   - Females lead in pass percentage and also are top-scorers
   - Student's Performance is not much related with test preparation course
   - Finishing preparation course is benefitial.
   ```

6. Under notebook create a file: model-training.ipynb

   ```python
   ## Model Training
   #### 1.1 Import Data and Required Packages
   ##### Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.
   # Basic Import
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   # Modelling
   from sklearn.metrics import mean_squared_error, r2_score
   from sklearn.neighbors import KNeighborsRegressor
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
   from sklearn.svm import SVR
   from sklearn.linear_model import LinearRegression, Ridge,Lasso
   from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
   from sklearn.model_selection import RandomizedSearchCV
   from catboost import CatBoostRegressor
   from xgboost import XGBRegressor
   import warnings
   #### Import the CSV Data as Pandas DataFrame
   df = pd.read_csv('data/stud.csv')
   #### Show Top 5 Records
   df.head()
   #### Preparing X and Y variables
   X = df.drop(columns=['math_score'],axis=1)
   X.head()
   print("Categories in 'gender' variable:     ",end=" " )
   print(df['gender'].unique())

   print("Categories in 'race_ethnicity' variable:  ",end=" ")
   print(df['race_ethnicity'].unique())

   print("Categories in'parental level of education' variable:",end=" " )
   print(df['parental_level_of_education'].unique())

   print("Categories in 'lunch' variable:     ",end=" " )
   print(df['lunch'].unique())

   print("Categories in 'test preparation course' variable:     ",end=" " )
   print(df['test_preparation_course'].unique())
   y = df['math_score']
   y
   # Create Column Transformer with 3 types of transformers
   num_features = X.select_dtypes(exclude="object").columns
   cat_features = X.select_dtypes(include="object").columns

   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   from sklearn.compose import ColumnTransformer

   numeric_transformer = StandardScaler()
   oh_transformer = OneHotEncoder()

   preprocessor = ColumnTransformer(
       [
           ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),
       ]
   )
   X = preprocessor.fit_transform(X)
   X.shape
   # separate dataset into train and test
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
   X_train.shape, X_test.shape
   #### Create an Evaluate Function to give all metrics after model Training
   def evaluate_model(true, predicted):
       mae = mean_absolute_error(true, predicted)
       mse = mean_squared_error(true, predicted)
       rmse = np.sqrt(mean_squared_error(true, predicted))
       r2_square = r2_score(true, predicted)
       return mae, rmse, r2_square
   models = {
       "Linear Regression": LinearRegression(),
       "Lasso": Lasso(),
       "Ridge": Ridge(),
       "K-Neighbors Regressor": KNeighborsRegressor(),
       "Decision Tree": DecisionTreeRegressor(),
       "Random Forest Regressor": RandomForestRegressor(),
       "XGBRegressor": XGBRegressor(),
       "CatBoosting Regressor": CatBoostRegressor(verbose=False),
       "AdaBoost Regressor": AdaBoostRegressor()
   }
   model_list = []
   r2_list =[]

   for i in range(len(list(models))):
       model = list(models.values())[i]
       model.fit(X_train, y_train) # Train model

       # Make predictions
       y_train_pred = model.predict(X_train)
       y_test_pred = model.predict(X_test)

       # Evaluate Train and Test dataset
       model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

       model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)


       print(list(models.keys())[i])
       model_list.append(list(models.keys())[i])

       print('Model performance for Training set')
       print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
       print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
       print("- R2 Score: {:.4f}".format(model_train_r2))

       print('----------------------------------')

       print('Model performance for Test set')
       print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
       print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
       print("- R2 Score: {:.4f}".format(model_test_r2))
       r2_list.append(model_test_r2)

       print('='*35)
       print('\n')
   ### Results
   pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"],ascending=False)
   ## Linear Regression
   lin_model = LinearRegression(fit_intercept=True)
   lin_model = lin_model.fit(X_train, y_train)
   y_pred = lin_model.predict(X_test)
   score = r2_score(y_test, y_pred)*100
   print(" Accuracy of the model is %.2f" %score)
   ## Plot y_pred and y_test
   plt.scatter(y_test,y_pred);
   plt.xlabel('Actual');
   plt.ylabel('Predicted');
   sns.regplot(x=y_test,y=y_pred,ci=None,color ='red');
   #### Difference between Actual and Predicted Values
   pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
   pred_df
   ```

   ### VSCode Terminal

   ```
   	pip install -r requirements.txt
   ```

   ```
   git status
   git add .
   git commit -m "Problem statement, EDA, modeling"
   git push -u origin main
   ```

### VSCode

### data_ingestion.py

```python
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")

        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data in completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
```

### VSCode Terminal

```python
python src/components/data_ingestion.py

git status
git add .
git commit -m "Data ingestion"
git push -u origin main
```

### VSCode

### data_transformation.py

```python
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation based on different types of data.
        '''

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns standard scaling completed.")
            logging.info("Categorical columns encoding completed.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
```

### VSCode

### utils.py

```python
import osimport os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
```

### VSCode

### data_ingestion.py

```python
import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")

        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
```

### VSCode Terminal

```python
python src/components/data_ingestion.py

git status
git add .
git commit -m "Data transformation"
git push -u origin main
```

### VSCode

#model_trainer.py

```python
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            logging.info("Best found model on both training and testing dataset.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
```

### VSCode

#utils.py

```python
import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
```

### VSCode

#data_ingestion.py

```python
import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")

        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
```

### VSCode

.gitignore

1. Correct the .artifacts to .artifacts/ under # Environments

### VSCode Terminal

```
python src/components/data_ingestion.py

git status
git add .
git commit -m "Model trainer"
git push -u origin main
```

### VSCode

#model_trainer.py

```python
import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params
            )

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            logging.info("Best found model on both training and testing dataset.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
```

### VSCode

#utils.py

```python
import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
```

### VSCode Terminal

```
python src/components/data_ingestion.py

git status
git add .
git commit -m "Hyperparameter tuning"
git push -u origin main
```

### VSCode

### app.py

```python
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder='templates')
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(port=5500, debug=True)
```

### VSCode

### predict_pipeline.py

```python
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys.exc_info())

class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys.exc_info())
```

### VSCode

### utils.py

```python
import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
```

### VSCode Explorer

1. Ensure that the [app.py](http://app.py) and templates folder are in the same working directory

### VSCode

### index.html

```python
<h1>Welcome to the home page</h1>
```

### VSCode

### home.html

```python
<html>
<body>
    <div class="login">
       <h1>Student Exam Performance Indicator</h1>

       <form action="{{ url_for('predict_datapoint')}}" method="post">
        <h1>
            <legend>Student Exam Performance Prediction</legend>
        </h1>
        <div class="mb-3">
            <label class="form-label">Gender</label>
            <select class="form-control" name="gender" placeholder="Enter you Gender" required>
                <option class="placeholder" selected disabled value="">Select your Gender</option>
                <option value="male">
                    Male
                </option>
                <option value="female">
                    Female
                </option>
            </select>
        </div>
        <div class="mb-3">
            <label class="form-label">Race or Ethnicity</label>
            <select class="form-control" name="ethnicity" placeholder="Enter you ethnicity" required>
                <option class="placeholder" selected disabled value="">Select Ethnicity</option>
                <option value="group A">
                    Group A
                </option>
                <option value="group B">
                    Group B
                </option>
                <option value="group C">
                    Group C
                </option>
                <option value="group D">
                    Group D
                </option>
                <option value="group E">
                    Group E
                </option>
            </select>
        </div>
        <div class="mb-3">
            <label class="form-label">Parental Level of Education</label>
            <select class="form-control" name="parental_level_of_education"
                placeholder="Enter you Parent Education" required>
                <option class="placeholder" selected disabled value="">Select Parent Education</option>
                <option value="associate's degree">
                    associate's degree
                </option>
                <option value="bachelor's degree">
                    bachelor's degree
                </option>
                <option value="high school">
                    high school
                </option>
                <option value="master's degree">
                    master's degree
                </option>
                <option value="some college">
                    some college
                </option>
                <option value="some high school">
                    some high school
                </option>
            </select>
        </div>
        <div class="mb-3">
            <label class="form-label">Lunch Type</label>
            <select class="form-control" name="lunch" placeholder="Enter you Lunch" required>
                <option class="placeholder" selected disabled value="">Select Lunch Type</option>
                <option value="free/reduced">
                    free/reduced
                </option>
                <option value="standard">
                    standard
                </option>
            </select>
        </div>
        <div class="mb-3">
            <label class="form-label">Test preparation Course</label>
            <select class="form-control" name="test_preparation_course" placeholder="Enter you Course"
                required>
                <option class="placeholder" selected disabled value="">Select Test_course</option>
                <option value="none">
                    None
                </option>
                <option value="completed">
                    Completed
                </option>
            </select>
        </div>
        <div class="mb-3">
            <label class="form-label">Writing Score out of 100</label>
            <input class="form-control" type="number" name="reading_score"
                placeholder="Enter your Reading score" min='0' max='100' />
        </div>
        <div class="mb-3">
            <label class="form-label">Reading Score out of 100</label>
            <input class="form-control" type="number" name="writing_score"
                placeholder="Enter your Reading Score" min='0' max='100' />
        </div>
        <div class="mb-3">
            <input class="btn btn-primary" type="submit" value="Predict your Maths Score" required />
        </div>
    </form>
    <h2>
       THE prediction is {{results}}
    </h2>
   <body>
</html>
```

### VSCode Terminal

```
python app.py

git status
git add .
git commit -m "Prediction pipeline"
git push -u origin main
```

### VSCode

### application.py

```python
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder='templates')
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
```

### VSCode Explorer

1. Create a folder under mlops: .ebextensions
2. Create a file under .ebentensions: python.config

### VSCode

```python
option_settings:
  "aws:elasticbeanstalk:container:python":
  WSGIPath: application:application
```

### VSCode Terminal

```
git status
git add .
git commit -m "Deployment configuration"
git push -u
```
