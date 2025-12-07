## Titanic Survival Prediction: Machine Learning Project

This repository contains a machine learning project that predicts whether a passenger on the Titanic survived the disaster based on features like age, sex, ticket class, and fare.

---

### Goal

The primary goal is to build a classification model that can accurately predict the `Survived` column (our target variable) using the available passenger data.

---

### Dataset

The analysis uses the **Titanic-Dataset.csv** file, which is the classic dataset often used for introductory machine learning challenges.

| Column        | Description                        |
| :------------ | :--------------------------------- |
| `PassengerId` | Unique ID for each passenger.      |
| `Survived`    | Target (0=No, 1=Yes).              |
| `Pclass`      | Ticket class (1st, 2nd, 3rd).      |
| `Name`        | Passenger's name.                  |
| `Sex`         | Passenger's sex (male/female).     |
| `Age`         | Age in years.                      |
| `SibSp`       | Number of siblings/spouses aboard. |
| `Parch`       | Number of parents/children aboard. |
| `Ticket`      | Ticket number.                     |
| `Fare`        | Passenger fare.                    |
| `Cabin`       | Cabin number.                      |
| `Embarked`    | Port of Embarkation (C, Q, S).     |

---

### Key Technologies and Libraries

* Python 3.x
* pandas: For data manipulation and analysis
* numpy: For numerical operations
* matplotlib & seaborn: For data visualization and exploratory data analysis (EDA)
* scikit-learn: For preprocessing, model building, and evaluation

  * `Pipeline` and `ColumnTransformer` are heavily used for structured preprocessing
* joblib: For saving the trained model pipeline

---

### Preprocessing Strategy

The data requires cleaning and transformation before modeling. A `ColumnTransformer` is used to apply specific steps to different types of features.

#### 1. Numerical Features (`Age`, `Fare`, `SibSp`, `Parch`)

* Missing Values: Use `SimpleImputer` to fill missing `Age` and `Fare` values with the median
* Scaling: Use `StandardScaler` to standardize the features

#### 2. Categorical Features (`Pclass`, `Sex`, `Embarked`)

* Missing Values: Use `SimpleImputer` to fill missing `Embarked` values with the most frequent value (mode)
* Encoding: Use `OneHotEncoder` to convert these features into numerical format

#### 3. Dropped Features

The following columns are dropped as they are not used directly in the model: `PassengerId`, `Name`, `Ticket`, and `Cabin`

---

### Model Training

Two classification algorithms were explored:

1. Logistic Regression (`LogisticRegression`)
2. Random Forest Classifier (`RandomForestClassifier`)

A `Pipeline` is used to chain the preprocessing steps (`ColumnTransformer`) and the final estimator (model), ensuring consistent data transformation during training and prediction.

The model is trained using cross-validation (`cross_val_score`) to reliably assess generalization performance.

---

### Results and Evaluation

The model's performance is evaluated using metrics such as Accuracy and a Classification Report (Precision, Recall, F1-Score).

* Accuracy: Overall fraction of correctly predicted outcomes
* Classification Report: Detailed performance breakdown for the "Survived" (1) and "Did Not Survive" (0) classes

The final trained model pipeline is saved to disk using `joblib` (e.g., `titanic_model_pipeline.joblib`) for future predictions without retraining.

---

### How to Run the Project

1. Clone the repository:

   ```bash
   git clone [repository-link]
   cd [repository-name]
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

3. Ensure data is present:
   Place the `Titanic-Dataset.csv` file in the appropriate folder (or adjust the file path).

4. Execute the Jupyter Notebook/Script:
   Run the code cells in the provided notebook to perform EDA, preprocessing, training, and evaluation.

---

### Next Steps (Potential Improvements)

* Feature Engineering: Extract the passenger title (Mr., Mrs., Miss, etc.) from the `Name` column and use it as a categorical feature
* Hyperparameter Tuning: Use `GridSearchCV` or `RandomizedSearchCV` to find optimal model parameters
* Advanced Imputation: Use predictive models (like KNN) to impute missing `Age` values instead of using the median

Do you want me to include that?
