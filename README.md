<!-- Customer Churn Prediction System
by Ritwik Bhandari


---INTRODUCTION---

Project Background
Customer churn refers to the phenomenon of customers discontinuing their relationship with a business. Predicting churn enables proactive customer retention strategies. This project aims to develop a machine learning solution integrated with a user-friendly interface to predict whether a customer is likely to churn.

Problem Statement and Scope
The objective is to predict churn based on customer demographics and account activity using machine learning models. The scope includes:
•	Data ingestion from MongoDB
•	Feature preprocessing, model training, and evaluation
•	PCA dimensionality reduction
•	Web-based frontend for real-time predictions
 

---LITERATURE REVIEW---

Existing Models and Solutions
Decision Trees, Random Forests, and Logistic Regression are commonly used models in churn prediction. These models offer good interpretability as well as accuracy. PCA is applied for dimensionality reduction. To remove the “curse of dimensionality” from an ML model, we use PCA which improves model efficiency and generalization.

Common Tools and Technologies
•	Machine Learning: scikit-learn
•	Database: MongoDB
•	Web Framework: Flask
•	Frontend: HTML, CSS
•	Data Source: CSV (Bank Churn dataset)
 

---DATASET OVERVIEW---

Data Source and Structure
The dataset used is the Churn_Modelling.csv file containing 14 attributes pertaining to customer specifications like demographics, account information, and churn status.

Key Features and Target Variable
•	Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
•	Target: Exited (1 = Churn, 0 = No Churn)

Preprocessing Steps
•	Dropped irrelevant columns (RowNumber, CustomerId, Surname)
•	Handled missing values with mean/mode imputation
•	One-hot encoded categorical variables
•	Scaled features using StandardScaler
•	Applied PCA to retain 90% variance
 

---METHODOLOGY---

Project Approach and Workflow
1.	CSV data initially upload to MongoDB
2.	Data extraction and preprocessing
3.	Model training (Random Forest, PCA + Random Forest, Logistic Regression)
4.	Saving model pipeline
5.	Flask-based app for real-time prediction using user inputs

Tools and Technologies Used
•	Python (pandas, scikit-learn, pymongo)
•	MongoDB for data persistence
•	Flask for backend deployment
•	HTML/CSS for the frontend
•	Pickle for model serialization
 

---EXPLORATORY DATA ANALYSIS (EDA)---

Summary Statistics and Visualizations
•	Mean and distribution of key numeric features
•	Churn ratio by geography and gender
•	Correlation matrix of numerical features

Key Insights from the Data
•	Age and Balance were strong indicators of churn.
•	Customers from Germany had a higher churn rate.
•	Inactive members and those with fewer products had higher churn.
 

---MODEL DEVELOPMENT---

Model Selection and Training Strategy
Trained three models:
•	Random Forest: Random Forest was selected as a primary model due to its ensemble nature—combining multiple decision trees to reduce overfitting and improve generalization. It performs well on structured datasets like ours, especially when categorical and numerical features are mixed. Its built-in feature importance ranking also supports interpretability.

•	PCA + Random Forest: To address potential multicollinearity and reduce model complexity, we introduced Principal Component Analysis (PCA) after feature scaling. This compressed the feature space to the most informative components, retaining 90% of the variance. Applying Random Forest on this transformed data enabled comparison between performance in original vs. reduced dimensions, offering insight into redundancy and feature compression.

•	Logistic Regression: Logistic Regression was used as a baseline model. It's a well-established linear classifier that is easy to implement, fast to train, and offers coefficients that are interpretable. Despite its linear nature, it performs surprisingly well when features are informative and properly pre-processed. It also helped validate the presence of non-linear relationships when compared with the ensemble models.
Evaluating multiple models helped in:
•	Verifying consistency of predictions across models
•	Understanding the trade-off between interpretability and complexity
•	Assessing model robustness on original and transformed data

Evaluation Metrics and Tuning
•	Accuracy, Precision, Recall, F1-score
•	Optimized Random Forest with:
o	max_depth=6, min_samples_leaf=8
o	PCA preserved 90% of variance
 

---RESULTS & EVALUATION---

Model Performance Comparison
Model	Accuracy (%)
Random Forest	~86%
PCA + Random Forest	~85%
Logistic Regression	~80%

Interpretation
•	Random Forest outperformed other models in accuracy and generalization.
•	PCA helped reduce dimensions with minimal accuracy trade-off.
 

---FRONT-END DEVELOPMENT---

Technology Stack and UI Design
•	Frontend: HTML, CSS (Dark-themed responsive form)
•	Backend: Flask routes and form handling
•	Intuitive form fields including dropdowns and numerical inputs

 
Model Integration and Interface Screenshots
•	Inputs sent via POST to /predict route
•	Backend processes the inputs, applies preprocessing, and predicts using the trained model
•	Result shown via alert box (e.g., "Prediction: Churn")

  
---CHALLENGES---

Dataset Crowdedness
•	An initial dataset was taken with high dimensionality (19 attributes)
•	No firm conclusion could be made from the data

Data and Model Limitations
•	Imbalanced churn distribution (majority = No Churn)
•	Feature distribution skewness needed scaling
•	Model risk of overfitting without PCA

Integration Issues
•	Ensuring feature order matched model expectations
•	Feature misalignment between HTML input and pickled scaler led to initial runtime errors
 

---CONCLUSION---

Summary of Outcomes
•	Built an end-to-end churn prediction pipeline
•	Achieved high accuracy with PCA + Random Forest
•	Developed a user-friendly web interface for real-time predictions

Project Impact
•	Can assist businesses in targeting at-risk customers
•	Provides an interactive tool for churn analytics with minimal input
 

---REFERENCES---

Scikit-learn Documentation: 
https://scikit-learn.org/stable/
MongoDB Docs: 
https://www.mongodb.com/docs/
Flask Docs:
https://flask.palletsprojects.com/
Churn Dataset Source: https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction
 -->
