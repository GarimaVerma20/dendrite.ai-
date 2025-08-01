# Dendrite.ai
## Problem Statement
This assignment is to parse the JSON file provided(algoparams_from_ui) and kick off in sequence the following machine learning steps programmatically like feature handling, feature generation and model building using Grid search after parsing hyper parameters. 

Used scikit-learn-compatible machine learning pipeline that automates key stages of **feature handling**, **feature reduction**, and **model selection**.

## Approach 

- The goal was to:

   - Parse this JSON file

   - Execute all Machine Learning steps in sequence

   - Dynamically adapt to different configurations
 
   - Scikit-learn-compatible pipeline architecture
### Key Features
- #### 1. Feature Handling
  
   - Handles both numerical and textual data based on defined strategies. 
   
   - **Numerical Features:** Supports imputation using mean values or custom constants. 
   
   - **Text Features:** Supports hashing vectorization.
 

- #### 2. Feature Reduction
  
   - **No Reduction:** Skips this step. 
   
   - **Correlation with Target:** Selects top k features most correlated with the target. 
   
   - **Tree-Based:** Uses feature importance from a Random Forest to select top k features. 
   
   - **PCA:** Applies Principal Component Analysis to keep top n components.

   
- #### 3. Model Selection with Grid Search
  
   - Automatically selects the best model from a predefined list. 
   
   - Supports both regression and classification tasks, based on the prediction_type from json. 
   
   - Evaluates multiple models (like Linear Regression, Decision Tree, Random Forest, etc.) along with their hyperparameter grids. 
   
   - Automatically runs GridSearchCV with user-defined hyperparameters. 
   
   - Returns the best model, its parameters, and evaluation metrics.
 


- ## Inputs
  
   - **DataFrame X:** Feature set as a pandas DataFrame 
   
   - **Target y:** Target values  
   
   - **Steps :** 
   
       - **feature_handling:** Specifies treatment strategy per feature (e.g., imputation, hashing) 
         
       - **feature_reduction:** Defines parameters for feature selection 
         
       - **model_selection:** Lists models and hyperparameters for grid search 


- ## Technologies Used
   - Python 
   
   - scikit-learn  
   
   - pandas, numpy  
   
   - HashingVectorizer 
   
   - GridSearchCV 
   
   - JSON for configuration         


**"Given the ** Iris dataset and a config that uses tree-based feature reduction and Random Forest with tuning, the pipeline outputs the best-performing model with best score 0.96."**
