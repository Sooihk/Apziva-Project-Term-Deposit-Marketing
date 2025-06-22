# Term Deposit Marketing

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project aims to predict whether a customer will subscribe to a term deposti based on data from a European bank's call center marketing campaigns. Using various classification models, the goal is the achieve at least 81% F1-Score through 5-fold cross validation while maintaining interpretability. The project also focuses on identifying key customer segments and features that influence purchasing decisions, helping to improve targeting strategies and campaign sucess rates.

## Context:
We are a small startup focusing mainly on providing machine learning solutions in the European banking market. We work on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.

We are interested in developing a robust machine learning system that leverages information coming from call center data.

Ultimately, we are looking for ways to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.

## Data Description: 
The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

## Attributes: 
* age : age of customer (numeric)  
* job : type of job (categorical)  
* marital : marital status (categorical)  
* education (categorical)  
* default: has credit in default? (binary)  
* balance: average yearly balance, in euros (numeric)  
* housing: has a housing loan? (binary)  
* loan: has personal loan? (binary)  
* contact: contact communication type (categorical)  
* day: last contact day of the month (numeric)  
* month: last contact month of year (categorical)  
* duration: last contact duration, in seconds (numeric)  
* campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  

Output (desired target):  
* y - has the client subscribed to a term deposit? (binary)  

## Download Data:
https://drive.google.com/file/d/1EW-XMnGfxn-qzGtGPa3v_C63Yqj2aGf7

## Goal(s):
Predict if the customer will subscribe (yes/no) to a term deposit (variable y)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         term_deposit_marketing and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── term_deposit_marketing   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes term_deposit_marketing a Python module
    │
    ├── eda_plotting.py               <- Code to produce exploratory plots of the dataset
    │
    ├── load_data.py              <- Scripts to download or generate data
    │
    ├── preprocessing.py             <- Code to transform features from original dataset into a finalized dataset
    │
    ├── model                
    │   ├── train_evaluate.py        <- Code to cv train various models
    │   ├── xgboost_tune.py          <- Code to hyperparameter tune using Optuna and threshold tuning          
    │   └── demographic.py           <- Code to use finalized model to create SHAP plots to understand customer demographic
    
```

## Getting Started: 
Working with Python 3.12.2 for this project. Clone the repository and install the dependencies:  

pip install -r requirements.txt  

## Exploratory Data Analysis and Feature Selection:  
To see the plots created in the EDA phase, run the following command:  

`python -m term_deposit_marketing.eda_plotting`

Run the following command for the preprocessing phase of transforming the imbalanced dataset in preparation for modeling:  

`python -m term_deposit_marketing.preprocessing`

## Training and Evaluating Classification Models:   
Run the following command to perform 5-fold cross validation on several classification models on the transformed dataset:  
`python -m term_deposit_marketing.model.train_evaluate`

Run the following command to perform Optuna Hyperparameter tuning and threshold tuning on the chosen model to improve model performance:  
`python -m term_deposit_marketing.model.xgboost_tune`

Run the following command to produce plots to determine segments of customers our client should prioritize and which features to focus on: 
`python -m term_deposit_marketing.model.demographic`
--------

