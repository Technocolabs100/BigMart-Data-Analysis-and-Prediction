# Mini project
## Project Description
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.
Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
The data has missing values as some stores do not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.
- I began by loading all the packages and the data, and checking the data
- I cleaned the null values in the data by using bfill, and cleaned the data in the Item_Type.
- I removed the outliers in the Item_Visibility column.
- I scaled and normalized the data.
- I used Several plots to illustrate the data to get a better understanding about the data.
- I used several encoding techniques like Label Encoding (ordinal encoding) and One Hot Encoding.
- I used different Machine learning models to predict the data so we can predict the data in the Test.csv
- The models used were:[Linear Regression, Regularized Linear Regression, RandomForest,XGBoost]
At the end i used XGBosst as it gave me the least mean absolute error
