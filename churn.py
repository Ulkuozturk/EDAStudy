import os
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
 
os.chdir("/Users/macbookpro/DataStudy/CustomerChurn/Codes")
churndata = pd.read_csv("BankChurners.csv")
churndata.head()

########################################################
# 1. EDA Phase  ########################################
########################################################

churndata.dtypes

CategoricalVar=["Attrition_Flag","Gender","Education_Level","Marital_Status","Income_Category","Card_Category"]
NumericalVar =["CLIENTNUM","Customer_Age","Dependent_count","Months_on_book","Total_Relationship_Count",
               "Months_Inactive_12_mon","Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal",
               "Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt","Total_Trans_Ct","Total_Ct_Chng_Q4_Q1",
               "Avg_Utilization_Ratio"]

churndata=churndata.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
                "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"],
                axis=1)

for i in CategoricalVar:
    unique=churndata[i].value_counts()
    print(unique)

        ## The number of categorical class is feasible
        ## The frequency of data in each class is fair
        ## However, maybe Income_Category is not equally distributed.

churndata[NumericalVar].describe().T
churndata.describe().T

        
        ## There is no missing value
churn_numeric=churndata[NumericalVar]
 
def my_normalize(df):
    norm_df=((df-df.min())/(df.max()-df.min()))
    return norm_df

churndataNormalised=my_normalize(churn_numeric)
churndataNormalised["Gender"]=churndata["Gender"]
churndataNormalised.boxplot(rot=90, fontsize=10)

            # Customer Age,Months_on_book, Months_Inactive_12_mon, Contacts_Count_12_mon,
            # Total_Amt_Chng_Q4_Q1, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1 has outliers..


churn_numeric.hist(figsize=(20,20))
            
Quantiles=churn_numeric.quantile(q=0.9)

out_df=churndata[churndata["Customer_Age"]<=Quantiles.Customer_Age]
out_df=out_df[out_df["Credit_Limit"]<=Quantiles.Credit_Limit]
out_df=out_df[out_df["Avg_Open_To_Buy"]<=Quantiles.Avg_Open_To_Buy]
churndata_without_outlier=out_df[out_df["Total_Trans_Amt"]<=Quantiles.Total_Trans_Amt]
del out_df
my_normalize(churndata_without_outlier[NumericalVar]).boxplot(NumericalVar,rot=90, fontsize=10)

## In here we can see that there are also some outliers below the quantiles.
## However, other values are within a limited range. Therefore, lets use like in this way.

corrMatrix = churndata_without_outlier[NumericalVar].corr()
#sn.heatmap(corrMatrix, annot=False)
#plt.show()

## Below there is a correlation extractor function.
def ExtractCorr(data,cor):
    colu=data.columns
    for col in colu:
        Correlated=data[data[col]>cor][col]
        print(Correlated)
        
ExtractCorr(corrMatrix,0.85)

### Corelation
#Credit_Limit       1.000000
#Avg_Open_To_Buy    0.986728
## Therefore we can exclude Avg_Open_To_Buy from our dataset


churndata1=churndata_without_outlier
churndata1= churndata1.drop(["Avg_Open_To_Buy","CLIENTNUM"], axis=1)
NumericalVar =["Customer_Age","Dependent_count","Months_on_book","Total_Relationship_Count",
               "Months_Inactive_12_mon","Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal",
               "Total_Amt_Chng_Q4_Q1","Total_Trans_Amt","Total_Trans_Ct","Total_Ct_Chng_Q4_Q1",
               "Avg_Utilization_Ratio"]


# Log transforming skew variables
skew_limit = 0.75 # define a limit above which we will log transform
skew_vals = churndata1[NumericalVar].skew()

skew_cols = (skew_vals
.sort_values(ascending=False)
.to_frame()
.rename(columns={0:'Skew'})
.query('abs(Skew) > {}'.format(skew_limit)))

## We have 4 variables skewed more than 0.75.
## This situation actually seen in hist plot also.

field = "Total_Ct_Chng_Q4_Q1"

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))

churndata1[field].hist(ax=ax_before)
churndata1[field].apply(np.log1p).hist(ax=ax_after)

ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
fig.suptitle('Field "{}"'.format(field))

for col in skew_cols.index.values:
    if col == "CLIENTNUM":
        continue 
    churndata1[col]=churndata1[col].apply(np.log1p)

### Log transformation has been completed lets look hist aganin

churndata1[NumericalVar].skew()
### There is no skewed variables more than 0.75.
## Lets continue

sn.set_theme(style="ticks")
Corrplot=churndata[NumericalVar].sample(frac=0.5, replace=True, random_state=1)
Corrplot["Attrition_Flag"]=churndata["Attrition_Flag"]

## sn.pairplot(Corrplot, hue="Attrition_Flag")


## Let add some new features that can provide more d
## Credit_Limit and Avg_Utilization_Ratio
## Total_Trans_Amt and Total_Trans_Ct are non linear we can add interaction term among these
churndata1["Total_Trans_Amt_X_Total_Trans_Ct"]=churndata1["Total_Trans_Ct"]*churndata1["Total_Trans_Amt"]
churndata1["Credit_Limit_X_Avg_Utilization_Ratio"]=churndata1["Avg_Utilization_Ratio"]*churndata1["Credit_Limit"]

## New features from categories.

churndata2=churndata1.copy()

def add_deviation_feature(X, feature, category):
    # temp groupby object
    category_gb = X.groupby(category)[feature]
    # create category means and standard deviations for each observation
    category_mean = category_gb.transform(lambda x: x.mean())
    category_std = category_gb.transform(lambda x: x.std())
    # compute stds from category mean for each feature value,
    # add to X as new feature
    deviation_feature = (X[feature] - category_mean) / category_std
    X[feature + '_Dev_' + category] = deviation_feature

add_deviation_feature(churndata2, 'Credit_Limit', 'Education_Level')
add_deviation_feature(churndata2, 'Total_Ct_Chng_Q4_Q1', 'Income_Category')


mask = churndata2.dtypes != np.object
float_cols = churndata2.columns[mask]

Corrplot=churndata2[float_cols].sample(frac=0.5, replace=True, random_state=1)
Corrplot["Attrition_Flag"]=churndata["Attrition_Flag"]
#sns_plot = sn.pairplot(Corrplot, hue="Attrition_Flag", height=6)
#sns_plot.savefig("Histogram_After_LogTransform.png")



## One-hot encoding the dummy variables:
churndata3=churndata2.copy()
churndata3 = pd.get_dummies(churndata3, columns=CategoricalVar, drop_first=True)
churndata3.describe().T

#churndata3.to_csv("ChurnData_Explored.csv")



### Hypothesis Testing
### This kind of approach exactly clear our predictivity regarding variables.
### We plot probability density functions of featues by churning customer 
### in order to reveal if there is any difference by these features
from scipy.stats import binom
from scipy import stats 
import math

def plot_ppf(data,feature):
 
    mu1 = data[data["Attrition_Flag"]=="Existing Customer"][feature].mean()
    variance1 = data[data["Attrition_Flag"]=="Existing Customer"][feature].var()
    sigma1 = math.sqrt(variance1)
    x1 = data[data["Attrition_Flag"]=="Existing Customer"][feature].sort_values(ascending=False)
    plt.plot(x1,stats.norm.pdf(x1, mu1, sigma1))

    mu2 = data[data["Attrition_Flag"]=="Attrited Customer"][feature].mean()
    variance2 = data[data["Attrition_Flag"]=="Attrited Customer"][feature].var()
    sigma2 = math.sqrt(variance2)
    x2 = data[data["Attrition_Flag"]=="Attrited Customer"][feature].sort_values(ascending=False)
    plt.plot(x2,stats.norm.pdf(x2, mu2, sigma2))
    plt.title(feature)

    plt.show()
    
for col in float_cols:
        plot_ppf(churndata2,col)
        
# As we can see that normal distribution is satisfied in each feature.
# However, one of the most significant inference from pdf is the similarity of
# each functions by output value. This make our prediction hard.



#### Fitting the base data set

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

df_Without_EDA=pd.get_dummies(churndata, columns=CategoricalVar, drop_first=True)

def classification_knn(data,k):
    y, X = data['Attrition_Flag_Existing Customer'], data.drop(columns='Attrition_Flag_Existing Customer')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # Preciision, recall, f-score from the multi-class support function
    print(classification_report(y_test, y_pred))
    print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
    print('F1 Score: ', round(f1_score(y_test, y_pred), 2))

classification_knn(df_Without_EDA,4)
classification_knn(churndata3,4)

### The accuracy score increase from % 75 to % 89 after some EDA phases with simlee Knn Classification algorithm