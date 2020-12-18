import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;



dataset=pd.read_csv("x.csv")

X=dataset.iloc[:,1:-1]


#People who bought this also bought!!!

"""

support ---->like probability of each item lets say m1

----m1/total

confidence ---> who bought  (m1 as m2) /total

lift ---->  confidence/support

"""

#prepare the input properly

transactions=[];

for i in range(0,9835):
    transactions.append(  [ str(X.values[i,j]) for j in range(0,31) ] )



from apyori import apriori

"""
support--->transaction containing a particular item /total transactions


"""
rules=apriori(transactions,min_support=0.002,min_confidence=0.2,min_lift= 3,min_length=2)

results=list(rules)

print(results)


"""
setup minimum support and confidence

take all subsets in transactions havinh higher support than
minimum support 

take all the rules of these subsets having higher confidence 
than minimum confidence

sort the rules by decresing lift

"""