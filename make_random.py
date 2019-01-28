import pandas as pd
import numpy as np
import sys

print("Arguments: ")
print("Binomial_Probability:" + sys.argv[1])
print("filein: " + sys.argv[2])
print("fileout: " + sys.argv[3])
print("")

data = pd.read_table(sys.argv[2])

print("Data IN")
print(data.head())
print("")

data.loc[:,"aCount"] = np.random.binomial(n=data.totalCount.values,p=float(sys.argv[1]))
data.loc[:,"bCount"] = data.loc[:,"totalCount"] - data.loc[:,"aCount"]

print("Data OUT")
print(data.head())
print("")

data.to_csv(sys.argv[3],sep="\t",header=True,index=False)

print("\n\nDone\n\n")

