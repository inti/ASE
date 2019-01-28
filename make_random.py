import pandas as pd
import numpy as np
import sys
import argparse
import logging

parser = argparse.ArgumentParser(description='Estimate Allelic Specific Expression probabilities')
parser.add_argument('--input', type=str, default=None, help='Input file to consider')
parser.add_argument('--output', type=str, default=None, help='Input files to consider')
parser.add_argument('--scale', type=int, default=1, help='Multiply readth length by this')
parser.add_argument('--p', type=float, default=0.5, help='Proportion for null')
parser.add_argument('--seed', type=int, default=1, help='Seed for random number generator')
args = parser.parse_args()


np.random.seed(args.seed)

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging_level = logging.INFO
logger.setLevel(logging_level)

data = pd.read_table(args.input)

logger.info("Data IN")
logger.info(data.head())

data.loc[:,"totalCount"] = data.totalCount.values*args.scale
data.loc[:,"aCount"] = np.random.binomial(n=data.totalCount.values,p=float(args.p))
data.loc[:,"bCount"] = data.loc[:,"totalCount"] - data.loc[:,"aCount"]

logger.info("Data OUT")
logger.info(data.head())


data.to_csv(args.output,sep="\t",header=True,index=False)

logger.info("\n\nDone\n\n")
print("hello")
