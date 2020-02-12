# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:21:08 2020

@author: Dell
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import fmin
import os
import datetime

dataLocations=[r'C:\Users\Dell\Desktop\data\GOIL VWAP closing prces.xlsx',\
              r'C:\Users\Dell\Desktop\data\SCB VWAP closing prces.xlsx',\
              r'C:\Users\Dell\Desktop\data\SIC VWAP closing prces.xlsx']
riskFreeRate=0.0003
dated=[]
returnData=[]
uploadedData=[]
logReturnsList=[]
#retrieve data downloaded from the internet
def retriveData(ticker):
        for tick in ticker:
            data=pd.read_excel(tick,parse_dates=[1]) 
            uploadedData.append(data)
        return uploadedData

#finding the annual return by grouping years and summing them together
def annualReturn(ticker):
    ticker=retriveData(ticker)
    for tick in ticker:
        #finding log returns of daily data
        logReturns=sp.log(tick['Closing Price VWAP (GHS)'][1:].values/\
                      tick['Closing Price VWAP (GHS)'][:-1].values) 
        logReturnsList.append(logReturns)
        #declaring the length of daily log returns
        x=len(logReturnsList[0])
    #using the length of returns to append the suitable number of dates   
    for _ in range(0,x):
             #getting the dates from the first item on list
             dates=ticker[0]['Date']
             #appending onlu 4 characters of date whixch are the year
             dated.append(dates[_][:4])
    indi=pd.Series(dated)
    #dataframe happens to be misshaped so we transform it
    dataTable=pd.DataFrame(logReturnsList).T
    dataTable.index=indi
    dataTable.columns=['GOIL','SCB','SIC']
    groupbys=sp.exp(dataTable.groupby(dataTable.index).sum())-1
    return groupbys

#find the portfolio variance used to calculate the sharpe ratio
def portfolioVariance(R,w):
    #find the correlation coefficient but before transform returns into three columns
    corr=sp.corrcoef(R.T)
    #find deviation along column axis
    standarDeviation=sp.std(R,axis=0)
    var=0.0
    n=len(w)
    #since we have weights and standard deviation,we find variances by pairing 
    #permutatively 2 stocks,find the vaariances and sum them as we do
    for i in range(n):
        for j in range(n):
            var+=w[i]*w[j]*standarDeviation[i]*standarDeviation[j]*corr[i,j]
    return var   
        

def sharpeRatio(R,w):
    var=portfolioVariance(R, w)
    meanReturn=sp.mean(R,axis=0)
    returns=sp.array(meanReturn)
    #sp.dot finds expected returns by multiplying weights with mean returns
    return (sp.dot(w,returns)-riskFreeRate)/sp.sqrt(var)

#turn database returns into scipy array
annualReturnArray=sp.array(annualReturn(dataLocations))

def sharpeNMinusOneStock(w):
    w2=sp.append(w,1-sum(w))
    return -sharpeRatio(annualReturnArray, w2)

#annualReturnArray=sp.array(annualReturn(dataLocations))
print('Efficient Portfolio Allocation Process')
print('Stocks used are:GOIL,SCB,SIC')
equalWeights=sp.ones(3,dtype=float)*1/3
print('Sharpe ratio for an Equally Weighted Portfolio:'\
      ,sharpeRatio(annualReturnArray,equalWeights))
print()

#if for n stocks we could only choose n-1 weights
weight0=sp.ones(3-1,dtype=float)*1/3
weight1=fmin(sharpeNMinusOneStock,weight0)
finalWeight=sp.append(weight1,1-sum(weight1))
finalSharpe=sharpeRatio(annualReturnArray,finalWeight)
print('optimal weights are:',finalWeight)
print('final sharpe Ratio is:',finalSharpe)

    
