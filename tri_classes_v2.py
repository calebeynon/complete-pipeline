# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:04:25 2024

@author: caleb
"""

import pandas as pd
import numpy as np
import chainladder as cl
from scipy.stats.mstats import winsorize
import pickle

def formatTriangle(triangle):
    ## Get the triangle in the right format for the cl package (tabular)
    CYarray = []
    AYarray = []
    incurred = []
    for AY in triangle['ay']:
        df = triangle[triangle['ay']==AY] #Filter to the correct year
        length = (df.iloc[0] != 0).sum() - 3 #Find the number of non zero entries in that row (how much development)
        CYarray += np.arange(AY,AY+length,1).tolist() #Get the CY's, by incrementing previous year
        AYarray += np.array([AY]).repeat(length).tolist() # Get the AY's by just repeating the AY we're working with n times
        incurred += [x for x in df.iloc[0].tolist()[3:] if x !=0] #Get the paid using the loss triangle and creating a list, filter to non zero entries
    ##

    return pd.DataFrame({'AY':AYarray,'CY':CYarray,'paid':incurred}) #Convert to a df

def bf_ultimate(apriori,obj,npw):
    cdfs = pd.melt(obj.cdf_.to_frame(),value_name='cdf',var_name='time')
    cdfs['flip'] = 1 - (1/cdfs['cdf'])
    cdfs =  cdfs.iloc[:-1]
    cur_paid = obj.latest_diagonal.to_frame().iloc[::-1].reset_index(drop=True)
    cur_paid.columns = ['cur_paid']
    df = pd.DataFrame({'factor':cdfs['flip'],'cur_paid':cur_paid['cur_paid']})
    df['ibnr'] = apriori*npw*df['factor']
    df['ult'] = df['ibnr']+df['cur_paid']
    return df['ult'].sum()



class Industry:
    """
    Industry object. Holds a year dictionary where each value points to the Year object for that particular year
    loc function points to a specific company object using a year and cocode reference
    """
    def __init__(self, paid, incurred):
        self.year = {}
        self.paid = paid
        self.incurred = incurred
        pairs = self.paid[['year','cocode']].drop_duplicates()
        self.ygls = pairs.values.tolist()
        
        for year,cocode in self.ygls:
            if year not in self.year:
                self.year[year] = Year(self,year)

    def loc(self,yr,cocode):
        return self.year[yr].company[cocode]
    
    
class Year:
    """
    Year object. Holds company dictionary where each value points to the company object for that cocode
    """
    def __init__(self,ind,yr):
        self.ind = ind
        self.company = {}
        self.yr = yr
        
        for year, cocode in self.ind.ygls:
            if cocode not in self.company:
                self.company[cocode] = Company(self.ind,self,cocode)
         
class Company:
    """
    Company object. Holds all the data for a specific cocode. Includes, paid triangle, mack object, mack df, and mack se
    Also contains function mackdf() which takes in a years ahead as input and returns the triangle from that many years ahead if it exists
    """
    def __init__(self,ind,year,cocode):
        self.year = year
        self.ind = ind
        self.cocode = cocode
        self._paid = None #Lazy initialization
        self._incurred = None
        self._mack = None #Lazy initialization
        self._macki = None
        self._mackdf = None 
        self._mackse = None
        self._fortri = None
        self._boot = None
        self._dev = None
        self._model = None
        self._full_triangle = None
        self._CL_increase = None
    
    
    def mackdf_(self, yearsAhead):
        try:
            newtri = self.ind.loc(self.year.yr+yearsAhead,self.cocode).mackdf
            return newtri
        except (KeyError, IndexError,AttributeError) as e:
            #print(f'Error: {e}')
            return None
    @property
    def paid(self):
        if self._paid is None:
            self._paid = self.ind.paid[
                (self.ind.paid['year'] == self.year.yr) &
                (self.ind.paid['cocode'] == self.cocode)
                ]
        return self._paid
    
    @property
    def incurred(self):
        if self._incurred is None:
            self._incurred = self.ind.incurred[
                (self.ind.incurred['year'] == self.year.yr) &
                (self.ind.incurred['cocode'] == self.cocode)
                ]
        return self._incurred
    
    @property
    def mack(self):
        if self._mack is None:
            tri = cl.Triangle(formatTriangle(self.paid), origin = 'AY', development = 'CY',columns = 'paid', cumulative = True)
            self._mack = cl.MackChainladder().fit(tri)
        return self._mack

    @property
    def macki(self):
        if self._macki is None:
            tri = cl.Triangle(formatTriangle(self.incurred), origin = 'AY', development = 'CY',columns = 'paid', cumulative = True)
            self._macki = cl.MackChainladder().fit(tri)
        return self._macki
    
    @property
    def mackdf(self):
        if self._mackdf is None:
            self._mackdf = self.mack.summary_.to_frame(origin_as_datetime=True)
            s = self._mackdf['Mack Std Err'].sum()
            totals = self.mackse
            self._mackdf['Prop SE'] = self._mackdf['Mack Std Err']/s
            self._mackdf['DAMD'] = 2*totals*self._mackdf['Prop SE']
            self._mackdf['ABE+DAMD'] = self._mackdf['DAMD'] + self._mackdf['Ultimate']
        return self._mackdf

    @property
    def dev(self):
        if self._dev is None:
            tri = cl.Triangle(formatTriangle(self.paid), origin = 'AY', development = 'CY',columns = 'paid', cumulative = True)
            self._dev = cl.Development().fit_transform(tri)
        return self._dev
    
    @property
    def model(self):
        if self._model is None:
            self._model = cl.Chainladder().fit(self.dev)
        return self._model

    @property
    def full_triangle(self):
        if self._full_triangle is None:
            self._full_triangle = self.model.full_triangle_
        return self._full_triangle
    
    @property
    def mackse(self):
        if self._mackse is None:
            self._mackse = self.mack.total_mack_std_err_['paid'][0]
        return self._mackse
    
    @property
    def retTri(self):
        if self._fortri is None:
            self._fortri = formatTriangle(self.paid)
        return self._fortri

    @property
    def CL_increase(self):
        if self._CL_increase is None:
            try:
                _ult = self.ind.loc(self.year.yr-1,self.cocode).model.ultimate_.sum()
            except Exception:
                return 2
            ult_ = self.model.ultimate_.sum()
            self._CL_increase = ult_-_ult > 0
            return self._CL_increase
        
    def boot(self,n):
        if self._boot is None:
            samples = cl.BootstrapODPSample(n_sims = n).fit(cl.Triangle(self.retTri,origin = 'AY',development = 'CY',columns ='paid',cumulative = True )).resampled_triangles_
            self._boot = cl.Chainladder().fit(samples)
        return self._boot
    
if __name__ == "__main__":
    import time
    import sys
    import ast
    t0 = time.time()
    p = pd.read_csv('paid triangles - 2024-0823.csv')
    i = pd.read_csv("incurred triangles - 2024-0823.csv")
    ind = Industry(p,i)
    
    ##Get the mackdf and se for each company in the dataset 
    dic = {}
    for ygl in ind.ygls:
        t1 = time.time()
        obj = ind.loc(ygl[0],ygl[1])
        dic[str(ygl)] = [obj.mackdf,obj.mackse]
        print(f'ygl: {ygl}. Iteration time = {time.time()-t1} seconds. Total time:{time.time()-t0} seconds.')
    print(f'total time: {(time.time()-t0)/60} minutes.')
    print(f"Size of dictionary: {sys.getsizeof(dic)} bytes")
    with open('AMDAOdictionary.pkl', 'wb') as file:
        pickle.dump(dic, file)
    
    
    with open('/Users/caleb/CL/AMDAOdictionary.pkl','rb') as file:
        dic = pickle.load(file)
    ## Get the two states plus the delta TREE for each company if it also has a triangle 5 years in the future
    t0 = time.time()
    A = {}
    for key, value in dic.items():
        t1 = time.time()
        listkey = ast.literal_eval(key)
        t2 = time.time()
        se = value[1]
        tDAMD = value[0]['ABE+DAMD'].dropna().tail(5) #Discretion
        tDA0 = value[0]['Ultimate'].dropna().tail(5) #No discretion
        t3 = time.time()        
        t5ult = ind.loc(listkey[0],listkey[1]).mackdf_(5)
        if t5ult is not None: 
            t5ult = t5ult['Ultimate'][:5]
        else: continue
        t4 = time.time()    
        if len(tDAMD)+len(tDA0)+len(t5ult) != 15: continue
        deltaTREE = (tDAMD.sum()-tDA0.sum())/se
        states = [(-1*(t5ult.sum() - tDA0.sum())/se),(-1*(t5ult.sum() - tDAMD.sum())/se)]
        t5 = time.time()
        A[key] = [states[0],states[1],deltaTREE]
        t6 = time.time()
        total = t6-t1
        print(f'Iteration: {key}. Iteration time: {total} seconds. Total time: {time.time()-t0} seconds.')
        
        
        
    #Winsorizing
    untr = [v[0] for v in A.values()]
    tr = [v[1] for v in A.values()]
    untr = winsorize(np.array(untr),limits = [0.01,0.01])
    tr = winsorize(np.array(tr),limits = [0.01,0.01])
    for (key,value), wvalue0, wvalue1 in zip(A.items(),untr,tr):
        value[0] = wvalue0
        value[1] = wvalue1
    
    with open('statesdictionary.pkl', 'wb') as file:
        pickle.dump(A,file)













