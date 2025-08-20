# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:27:16 2025

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
        self._mack = None #Lazy initialization
        self._mackdf = None 
        self._mackse = None
        self._fortri = None
        self._boot = None
        self._dev = None
    
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
    def dev(self):
        if self._dev is None:
            tri = cl.Triangle(formatTriangle(self.paid), origin = 'AY', development = 'CY',columns = 'paid', cumulative = True)
            self._dev = cl.Development().fit(tri)
        return self._dev
    
    @property
    def mack(self):
        if self._mack is None:
            tri = cl.Triangle(formatTriangle(self.paid), origin = 'AY', development = 'CY',columns = 'paid', cumulative = True)
            self._mack = cl.MackChainladder().fit(tri)
        return self._mack

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
    def mackse(self):
        if self._mackse is None:
            self._mackse = self.mack.total_mack_std_err_['paid'][0]
        return self._mackse
    
    @property
    def retTri(self):
        if self._fortri is None:
            self._fortri = formatTriangle(self.paid)
        return self._fortri

    def boot(self,n):
        if self._boot is None:
            samples = cl.BootstrapODPSample(n_sims = n).fit(cl.Triangle(self.retTri,origin = 'AY',development = 'CY',columns ='paid',cumulative = True )).resampled_triangles_
            self._boot = cl.Chainladder().fit(samples)
        return self._boot
    