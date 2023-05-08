#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import re

def whitespace_remover(dataframe):
   
    # iterating over the columns
    for i in dataframe.columns:
         
        # checking datatype of each columns
        if dataframe[i].dtype == 'object':
             
            # applying strip function on column
            dataframe[i] = dataframe[i].map(str.strip)
        else:
             
            # if condn. is False then it will do nothing.
            pass
    return(dataframe)


# In[10]:


messy_df = pd.read_csv("../data/census.csv")

## first clean columns
col1 = [re.sub("\\s+", "", col) for col in messy_df.columns]
messy_df.columns = col1

## then, clean whitespace itself
clean_df = whitespace_remover(messy_df)
clean_df.to_csv("../data/census_clean.csv", index = False)

