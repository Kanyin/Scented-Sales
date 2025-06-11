#!/usr/bin/env python
# coding: utf-8

# Here we clean the dataset and add a number of important variables.

# In[177]:


import pandas as pd
import os
import re
import jellyfish


# In[178]:


os.chdir("/Users/kanyin/Documents/Kanyin/Data Science/Data Projects/Perfume E-Commerce")


# In[179]:


perfm=pd.read_csv("Data/ebay_mens_perfume.csv")
perfm["Sex"]="Male"
perfw=pd.read_csv("Data/ebay_womens_perfume.csv")
perfw["Sex"]="Female"


# In[180]:


perfm.info()


# In[181]:


perfw.info()


# Since they have the same columns lets combine.

# In[182]:


perf=pd.concat([perfm,perfw], axis=0)
perf.info()


# In[183]:


#perf.head(10)
#perf.tail(10)


# Let's first start with a random forest prediction model of the data.

# In[184]:


from difflib import SequenceMatcher as sq
def similar(this, that):
    return sq(None, a, b).ratio()


# In[185]:


cnts=perf['title'].value_counts()
names=perf['title'].unique()
names


# In[186]:


# I find that many of the titles have ounce information. To find duplicates, it may be important to remove them
pattern = r'\b\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?\s*(?:[Ff][Ll]\s*)?[Oo][Zz]\b|\b\d+(?:\.\d+)?\s*[Mm][Ll]\b'
perf['size']=perf['title'].str.findall(pattern)


# In[187]:


perf['title']=perf['title'].str.replace(pattern, '', regex=True)
perf.head(15)


# In[188]:


perf[['Available', 'Sold']]=perf['availableText'].str.split('/', expand=True)


# In[189]:


import datetime as dt
perf['lastUpdated']=perf['lastUpdated'].str.replace(r'\bPDT\b|\bPST\b|\bEST\b', '', regex=True).str.strip()
perf['lastUpdated']=pd.to_datetime(perf['lastUpdated'], errors='coerce')


# In[190]:


perf.head(3)


# In[191]:


# Have a categorical variable for more than 10 or less than 10 available
perf['Sold']=perf['Sold'].str.replace("sold", "").str.strip()
perf['Sold']=perf['Sold'].str.replace(",", "")
perf['Sold']=perf['Sold'].str.replace('vendidos', "")
perf['Sold']=pd.to_numeric(perf['Sold'])


# In[192]:


perf['brand']=perf['brand'].astype(str)
def to_rebrand(brand):
    return( not brand or 
            brand.lower() == 'unbranded' or
            brand.lower().startswith('as')
          )

def rebrand(row):
    global new_brand
    if to_rebrand(row['brand']):
        print(f"\nTitle: {row['title']}")
        print(f"Current brand: '{row['brand']}'")
        new_brand = input("Enter correct brand (or press Enter to keep existing): ")
        return new_brand if new_brand else row['brand']
    else:
        return row['brand']

perf['Label']=perf.apply(rebrand, axis=1)


# In[199]:


perf['Label'].head(5)


# In[200]:


#Change YSLs to Yves Saint Laurent
#Change some unknowns to Assorted
#Victor & Rold to Viktor & Rolf
#Ralph Lauren to Polo Ralph Lauren
perf['Label']=perf['Label'].str.replace('Victor', 'Viktor', regex=True).str.replace('Ralph', 'Polo Ralph')
perf['Label']=perf['Label'].str.replace('Dolce&Gabbana', 'Dolce & Gabbana')
perf['Label']=perf['Label'].astype(str).str.title()
perf.tail(10)


# In[202]:


perf.info()


# In[203]:


#perfog=pd.concat([perfm,perfw], axis=0)
#perfog['size']=perf['size']
perf['size']=perf['size'].astype(str)
perf['size']=perf['size'].str.replace(r"[^0-9.]", '', regex=True).str.replace(r'100','3.4',regex=True).str.replace(r'(\.\d).*', r'\1',regex=True)
perf['size']=perf['size'].str.replace(r'[\[\]]', '', regex=True).str.replace(r',.*', '', regex=True).str.replace(r'.*/', '', regex=True)
perf['size']=perf['size'].str.replace(r'3.3', '3.4').str.strip()
perf['size']=pd.to_numeric(perf['size'])
perf['size']


# In[205]:


perf.to_csv('Data/backup.csv', index=False) #saved as backup


# In[210]:


#drop redundant columns
perf=perf.drop(['brand', 'priceWithCurrency','availableText'], axis=1)
perf.rename(columns={'Label': 'brand'}, inplace=True)


# In[217]:


#perf=perf.drop(['priceWithCurrency', 'availableText'],axis=1)
perf


# In[218]:


perf.to_csv('Data/perf.csv', index=False)

