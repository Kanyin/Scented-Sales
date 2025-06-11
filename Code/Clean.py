#!/usr/bin/env python
# coding: utf-8

# Here we clean the dataset and add a number of important variables.

import pandas as pd
import os
import re
import jellyfish


os.chdir("/Users/kanyin/Documents/Kanyin/Data Science/Data Projects/Perfume E-Commerce")


perfm=pd.read_csv("Data/ebay_mens_perfume.csv")
perfm["Sex"]="Male"
perfw=pd.read_csv("Data/ebay_womens_perfume.csv")
perfw["Sex"]="Female"


perfm.info()

# Since they have the same columns lets combine.


perf=pd.concat([perfm,perfw], axis=0)
perf.info()

# Let's first start with a random forest prediction model of the data.


from difflib import SequenceMatcher as sq
def similar(this, that):
    return sq(None, a, b).ratio()


cnts=perf['title'].value_counts()
names=perf['title'].unique()
names


# I find that many of the titles have ounce information. To find duplicates, it may be important to remove them
pattern = r'\b\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?\s*(?:[Ff][Ll]\s*)?[Oo][Zz]\b|\b\d+(?:\.\d+)?\s*[Mm][Ll]\b'
perf['size']=perf['title'].str.findall(pattern)



perf['title']=perf['title'].str.replace(pattern, '', regex=True)
perf.head(15)



perf[['Available', 'Sold']]=perf['availableText'].str.split('/', expand=True)



import datetime as dt
perf['lastUpdated']=perf['lastUpdated'].str.replace(r'\bPDT\b|\bPST\b|\bEST\b', '', regex=True).str.strip()
perf['lastUpdated']=pd.to_datetime(perf['lastUpdated'], errors='coerce')



# Have a categorical variable for more than 10 or less than 10 available
perf['Sold']=perf['Sold'].str.replace("sold", "").str.strip()
perf['Sold']=perf['Sold'].str.replace(",", "")
perf['Sold']=perf['Sold'].str.replace('vendidos', "")
perf['Sold']=pd.to_numeric(perf['Sold'])





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


perf['Label'].head(5)





#Change YSLs to Yves Saint Laurent
#Change some unknowns to Assorted
#Victor & Rold to Viktor & Rolf
#Ralph Lauren to Polo Ralph Lauren
perf['Label']=perf['Label'].str.replace('Victor', 'Viktor', regex=True).str.replace('Ralph', 'Polo Ralph')
perf['Label']=perf['Label'].str.replace('Dolce&Gabbana', 'Dolce & Gabbana')
perf['Label']=perf['Label'].astype(str).str.title()
perf.tail(10)



perf.info()



#perfog=pd.concat([perfm,perfw], axis=0)
#perfog['size']=perf['size']
perf['size']=perf['size'].astype(str)
perf['size']=perf['size'].str.replace(r"[^0-9.]", '', regex=True).str.replace(r'100','3.4',regex=True).str.replace(r'(\.\d).*', r'\1',regex=True)
perf['size']=perf['size'].str.replace(r'[\[\]]', '', regex=True).str.replace(r',.*', '', regex=True).str.replace(r'.*/', '', regex=True)
perf['size']=perf['size'].str.replace(r'3.3', '3.4').str.strip()
perf['size']=pd.to_numeric(perf['size'])
perf['size']



perf.to_csv('Data/backup.csv', index=False) #saved as backup




#drop redundant columns
perf=perf.drop(['brand', 'priceWithCurrency','availableText'], axis=1)
perf.rename(columns={'Label': 'brand'}, inplace=True)



#perf=perf.drop(['priceWithCurrency', 'availableText'],axis=1)
perf





perf.to_csv('Data/perf.csv', index=False)

