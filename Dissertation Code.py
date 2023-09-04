#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
df = pd.read_csv('ipumsi_00004.csv')


# # Initial Preprocessing and Cleansing

# In[2]:


df['COUNTRY'].value_counts()


# In[3]:


#Looking at the empty cells

empty_cells_count = df.isnull().sum()
total_cells_count = len(df)
non_empty_cells_count = total_cells_count - empty_cells_count

result_table = pd.DataFrame({
    'Empty Cells Count': empty_cells_count,
    'Non-Empty Cells Count': non_empty_cells_count
})

print(result_table)


# In[4]:


#Looking at value counts for each column to remove unkown values
for column in df.columns:
    print(f"Column: {column}")
    print(df[column].value_counts())
    print("\n")


# In[5]:


# # Filtering out empty/unknown cells
conditions = [
    (df['OWNERSHIP'].isin([9])),
    (df['OWNERSHIPD'].isin([999])),
    (df['WATSUP'].isin([99])),
    (df['HEAT'].isin([9])),
    (df['ROOMS'].isin([98, 99])),
    (df['HHTYPE'] == 99),
    (df['ELDCH'] == 99),
    (df['YNGCH'] == 99),
    (df['RELATE'] == 9),
    (df['RELATED'] == 9999),
    (df['AGE'] == 999),
    (df['NCHLT5'] == 98),
    (df['MARST'].isin([9])),
    (df['EDATTAIN'].isin([9])),
    (df['EDATTAIND'].isin([999])),
    (df['EEMPSTAT'].isin([999])),
    (df['LABFORCE'].isin([8, 9])),
    (df['OCCISCO'].isin([97, 98])),
    (df['INDGEN'].isin([998, 999])),
    (df['ECLASSWK'].isin([6, 9])),
    df['WATSUP'].isna()
]

filtered_df = df[~pd.concat(conditions, axis=1).any(axis=1)]


# In[6]:


# #Removing Household Weight and Person Weight (flat weights)
filtered_df=filtered_df.drop('HHWT',axis=1)
filtered_df=filtered_df.drop('PERWT',axis=1)


# In[7]:


filtered_df.info()


# In[8]:


#looking at the empty cells

empty_cells_count = filtered_df.isnull().sum()
total_cells_count = len(filtered_df)
non_empty_cells_count = total_cells_count - empty_cells_count

result_table = pd.DataFrame({
    'Empty Cells Count': empty_cells_count,
    'Non-Empty Cells Count': non_empty_cells_count
})

print(result_table)


# In[9]:


# Dropping features with empty rows
filtered_df=filtered_df.drop('OWNERSHIP',axis=1)
filtered_df=filtered_df.drop('OWNERSHIPD',axis=1)

# checking for any more empty cells
empty_cells_count = filtered_df.isnull().sum()
total_cells_count = len(filtered_df)
non_empty_cells_count = total_cells_count - empty_cells_count

result_table = pd.DataFrame({
    'Empty Cells Count': empty_cells_count,
    'Non-Empty Cells Count': non_empty_cells_count
})

print(result_table)


# In[10]:


# Mapping dictionary for category reduction
watsup_mapping = {
    0.0:0,
    10.0:1,
    11.0:1,
    15.0:1,
    16.0:1,
    17.0:1,
    20.0:2,
}
filtered_df['WATSUP'] = filtered_df['WATSUP'].replace(watsup_mapping)


# In[11]:


# re-arranging the values such that higher values correlate to an increased well-being
filtered_df['WATSUP'] = filtered_df['WATSUP'].replace({1: 2, 2: 1})


# In[12]:


# Mapping dictionary for category reduction
heat_mapping = {
    0.0:0,
    1.0:1,
    2.0:2,
    3.0:2,
    4.0:2,
    5.0:2,
}

filtered_df['HEAT'] = filtered_df['HEAT'].replace(heat_mapping)


# In[13]:


#Recoding 'MOMLOC' to identify the presence/absence of mothers
column_to_recode = 'MOMLOC'

filtered_df[column_to_recode] = filtered_df[column_to_recode].apply(lambda x: 1 if x > 0 else x)


# In[14]:


#Recoding 'POPLOC' to identify the presence/absence of fathers
column_to_recode = 'POPLOC'

filtered_df[column_to_recode] = filtered_df[column_to_recode].apply(lambda x: 1 if x > 0 else x)


# In[15]:


#Recoding 'SPLOC' to identify the presence/absence of spouses
column_to_recode = 'SPLOC'

filtered_df[column_to_recode] = filtered_df[column_to_recode].apply(lambda x: 1 if x > 0 else x)


# # Univariate Analysis of Non-category Variables

# In[16]:


# Creating subplots for univariate analysis
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(20, 20))
axes = axes.flatten()

column_names = ['PERSONS','ROOMS','NFAMS','NCOUPLES','NMOTHERS','NFATHERS','FAMSIZE','NCHILD','NCHLT5','ELDCH','YNGCH','AGE']

# Dictionary to map column names to custom titles
title_mapping = {
    'PERSONS': "Persons",
    'ROOMS': 'Number of Rooms',
    'NFAMS':'Number of Families',
    'NCOUPLES': 'Number of Couples',
    'NMOTHERS': 'Number of Mothers',
    'NFATHERS': 'Number of Fathers',
    'FAMSIZE':'Family Size',
    'NCHILD':'Number of Children',
    'NCHLT5':'Number of Children Under 5',
    "ELDCH":'Age of Eldest Child',
    'YNGCH':'Age of Youngest Child',
    'AGE':'Age',
}
# Customising each subplot
for i, column in enumerate(column_names):
    ax = axes[i]
    filtered_df[column].hist(ax=ax, bins=50, color='pink')

    ax.set_title(title_mapping.get(column, column))

plt.show()


# # Univariate Analysis of Category Variables 

# In[17]:


# Univariate Analysis for Country
country_labels = {
    40: 'Austria',
    300: 'Greece',
    348: 'Hungary',
    616: 'Poland',
    620: 'Portugal',
    642: 'Romania',
    724: 'Spain'
}

country_counts = filtered_df['COUNTRY'].value_counts()
plt.figure(figsize=(12, 6)) 
plt.bar([country_labels[code] for code in country_counts.index], country_counts.values, color='purple')

plt.xlabel('Country')
plt.ylabel('Frequency')
plt.title('Frequency of Each Country')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[18]:


# Univariate Analysis for Water Supply
watsup_labels = {
    0:'N/A',
    1:'No Piped Water',
    2:'Piped Water'
}
watsup_counts = filtered_df['WATSUP'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([watsup_labels[code] for code in watsup_counts.index], watsup_counts.values, color='purple')

plt.xlabel('Water Supply')
plt.ylabel('Frequency')
plt.title('Frequency by Water Supply')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[19]:


# Univariate Analysis for Educational Attainment
edattain_labels = {
    1: 'Less than Primary Completed',
    2: 'Primary Completed',
    3: 'Secondary Completed',
    4: 'University Completed'
}
edattain_counts = filtered_df['EDATTAIN'].value_counts().sort_index()
plt.figure(figsize=(15, 6)) 
plt.bar(edattain_counts.index, edattain_counts.values, color='purple')

plt.xlabel('Educational Attainment')
plt.ylabel('Frequency')
plt.title('Frequency by Educational Attainment')
plt.xticks(ticks=edattain_counts.index, labels=[edattain_labels[label] for label in edattain_counts.index], rotation=45, ha='right')

plt.show()


# In[20]:


# Univariate Analysis for Class of Work
eclasswk_labels = {
    0:'N/A',
    1:'Employees',
    2:'Employers',
    3:'Own-account worker',
    4:'Contributing family workers',
    5:'Persons not classifiable by status'
}
eclasswk_counts = filtered_df['ECLASSWK'].value_counts().sort_index()
plt.figure(figsize=(15, 6)) 
plt.bar(eclasswk_counts.index, eclasswk_counts.values, color='purple')

plt.xlabel('Class of Work')
plt.ylabel('Frequency')
plt.title('Frequency by Class of Work')
plt.xticks(ticks=eclasswk_counts.index, labels=[eclasswk_labels[label] for label in eclasswk_counts.index], rotation=45, ha='right')

plt.show()


# In[21]:


# Univariate Analysis for Heat
heat_labels = {
    1:'No heating available',
    2:'Heating available'
}
heat_counts = filtered_df['HEAT'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([heat_labels[code] for code in heat_counts.index], heat_counts.values, color='purple')

plt.xlabel('Heat')
plt.ylabel('Frequency')
plt.title('Frequency of Heat Availability')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[22]:


# Univariate Analysis for Household type
hhtype_labels = {
    3:'Married/cohab with children',
    4:'Single-parent family',
    6:'Extended family, relatives only',
    7:'Composite huosehold, family, and non-relatives',
    8:'Non-family household',
    9:'Unclassified subfamily',
    10:'Other relative or non-relative household',
    11:'Group quarters'
}
hhtype_counts = filtered_df['HHTYPE'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([hhtype_labels[code] for code in hhtype_counts.index], hhtype_counts.values, color='purple')

plt.xlabel('Household Type')
plt.ylabel('Frequency')
plt.title('Frequency of Household Type')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[23]:


# Univariate Analysis of Person Number
pernum_counts = filtered_df['PERNUM'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
bars = plt.bar(pernum_counts.index, pernum_counts.values, color='purple')

plt.xlabel('Person Number')
plt.ylabel('Frequency')
plt.title('Frequency of Person Number')

plt.xticks(range(1, len(pernum_counts) + 1))
plt.xticks(rotation=45, ha='right')

for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), 
             ha='center', va='bottom', color='black', fontsize=10)

plt.show()


# In[24]:


# Univariate Analysis for Location of Mother
momloc_labels = {
    0:'Mother Absent',
    1:'Mother Present'
}
momloc_counts = filtered_df['MOMLOC'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([momloc_labels[code] for code in momloc_counts.index], momloc_counts.values, color='purple')

plt.xlabel('Location of Mother')
plt.ylabel('Frequency')
plt.title('Frequency of Location of Mother')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[25]:


# Univariate Analysis for Location of Father
poploc_labels = {
    0:'Father Absent',
    1:'Father Present'
}
poploc_counts = filtered_df['POPLOC'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([poploc_labels[code] for code in poploc_counts.index], poploc_counts.values, color='purple')

plt.xlabel('Location of Father')
plt.ylabel('Frequency')
plt.title('Frequency of Location of Father')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[26]:


# Univariate Analysis for Location of Spouse
sploc_labels = {
    0:'Spouse Absent',
    1:'Spouse Present'
}
sploc_counts = filtered_df['SPLOC'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([sploc_labels[code] for code in sploc_counts.index], sploc_counts.values, color='purple')

plt.xlabel('Location of Spouse')
plt.ylabel('Frequency')
plt.title('Frequency of Location of Spouse')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[27]:


# Univariate Analysis for Relationship to Household Head
relate_labels = {
    1:'Head',
    2:'Spouse/partner',
    3:'Child',
    4:'Other relative',
    5:'Non-relative',
    6:'Other relative/non-relative'
}
relate_counts = filtered_df['RELATE'].value_counts().sort_index()
plt.figure(figsize=(15, 8)) 
plt.bar([relate_labels[code] for code in relate_counts.index], relate_counts.values, color='purple')
plt.xlabel('Relationship to Household Head')
plt.ylabel('Frequency')
plt.title('Frequency of Household Head Relationships')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[28]:


# Univariate Analysis for Sex
sex_labels = {
    1:'Male',
    2:'Female'
}
sex_counts = filtered_df['SEX'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([sex_labels[code] for code in sex_counts.index], sex_counts.values, color='purple')

plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.title('Frequency of Sex')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[29]:


# Univariate Analysis for Marital Status
marst_labels = {
    0:'N/A',
    1:'Single/never married',
    2:'Married/in union',
    3:'Separated/divorced/spouse absent',
    4:'Widowed'
}
marst_counts = filtered_df['MARST'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([marst_labels[code] for code in marst_counts.index], marst_counts.values, color='purple')

plt.xlabel('Marital Status')
plt.ylabel('Frequency')
plt.title('Frequency of Marital Status')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[30]:


# Univariate Analysis for Employment Status
eempstat_labels = {
    0:'N/A',
    110:'Employed',
    120:'Unemployed',
    121:'Unemployed, never worked before',
    200:'Not economically active, unspecified',
    210:'Student',
    220:'Pension or capital income recipients',
    230:'Homemakers',
    240:'Others'
}
eempstat_counts = filtered_df['EEMPSTAT'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([eempstat_labels[code] for code in eempstat_counts.index], eempstat_counts.values, color='purple')
plt.xlabel('Employment Status')
plt.ylabel('Frequency')
plt.title('Frequency of Employment Status')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[31]:


# Univariate Analysis for Labour Force Participation
labforce_labels = {
    0:'N/A',
    1:'Not in Labour Force',
    2:'In Labour Force'
}
labforce_counts = filtered_df['LABFORCE'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([labforce_labels[code] for code in labforce_counts.index], labforce_counts.values, color='purple')

plt.xlabel('Labour Force Participation')
plt.ylabel('Frequency')
plt.title('Frequency of Labour Force Participation')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[32]:


# Univariate Analysis for Occupation
occisco_labels = {
    1:'Legislators, senior officials, managers',
    2:'Professionals',
    3:'Technicians, associate professionals',
    4:'Clerks',
    5:'Service workers, shop, market sales',
    6:'Skilled agricultural & fishery workers',
    7:'Crafts, related trades',
    8:'Plant & machine operators & assemblers',
    9:'Elementary occupations',
    10:'Armed forces',
    99:'N/A'
}
occisco_counts = filtered_df['OCCISCO'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([occisco_labels[code] for code in occisco_counts.index], occisco_counts.values, color='purple')

plt.xlabel('Occupation')
plt.ylabel('Frequency')
plt.title('Frequency of Occupation')

plt.xticks(rotation=45, ha='right')

plt.show()


# # Pearson's Correlation Coefficient Matrix

# In[33]:


# Pearson's Correlation Coefficient in Matrix form
df_corr = filtered_df.corr().transpose()
mask = np.triu(np.ones_like(df_corr))
f, ax = plt.subplots(figsize = (30, 20))
sns.heatmap(df_corr,mask=mask,fmt=".2f",annot=True)
plt.show()


# # Bivariate Analysis

# In[34]:


# NO Water Supply x Country 
# country_labels = {
#     40: 'Austria',
#     300: 'Greece',
#     348: 'Hungary',
#     616: 'Poland',
#     620: 'Portugal',
#     642: 'Romania',
#     724: 'Spain'
# }

filtered_country = filtered_df.loc[filtered_df['WATSUP'] == 1, 'COUNTRY']

country_counts = filtered_country.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'COUNTRY': country_counts.index, 'Count': country_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='COUNTRY', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Count of Households without Water Supply (by Country)')
# plt.xticks(ticks=range(len(country_labels)), labels=[country_labels[label] for label in country_counts.index], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[35]:


# NO Heat x Country
# country_labels = {
#     300: 'Greece',
#     348: 'Hungary',
#     620: 'Portugal',
#     724: 'Spain'
# }

filtered_country = filtered_df.loc[filtered_df['HEAT'] == 1, 'COUNTRY']

country_counts = filtered_country.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'COUNTRY': country_counts.index, 'Count': country_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='COUNTRY', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Count of Households without Heat (by Country)')
# plt.xticks(ticks=range(len(country_labels)), labels=[country_labels[label] for label in country_counts.index], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[36]:


# NO Heat x Class of Work
eclasswk_labels = {
    0:'N/A',
    1:'Employees',
    2:'Employers',
    3:'Own-account worker',
    4:'Contributing family workers',
    5:'Persons not classifiable by status'
}

filtered_eclasswk = filtered_df.loc[filtered_df['HEAT'] == 1, 'ECLASSWK']

eclasswk_counts = filtered_eclasswk.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'ECLASSWK': eclasswk_counts.index, 'Count': eclasswk_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='ECLASSWK', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Class of Work')
plt.ylabel('Count')
plt.title('Count of Households without Heat (by Class of Work)')
plt.xticks(ticks=range(len(eclasswk_labels)), labels=[eclasswk_labels[label] for label in eclasswk_counts.index], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[37]:


# NO Water x Class of Work
eclasswk_labels = {
    0:'N/A',
    1:'Employees',
    2:'Employers',
    3:'Own-account worker',
    4:'Contributing family workers',
    5:'Persons not classifiable by status'
}

filtered_eclasswk = filtered_df.loc[filtered_df['WATSUP'] == 1, 'ECLASSWK']

eclasswk_counts = filtered_eclasswk.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'ECLASSWK': eclasswk_counts.index, 'Count': eclasswk_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='ECLASSWK', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Class of Work')
plt.ylabel('Count')
plt.title('Count of Households without Water Supply (by Class of Work)')
plt.xticks(ticks=range(len(eclasswk_labels)), labels=[eclasswk_labels[label] for label in eclasswk_counts.index], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[38]:


# NO Water Supply x Educational Attainment
edattain_labels = {
    2: 'Less than Primary Completed',
    3: 'Primary Completed',
    1: 'Secondary Completed',
    4: 'University Completed'
}

filtered_edattain = filtered_df.loc[filtered_df['WATSUP'] == 1, 'EDATTAIN']

edattain_counts = filtered_edattain.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'EDATTAIN': edattain_counts.index, 'Count': edattain_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='EDATTAIN', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Educational Attainment')
plt.ylabel('Count')
plt.title('Count of Households without Water Supply (by Educational Attainment)')
plt.xticks(ticks=range(len(edattain_labels)), labels=[edattain_labels[label] for label in edattain_counts.index], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[39]:


# NO Heat x Educational Attainment
edattain_labels = {
    2:'Primary not Completed',
    1:'Primary Completed',
    3:'Secondary Completed',
    4:'University Completed'
}

filtered_edattain = filtered_df.loc[filtered_df['HEAT'] == 1, 'EDATTAIN']

edattain_counts = filtered_edattain.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'EDATTAIN': edattain_counts.index, 'Count': edattain_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='EDATTAIN', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Educational Attainment')
plt.ylabel('Count')
plt.title('Count of Households without Heat (by Educational Attainment)')
plt.xticks(ticks=range(len(edattain_labels)), labels=[edattain_labels[label] for label in edattain_counts.index], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[40]:


# NO Water Supply and NO Heat x Educational Attainment
filtered_data = filtered_df[(filtered_df['WATSUP'] == 1) & (filtered_df['HEAT'] == 1)]

# Filtering Educational Attainment column
edattain_counts = filtered_data['EDATTAIN'].value_counts()

# Sorting
edattain_counts_sorted = edattain_counts.sort_index()

x_labels = ['Less than Primary Completed', 'Primary Completed', 'Secondary Completed', 'University Completed']

# Bar Chart
plt.figure(figsize=(20, 6))
sns.barplot(x=x_labels, y=edattain_counts_sorted.values, palette='YlGnBu')
plt.xlabel('Educational Attainment')
plt.ylabel('Count')
plt.title('Distribution of Educational Attainment for No Water and Heat')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[41]:


# NO Water Supply x Occupation
# occisco_labels = {
#     1:'Legislators, senior officials, managers',
#     2:'Professionals',
#     3:'Technicians, associate professionals',
#     4:'Clerks',
#     5:'Service workers, shop, market sales',
#     6:'Skilled agricultural & fishery workers',
#     7:'Crafts, related trades',
#     8:'Plant & machine operators & assemblers',
#     9:'Elementary occupations',
#     10:'Armed forces',
#     99:'N/A'
# }

filtered_occisco = filtered_df.loc[filtered_df['WATSUP'] == 1, 'OCCISCO']

occisco_counts = filtered_occisco.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'OCCISCO': occisco_counts.index, 'Count': occisco_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='OCCISCO', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.title('Count of Households without Water Supply (by Occupation)')
# plt.xticks(ticks=range(len(occisco_labels)), labels=[occisco_labels[label] for label in occisco_counts.index], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[42]:


# NO Heat x Occupation
# occisco_labels = {
#     1:'Legislators, senior officials, managers',
#     2:'Professionals',
#     3:'Technicians, associate professionals',
#     4:'Clerks',
#     5:'Service workers, shop, market sales',
#     6:'Skilled agricultural & fishery workers',
#     7:'Crafts, related trades',
#     8:'Plant & machine operators & assemblers',
#     9:'Elementary occupations',
#     10:'Armed forces',
#     99:'N/A'
# }

filtered_occisco = filtered_df.loc[filtered_df['HEAT'] == 1, 'OCCISCO']

occisco_counts = filtered_occisco.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'OCCISCO': occisco_counts.index, 'Count': occisco_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='OCCISCO', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.title('Count of Households without Heat (by Occupation)')
# plt.xticks(ticks=range(len(occisco_labels)), labels=[occisco_labels[label] for label in occisco_counts.index], rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[43]:


# NO Heat x Family Size

filtered_famsize = filtered_df.loc[filtered_df['HEAT'] == 1, 'FAMSIZE']

famsize_counts = filtered_famsize.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'FAMSIZE': famsize_counts.index, 'Count': famsize_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='FAMSIZE', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.title('Count of Households without Heat (by Family Size)')
plt.tight_layout()
plt.show()


# In[44]:


# NO Heat x Number of Children

filtered_nchild = filtered_df.loc[filtered_df['HEAT'] == 1, 'NCHILD']

nchild_counts = filtered_nchild.value_counts()

# Contingency table
contingency_table = pd.DataFrame({'NCHILD': nchild_counts.index, 'Count': nchild_counts.values})

# Bar chart
plt.figure(figsize=(15, 6))
sns.barplot(x='NCHILD', y='Count', data=contingency_table, palette='YlGnBu')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.title('Count of Households without Heat (by Number of Children)')
plt.tight_layout()
plt.show()


# In[45]:


# Homemakers x Family Size
filtered_data = filtered_df[filtered_df['EEMPSTAT'] == 230]

# Filtering Family Size column
famsize_counts = filtered_data['FAMSIZE'].value_counts()

# Sorting
famsize_counts_sorted = famsize_counts.sort_index()

# Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x=famsize_counts_sorted.index, y=famsize_counts_sorted.values, palette='YlGnBu')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.title('Distribution of Family Sizes for Homemakers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Bivariate Analysis (Contingency Tables)

# In[46]:


# Heat x Wsater Supply
contingency_table = pd.crosstab(index=filtered_df['WATSUP'], columns=filtered_df['HEAT'])

percentage_table = contingency_table.apply(lambda r: r/r.sum(), axis=1) * 100

watsup_labels = ['N/A', 'No Piped Water','Piped Water']
heat_labels = ['No Heating Available', 'Heating Available']

plt.figure(figsize=(8, 6))
sns.heatmap(percentage_table, annot=True, fmt='.1f', cmap='YlGnBu',xticklabels=heat_labels, yticklabels=watsup_labels)
plt.xlabel('Heat')
plt.ylabel('Water Supply')
plt.title('Contingency Table Heatmap: Water Supply vs. Heat (Percentages)')
plt.tight_layout()
plt.show()


# In[47]:


# Educational Attainment x Water Supply
edattain_labels = {
    1: 'Primary not Completed',
    2: 'Primary Completed',
    3: 'Secondary Completed',
    4: 'University Completed'
}
watsup_labels = {
    0:'N/A',
    1:'No Piped Water',
    2:'Piped Water'
}
contingency_table = pd.crosstab(index=filtered_df['EDATTAIN'], columns=filtered_df['WATSUP'])

percentage_table = contingency_table.apply(lambda r: r/r.sum(), axis=1) * 100

plt.figure(figsize=(18, 12))
sns.heatmap(percentage_table, annot=True, fmt='.1f', cmap='YlGnBu')
plt.xlabel('Water SUpply')
plt.ylabel('Educational Attainment')
plt.xticks(ticks=range(len(watsup_labels)), labels=[watsup_labels[label] for label in contingency_table.columns], rotation=45)
plt.yticks(ticks=range(len(edattain_labels)), labels=[edattain_labels[label] for label in contingency_table.index])
plt.title('Contingency Table Heatmap: Educational Attainment vs. Water Supply (Percentages)')
plt.tight_layout()
plt.show()


# In[48]:


# Educational Attainment x Sex
edattain_labels = {
    1: 'Primary not Completed',
    2: 'Primary Completed',
    3: 'Secondary Completed',
    4: 'University Completed'
}
sex_labels = {
    1: 'Male',
    2: 'Female'
}
contingency_table = pd.crosstab(index=filtered_df['EDATTAIN'], columns=filtered_df['SEX'])

percentage_table = contingency_table.apply(lambda r: r/r.sum(), axis=1) * 100

plt.figure(figsize=(18, 12))
sns.heatmap(percentage_table, annot=True, fmt='.1f', cmap='YlGnBu')
plt.xlabel('Sex')
plt.ylabel('Educational Attainment')
plt.xticks(ticks=range(len(sex_labels)), labels=[sex_labels[label] for label in contingency_table.columns], rotation=45)
plt.yticks(ticks=range(len(edattain_labels)), labels=[edattain_labels[label] for label in contingency_table.index])
plt.title('Contingency Table Heatmap: Educational Attainment vs. Sex (Percentages)')
plt.tight_layout()
plt.show()


# In[49]:


# Labour Force x Employment Status
contingency_table = pd.crosstab(filtered_df['LABFORCE'], filtered_df['EEMPSTAT'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:, :]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

#Labeling x and y axis ticks
x_labels=['Employed','Unemployed','Unemployed, never worked before','Not economically active, unspecified','Student','Pension or capital income recipients','Homemakers','Others']
y_labels = ['Not in labour force','In labour force']

plt.figure(figsize=(32, 6))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=x_labels, yticklabels=y_labels)

# Labeling x and y axes
plt.xlabel('Employment Status')
plt.ylabel('Labour Force Participation')
plt.title("Employment Status Versus Labour Force Participation")

plt.tight_layout()
plt.show()


# In[50]:


# Employment Status x Sex
contingency_table = pd.crosstab(filtered_df['SEX'], filtered_df['EEMPSTAT'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:, :]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

#Labeling x and y axis ticks
x_labels=['Employed','Unemployed','Unemployed, never worked before','Not economically active, unspecified','Student','Pension or capital income recipients','Homemakers','Others']
y_labels = ['Male','Female']

plt.figure(figsize=(32, 6))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=x_labels, yticklabels=y_labels)

# Labeling x and y axes
plt.xlabel('Employment Status')
plt.ylabel('Sex')
plt.title("Employment Status Versus Sex")

plt.tight_layout()
plt.show()


# In[51]:


# Spouse Location x Number of Children
contingency_table = pd.crosstab(filtered_df['NCHILD'], filtered_df['SPLOC'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:, :]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

# Labeling x axis ticks
x_labels=['Spouse Absent','Spouse Present']

plt.figure(figsize=(12, 6))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu',xticklabels=x_labels)

# Labeling x and y axes
plt.xlabel('Presence of Spouse')
plt.ylabel('Number of Children')
plt.title("Presence of Spouse Versus Number of Children")

plt.tight_layout()
plt.show()


# In[52]:


# Family Size x Number of Children Under 5
contingency_table = pd.crosstab(filtered_df['NCHLT5'], filtered_df['FAMSIZE'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:4, :12]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

plt.figure(figsize=(20, 8))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu')

# Labeling x and y axes
plt.xlabel('Family Size')
plt.ylabel('Number of Children Under 5')
plt.title("Family Size Versus Number of Children Under 5")

plt.tight_layout()
plt.show()


# In[53]:


# Educational Attainment x Marital Status
contingency_table = pd.crosstab(filtered_df['MARST'], filtered_df['EDATTAIN'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:, :]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

# Labeling x and y axis ticks
x_labels=['Primary not Completed','Primary Completed','Secondary Completed','University Completed']
y_labels=['Single/never married','Married/in union','Separated/divorced/spouse absent','Widowed']

plt.figure(figsize=(20, 8))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu',xticklabels=x_labels,yticklabels=y_labels)

# Labeling x and y axes
plt.xlabel('Educational Attainment')
plt.ylabel('Marital Status')
plt.title("Educational Attainment Versus Marital Status")

plt.tight_layout()
plt.show()


# In[54]:


# Educational Attainment x Country
contingency_table = pd.crosstab(filtered_df['EDATTAIN'], filtered_df['COUNTRY'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:, :]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

# Labeling x and y axis ticks
x_labels=['Austria','Greece','Hungary','Poland','Portugal','Romania','Spain']
y_labels=['Primary not Completed','Primary Completed','Secondary Completed','University Completed']


plt.figure(figsize=(20, 8))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu',xticklabels=x_labels,yticklabels=y_labels)

# Labeling x and y axes
plt.xlabel('Country')
plt.ylabel('Educational Attainment')
plt.title("Educational Attainment by Country")

plt.tight_layout()
plt.show()


# In[55]:


# Marital Status x Country
contingency_table = pd.crosstab(filtered_df['MARST'], filtered_df['COUNTRY'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:, :]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

# Labeling x and y axis ticks
x_labels=['Austria','Greece','Hungary','Poland','Portugal','Romania','Spain']
y_labels=['Single/never married','Married/in union','Separated/divorced/spouse absent','Widowed']


plt.figure(figsize=(20, 8))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu',xticklabels=x_labels,yticklabels=y_labels)

# Labeling x and y axes
plt.xlabel('Country')
plt.ylabel('Marital Status')
plt.title("Marital Status by Country")

plt.tight_layout()
plt.show()


# # Feature Selection

# In[56]:


# Sorting absolute correlations
# Compute the absolute correlations for each feature
abs_corr_sum = df_corr.abs().sum()

# Sort the features in descending order of total absolute correlations
sorted_features = abs_corr_sum.sort_values(ascending=False)

# Create a list of most to least important features
most_to_least_important = sorted_features.index.tolist()

# Print the list of most to least important features
print(most_to_least_important)


# In[57]:


filtered_df2 = filtered_df.copy()
filtered_df3 = filtered_df.copy()
filtered_df4 = filtered_df.copy()
filtered_df5 = filtered_df.copy()
filtered_df6 = filtered_df.copy()
filtered_df7 = filtered_df.copy()
filtered_df8 = filtered_df.copy()
filtered_df9 = filtered_df.copy()


# In[58]:


# # Removing redundant features, taking out the lower absolute correlation counterparts
todrop = ['SAMPLE','SERIAL','YEAR','PERSONS','MARSTD','RELATED','EDATTAIND','OCCISCO','ECLASSWK']
filtered_df1 = filtered_df.drop(columns=todrop)

# # Print information about the updated DataFrame
filtered_df1.info()


# In[59]:


filtered_df2 = filtered_df2.drop(columns=todrop)
filtered_df3 = filtered_df3.drop(columns=todrop)
filtered_df4 = filtered_df4.drop(columns=todrop)


# In[60]:


#Scaling and mean-centering data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(filtered_df1)


# In[61]:


# Computing covariance matrix
covariance_matrix = np.cov(data_scaled, rowvar=False)

print(covariance_matrix)


# In[62]:


filtered_df1.info()


# In[63]:


# covariance matrix
variable_labels = ['Country','Water Supply','Heat','N. Rooms','Household Type','N. Families','N. Couples','N. Mothers','N. Fathers','Person Number','Loc. Mother','Loc. Father','Loc. Spouse','Family Size','N. Children','N. Children Under 5','Eldest Child Age','Youngest Child Age','Relationship to Head','Age','Sex','Marital Status','Educational Attainment','Employment Status','Labour Force Participation','Industry']
# cov_df_transposed = covariance_matrix.transpose()
# Create a mask for the upper triangular part of the matrix
mask = np.triu(np.ones_like(covariance_matrix, dtype=bool))

# Create a heatmap using Seaborn
plt.figure(figsize=(30, 20))
sns.heatmap(covariance_matrix, annot=True,fmt='.3f',cmap='coolwarm', center=0, mask=mask,xticklabels=variable_labels,yticklabels=variable_labels)

# Add labels and title
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.title('Covariance Matrix Heatmap')

plt.tight_layout()
plt.show()


# In[64]:


# Calculating the total absolute covariances for each feature
total_absolute_covariances = np.sum(np.abs(covariance_matrix), axis=1)

feature_covariance_pairs = list(zip(variable_labels, total_absolute_covariances))
sorted_features = sorted(feature_covariance_pairs, key=lambda x: x[1], reverse=True)

for rank, (feature, covariance) in enumerate(sorted_features, start=1):
    print(f"Rank {rank}: Feature '{feature}' - Total Absolute Covariance: {covariance:.3f}")


# In[65]:


#calculating eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)


# In[66]:


# Pairing eigenvalues with their corresponding eigenvectors
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

# Sorting the eigenvalue-eigenvector pairs in descending order based on eigenvalues
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Extracting sorted eigenvectors
sorted_eigenvectors = [pair[1] for pair in eig_pairs]


# In[67]:


#Generateing eigenvalue spectrum
# Calculating proportion of variance explained by each component
explained_variance_ratio = eigenvalues / sum(eigenvalues)

# Plotting the eigenvalue spectrum
plt.figure(figsize=(14, 8))
bars = plt.bar(range(1, len(eigenvalues) + 1), explained_variance_ratio, alpha=0.7, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('Eigenvalue Spectrum')
plt.xticks(range(1, len(eigenvalues) + 1))

# Annotating each bar with the variance explained
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{explained_variance_ratio[i]:.2f}', ha='center', va='bottom')

plt.show()


# # Narrowing down features
# We will now remove the bottom five features and recalculate the PCA

# In[68]:


filtered_df2 = filtered_df2[['YNGCH','ELDCH','AGE','NCOUPLES','FAMSIZE','LABFORCE','NFATHERS','EEMPSTAT','SPLOC',
                            'HHTYPE','MARST','NMOTHERS','PERNUM','RELATE','INDGEN','EDATTAIN','NCHLT5','MOMLOC',
                            'POPLOC','SEX','NCHILD']]


# In[ ]:





# In[69]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(filtered_df2)


# In[70]:


# Computing covariance matrix
covariance_matrix = np.cov(data_scaled, rowvar=False)

print(covariance_matrix)


# In[71]:


# covariance matrix
mask = np.triu(np.ones_like(covariance_matrix, dtype=bool))

# Create a heatmap using Seaborn
plt.figure(figsize=(30, 20))
sns.heatmap(covariance_matrix, annot=True,fmt='.3f',cmap='coolwarm', center=0, mask=mask)

# Add labels and title
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.title('Covariance Matrix Heatmap')

plt.tight_layout()
plt.show()


# In[72]:


# Calculating the total absolute covariances for each feature
total_absolute_covariances = np.sum(np.abs(covariance_matrix), axis=1)

feature_covariance_pairs = list(zip(variable_labels, total_absolute_covariances))
sorted_features = sorted(feature_covariance_pairs, key=lambda x: x[1], reverse=True)

for rank, (feature, covariance) in enumerate(sorted_features, start=1):
    print(f"Rank {rank}: Feature '{feature}' - Total Absolute Covariance: {covariance:.3f}")


# In[73]:


#calculating eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)


# In[74]:


# Pairing eigenvalues with their corresponding eigenvectors
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

# Sorting the eigenvalue-eigenvector pairs in descending order based on eigenvalues
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Extracting sorted eigenvectors
sorted_eigenvectors = [pair[1] for pair in eig_pairs]


# In[75]:


#Generateing eigenvalue spectrum
# Calculating proportion of variance explained by each component
explained_variance_ratio = eigenvalues / sum(eigenvalues)

# Plotting the eigenvalue spectrum
plt.figure(figsize=(14, 8))
bars = plt.bar(range(1, len(eigenvalues) + 1), explained_variance_ratio, alpha=0.7, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('Eigenvalue Spectrum')
plt.xticks(range(1, len(eigenvalues) + 1))

# Annotating each bar with the variance explained
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{explained_variance_ratio[i]:.2f}', ha='center', va='bottom')

plt.show()


# In[76]:


from sklearn.decomposition import PCA

# #Exctracting and Standardising features
X = filtered_df2
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performing PCA
n_components = X.shape[1]
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Variance explained
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative variance explained
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plotting eigenvalue spectrum
plt.figure(figsize=(10, 6))
plt.bar(range(1,(n_components) + 1), explained_variance_ratio, align='center', label='Explained Variance')
plt.step(range(1,(n_components) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance by Principal Components')
plt.legend()
plt.show()


# In[77]:


# Selecting number of components based on results
selected_components = 10

# Perform PCA with the selected number of components
pca_final = PCA(n_components=selected_components)
X_final = pca_final.fit_transform(X)

# Variance explained
explained_variance_ratio = pca_final.explained_variance_ratio_

# Cumulative variance explained
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plotting eigenvalue spectrum
plt.figure(figsize=(10, 6))
plt.bar(range(1,(selected_components) + 1), explained_variance_ratio, align='center', label='Explained Variance')
plt.step(range(1,(selected_components) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance by Principal Components')
plt.legend()
plt.show()


# # Polychoric PCA

# In[78]:


import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo


ordinal_columns = ['COUNTRY', 'WATSUP', 'HEAT', 'HHTYPE', 'MOMLOC', 'POPLOC', 'SPLOC', 'RELATE',
                   'SEX', 'MARST', 'EDATTAIN', 'EEMPSTAT', 'LABFORCE', 'INDGEN']

ordinal_data = filtered_df5[ordinal_columns]

# Calculating Bartlett's Sphericity test
chi_square_value, p_value = calculate_bartlett_sphericity(ordinal_data)
print(f"Bartlett's test statistic: {chi_square_value:.4f}")
print(f"P-value: {p_value:.4f}")

# Calculating Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy
kmo_all, kmo_model = calculate_kmo(ordinal_data)
print(f"KMO for each variable:\n{kmo_all}")
print(f"KMO overall: {kmo_model}")


# In[79]:


# Performing Polychoric PCA
n_factors = len(ordinal_columns)
fa = FactorAnalyzer(n_factors, rotation=None)
fa.fit(ordinal_data)

# Obtaining Factor Loadings
print("Factor Loadings:")
print(fa.loadings_)


# In[80]:


# Displaying Factor loadings via DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])]

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[81]:


#Explained variance for each factor
explained_variance = fa.get_eigenvalues()[0] / np.sum(fa.get_eigenvalues()[0])

# Scree plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Explained Variance')
plt.xticks(np.arange(1, len(explained_variance) + 1))
plt.grid(True)
plt.show()


# # Only keeping the first 9 factors

# In[82]:


# Reducing the number of factors
n_factors = 9
fa = FactorAnalyzer(n_factors, rotation=None)
fa.fit(ordinal_data)

# Obtaining Factor Loadings
print("Factor Loadings:")
print(fa.loadings_)


# In[83]:


# Displaying factor loadings via DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])]

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[84]:


# Calculating factor scores using regression-based method
factor_scores = np.dot(ordinal_data, fa.loadings_)

# Calculating standard deviations of factor scores
factor_scores_std = factor_scores.std(axis=0)

# Checking for zero standard deviations
non_zero_std_indices = factor_scores_std > 0

# Normalizing factor scores
normalized_factor_scores = np.zeros_like(factor_scores)
normalized_factor_scores[:, non_zero_std_indices] = (factor_scores[:, non_zero_std_indices] - factor_scores[:, non_zero_std_indices].mean(axis=0)) / factor_scores_std[non_zero_std_indices]

print("Factor Scores:")
print(normalized_factor_scores)


# In[85]:


# Calculating absolute values of factor loadings (Formula 6)
abs_factor_loadings = np.abs(fa.loadings_)

# Calculating the sum of absolute values for each factor
sum_abs_loadings = abs_factor_loadings.sum(axis=0)

# Calculating the weights proportional to absolute values of factor loadings (Formula 7)
factor_weights = sum_abs_loadings / sum_abs_loadings.sum()

print("Factor Weights:")
print(factor_weights)


# In[86]:


# Calculating the composite poverty indicator for each observation (Formula 7)
composite_indicator = np.dot(normalized_factor_scores, factor_weights)

min_value = composite_indicator.min()
max_value = composite_indicator.max()
normalized_indicator = (composite_indicator - min_value) / (max_value - min_value)

filtered_df5['Composite_Indicator'] = normalized_indicator


# In[87]:


# Histogram of composite scores
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_df5, x='Composite_Indicator', bins=30, kde=True)
plt.xlabel('Composite Indicator')
plt.ylabel('Frequency')
plt.title('Histogram of Composite Indicator Scores')
plt.show()


# # Polychoric PCA with Discrete, Ordinal, and Categorical Data

# In[88]:


ordinal_columns = ['COUNTRY', 'WATSUP', 'HEAT','ROOMS', 'HHTYPE','NFAMS','NCOUPLES','NMOTHERS','NFATHERS', 'PERNUM',
                   'MOMLOC', 'POPLOC', 'SPLOC', 'FAMSIZE','NCHILD','NCHLT5','ELDCH','YNGCH','RELATE','AGE',
                   'SEX', 'MARST', 'EDATTAIN', 'EEMPSTAT', 'LABFORCE', 'INDGEN']

ordinal_data = filtered_df3[ordinal_columns]

#Bartlett's Sphericity test
chi_square_value, p_value = calculate_bartlett_sphericity(ordinal_data)
print(f"Bartlett's test statistic: {chi_square_value:.4f}")
print(f"P-value: {p_value:.4f}")

#KMO measure of sampling adequacy
kmo_all, kmo_model = calculate_kmo(ordinal_data)
print(f"KMO overall: {kmo_all}")
print(f"KMO for each variable:\n{kmo_model}")


# In[89]:


# Performing polychoric PCA
n_factors = len(ordinal_columns)  # Number of factors
fa = FactorAnalyzer(n_factors, rotation=None)  # You can specify a rotation method if needed
fa.fit(ordinal_data)

print("Factor Loadings:")
print(fa.loadings_)


# In[90]:


# Factor Loadings DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])]

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[91]:


#Explained variance for each factor
explained_variance = fa.get_eigenvalues()[0] / np.sum(fa.get_eigenvalues()[0])

# Scree plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Explained Variance')
plt.xticks(np.arange(1, len(explained_variance) + 1))
plt.grid(True)
plt.show()


# # Re-adjusting Number of Factors

# In[92]:


# Reducing to 11 factors
n_factors = 11
fa = FactorAnalyzer(n_factors, rotation=None)
fa.fit(ordinal_data)

print("Factor Loadings:")
print(fa.loadings_)


# In[93]:


# Factor loadings DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])]

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[94]:


# Reducing to 7 factors
n_factors = 7
fa = FactorAnalyzer(n_factors, rotation=None)
fa.fit(ordinal_data)

print("Factor Loadings:")
print(fa.loadings_)


# In[95]:


# Factor loadings DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])]

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[96]:


# Calculating factor scores using regression-based method
factor_scores = np.dot(ordinal_data, fa.loadings_)

# Calculating standard deviations of factor scores
factor_scores_std = factor_scores.std(axis=0)

non_zero_std_indices = factor_scores_std > 0

# Normalize factor scores
normalized_factor_scores = np.zeros_like(factor_scores)
normalized_factor_scores[:, non_zero_std_indices] = (factor_scores[:, non_zero_std_indices] - factor_scores[:, non_zero_std_indices].mean(axis=0)) / factor_scores_std[non_zero_std_indices]

# Factor Scores
print("Factor Scores:")
print(normalized_factor_scores)


# In[97]:


# Calculating absolute values of factor loadings (Formula 6)
abs_factor_loadings = np.abs(fa.loadings_)

# Calculating the sum of absolute values for each factor
sum_abs_loadings = abs_factor_loadings.sum(axis=0)

# Calculating weights proportional to absolute values of factor loadings (Formula 7)
factor_weights = sum_abs_loadings / sum_abs_loadings.sum()

print("Factor Weights:")
print(factor_weights)


# In[98]:


# Calculating the composite poverty indicator for each observation (Formula 7)
composite_indicator = np.dot(normalized_factor_scores, factor_weights)

min_value = composite_indicator.min()
max_value = composite_indicator.max()
normalized_indicator = (composite_indicator - min_value) / (max_value - min_value)

filtered_df3['Composite_Indicator'] = normalized_indicator


# In[99]:


# Histogram of composite scores
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_df3, x='Composite_Indicator', bins=30, kde=True)
plt.xlabel('Composite Indicator')
plt.ylabel('Frequency')
plt.title('Histogram of Composite Indicator Scores')
plt.show()


# In[100]:


# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df3, y='Composite_Indicator')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator Scores')
plt.show()


# In[101]:


# Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(x=filtered_df3['Composite_Indicator'], color='blue', inner='quartiles')
plt.xlabel('Composite Indicator')
plt.ylabel('Density')
plt.title('Distribution of Composite Indicator Scores')
plt.grid(True)
plt.show()


# In[102]:


# Histogram
plt.figure(figsize=(12, 6))
sns.histplot(data=filtered_df3, x='Composite_Indicator', kde=True)
plt.xlabel('Composite Indicator')
plt.ylabel('Frequency')
plt.title('Histogram of Composite Indicator Scores')
plt.grid(True)
plt.show()


# In[103]:


# Boxplot of Indicator x Water Supply
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df3, x='WATSUP', y='Composite_Indicator')
plt.xlabel('Water Supply')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Water Supply')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[104]:


# Boxplot of Indicator x Educational Attainment
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df3, x='EDATTAIN', y='Composite_Indicator')
plt.xlabel('Educatational Attainment')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Educatational Attainment')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[105]:


# Boxplot of Indicator x Heat
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df3, x='HEAT', y='Composite_Indicator')
plt.xlabel('Heat Availability')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Heat Availability')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[106]:


# Boxplot of Indicator x Country
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df3, x='COUNTRY', y='Composite_Indicator')
plt.xlabel('Country')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Country')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[107]:


# Boxplot of Indicator x Employment Status
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df3, x='EEMPSTAT', y='Composite_Indicator')
plt.xlabel('Employment Status')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Employment Status')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[108]:


# Boxplot of Indicator x Labour Force Participation
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df3, x='LABFORCE', y='Composite_Indicator')
plt.xlabel('Labour Force Participation')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Labour Force Participation')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[109]:


# Boxplot of Indicator x Number of Children
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df3, x='NCHILD', y='Composite_Indicator')
plt.xlabel('Number of Children')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Number of Children')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[110]:


# Boxplot of Indicator x Family Size
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df3, x='FAMSIZE', y='Composite_Indicator')
plt.xlabel('Family Size')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Family Size')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# # Polychoric PCA with Variable Reduction

# In[111]:


ordinal_columns = ['COUNTRY', 'WATSUP', 'HEAT','ROOMS', 'HHTYPE','NFAMS','NCOUPLES','NMOTHERS','NFATHERS', 'PERNUM',
                   'MOMLOC', 'POPLOC', 'SPLOC', 'FAMSIZE','NCHILD','NCHLT5','ELDCH','YNGCH','RELATE','AGE',
                   'SEX', 'MARST', 'EDATTAIN', 'EEMPSTAT', 'LABFORCE', 'INDGEN']

ordinal_data = filtered_df4[ordinal_columns]

# Bartlett's Sphericity test
chi_square_value, p_value = calculate_bartlett_sphericity(ordinal_data)
print(f"Bartlett's test statistic: {chi_square_value:.4f}")
print(f"P-value: {p_value:.4f}")

# KMO Measure
kmo_all, kmo_model = calculate_kmo(ordinal_data)
print(f"KMO overall: {kmo_all}")
print(f"KMO for each variable:\n{kmo_model}")


# In[112]:


# Removing any variables with KMO scores below 0.5
ordinal_columns = ['COUNTRY', 'WATSUP', 'ROOMS', 'HHTYPE','NCOUPLES','NMOTHERS','NFATHERS', 'PERNUM',
                   'MOMLOC', 'POPLOC', 'SPLOC', 'FAMSIZE','NCHLT5','ELDCH','YNGCH','RELATE','AGE',
                   'SEX', 'MARST', 'EDATTAIN', 'EEMPSTAT', 'LABFORCE', 'INDGEN']

ordinal_data = filtered_df4[ordinal_columns]

#Bartlett's Sphericity test
chi_square_value, p_value = calculate_bartlett_sphericity(ordinal_data)
print(f"Bartlett's test statistic: {chi_square_value:.4f}")
print(f"P-value: {p_value:.4f}")

# KMO measure
kmo_all, kmo_model = calculate_kmo(ordinal_data)
print(f"KMO overall: {kmo_all}")
print(f"KMO for each variable:\n{kmo_model}")


# In[113]:


# Performing polychoric PCA
n_factors = len(ordinal_columns)
fa = FactorAnalyzer(n_factors, rotation=None)
fa.fit(ordinal_data)

# Obtaining Factor Loadings
print("Factor Loadings:")
print(fa.loadings_)


# In[114]:


# Factor loadings DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])]

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[115]:


# Explained variance
explained_variance = fa.get_eigenvalues()[0] / np.sum(fa.get_eigenvalues()[0])

# Scree plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Explained Variance')
plt.xticks(np.arange(1, len(explained_variance) + 1))
plt.grid(True)
plt.show()


# # Only keeping 8 factors

# In[116]:


# Reducing to 8 Factors
n_factors = 8 
fa = FactorAnalyzer(n_factors, rotation=None)  # You can specify a rotation method if needed
fa.fit(ordinal_data)

# Factor loadings
print("Factor Loadings:")
print(fa.loadings_)


# In[117]:


# Factor loadings DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns 
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])] 

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[118]:


# Calculating factor scores using regression-based method
factor_scores = np.dot(ordinal_data, fa.loadings_)

# Calculating standard deviations of factor scores
factor_scores_std = factor_scores.std(axis=0)

# Check for zero standard deviations
non_zero_std_indices = factor_scores_std > 0

# Normalize factor scores
normalized_factor_scores = np.zeros_like(factor_scores)
normalized_factor_scores[:, non_zero_std_indices] = (factor_scores[:, non_zero_std_indices] - factor_scores[:, non_zero_std_indices].mean(axis=0)) / factor_scores_std[non_zero_std_indices]

print("Factor Scores:")
print(normalized_factor_scores)
np.savetxt("factor scores.csv", normalized_factor_scores, delimiter=",")


# In[119]:


# Calculating absolute values of factor loadings (Formula 6)
abs_factor_loadings = np.abs(fa.loadings_)

# Calculating the sum of absolute values for each factor
sum_abs_loadings = abs_factor_loadings.sum(axis=0)

# Calculating weights proportional to absolute values of factor loadings (Formula 7)
factor_weights = sum_abs_loadings / sum_abs_loadings.sum()

print("Factor Weights:")
print(factor_weights)


# In[120]:


# Calculating the composite poverty indicator for each observation (Formula 7)
composite_indicator = np.dot(normalized_factor_scores, factor_weights)

min_value = composite_indicator.min()
max_value = composite_indicator.max()
normalized_indicator = (composite_indicator - min_value) / (max_value - min_value)

filtered_df4['Composite_Indicator'] = normalized_indicator


# In[121]:


# Histogram of the composite scores
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_df4, x='Composite_Indicator', bins=30, kde=True)
plt.xlabel('Composite Indicator')
plt.ylabel('Frequency')
plt.title('Histogram of Composite Indicator Scores')
plt.show()


# In[122]:


# Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(x=filtered_df4['Composite_Indicator'], color='blue', inner='quartiles')
plt.xlabel('Composite Indicator')
plt.ylabel('Density')
plt.title('Distribution of Composite Indicator Scores')
plt.grid(True)
plt.show()


# In[123]:


# Boxplot of Indicator x Water Supply
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df4, x='WATSUP', y='Composite_Indicator')
plt.xlabel('Water Supply')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Water Supply')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[124]:


# Boxplot of Indicator x Educational Attainment
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df4, x='EDATTAIN', y='Composite_Indicator')
plt.xlabel('Educatational Attainment')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Educatational Attainment')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[125]:


# Boxplot of Indicator x Number of Children
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df4, x='NCHILD', y='Composite_Indicator')
plt.xlabel('Number of Children')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Number of Children')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[126]:


# Boxplot of Indicator x Family Size
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df4, x='FAMSIZE', y='Composite_Indicator')
plt.xlabel('Family Size')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Family Size')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[141]:


# Investigating the Top 10% and Bottom 10% of Indicator Scores
top_10_cutoff = np.percentile(normalized_indicator, 90)
top_5_cutoff = np.percentile(normalized_indicator, 95)
bottom_10_cutoff = np.percentile(normalized_indicator, 10)

# Sectioning off the data
top_10_subset = filtered_df4[filtered_df4['Composite_Indicator'] >= top_10_cutoff]
top_5_subset = filtered_df4[filtered_df4['Composite_Indicator'] >= top_5_cutoff]
bottom_10_subset = filtered_df4[filtered_df4['Composite_Indicator'] <= bottom_10_cutoff]


top_10_subset['Composite_Indicator'].describe()


# In[132]:


bottom_10_subset['Composite_Indicator'].describe()


# In[146]:


# Boxplot of Indicator x Family Size (Top 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=top_10_subset, x='FAMSIZE', y='Composite_Indicator')
plt.xlabel('Family Size')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Family Size (Top 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[148]:


# Boxplot of Indicator x Family Size (Bottom 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=bottom_10_subset, x='FAMSIZE', y='Composite_Indicator')
plt.xlabel('Family Size')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Family Size (Bottom 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[149]:


# Boxplot of Indicator x Number of Children (Top 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=top_10_subset, x='NCHILD', y='Composite_Indicator')
plt.xlabel('Number of Children')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Number of Children (Top 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[150]:


# Boxplot of Indicator x Number of Children (Bottom 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=bottom_10_subset, x='NCHILD', y='Composite_Indicator')
plt.xlabel('Number of Children')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Number of Children (Bottom 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[151]:


# Boxplot of Indicator x Educational Attainment (Top 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=top_10_subset, x='EDATTAIN', y='Composite_Indicator')
plt.xlabel('Educatational Attainment')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Educatational Attainment (Top 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[152]:


# Boxplot of Indicator x Educational Attainment (Bottom 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=bottom_10_subset, x='EDATTAIN', y='Composite_Indicator')
plt.xlabel('Educatational Attainment')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Educatational Attainment (Bottom 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[158]:


# Boxplot of Indicator x Country (Top 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=top_10_subset, x='COUNTRY', y='Composite_Indicator')
plt.xlabel('Country')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Country (Top 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[157]:


# Boxplot of Indicator x Country (Bottom 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=bottom_10_subset, x='COUNTRY', y='Composite_Indicator')
plt.xlabel('Country')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Country (Bottom 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[160]:


# Boxplot of Indicator x Labour Force Participation (Top 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=top_10_subset, x='LABFORCE', y='Composite_Indicator')
plt.xlabel('Labour Force Participation')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Labour Force Participation (Top 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[161]:


# Boxplot of Indicator x Labour Force Participation (Bottom 10%)
plt.figure(figsize=(12, 6))
sns.boxplot(data=bottom_10_subset, x='LABFORCE', y='Composite_Indicator')
plt.xlabel('Labour Force Participation')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Labour Force Participation (Bottom 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[166]:


# Boxplot of Indicator x Age (Top 10%)
plt.figure(figsize=(20, 10))
sns.boxplot(data=top_10_subset, x='AGE', y='Composite_Indicator')
plt.xlabel('Age')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Age (Top 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[167]:


# Boxplot of Indicator x Age (Top 10%)
plt.figure(figsize=(20, 10))
sns.boxplot(data=bottom_10_subset, x='AGE', y='Composite_Indicator')
plt.xlabel('Age')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Age (Bottom 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[181]:


# Boxplot of Indicator x Age of Youngest Child (Top 10%)
plt.figure(figsize=(20, 10))
sns.boxplot(data=top_10_subset, x='YNGCH', y='Composite_Indicator')
plt.xlabel('Age of Youngest Child')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Age of Youngest Child (Top 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[182]:


# Boxplot of Indicator x Age of Youngest Child (Bottom 10%)
plt.figure(figsize=(20, 10))
sns.boxplot(data=bottom_10_subset, x='YNGCH', y='Composite_Indicator')
plt.xlabel('Age of Youngest Child')
plt.ylabel('Composite Indicator')
plt.title('Box Plot of Composite Indicator by Age of Youngest Child (Bottom 10%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[187]:


total_count = len(filtered_df4)
between_08_08_count = len(filtered_df4[(filtered_df4['Composite_Indicator'] >= 0.8) & (filtered_df4['Composite_Indicator'] <= 1.0)])

percentage = (between_08_08_count / total_count) * 100

print(f"Percentage of values between 0.8 and 1.0: {percentage:.2f}%")


# In[ ]:




