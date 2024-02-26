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
    (df['WATSUP'].isin([0,99])),
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
    (df['EDATTAIN'].isin([0,9])),
    (df['EDATTAIND'].isin([999])),
    (df['EEMPSTAT'].isin([0,999])),
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


# Mapping dictionary for category reduction. 0 refers to no access to water, and 1 refers to access to water available.
watsup_mapping = {
    10.0:1,
    11.0:1,
    15.0:1,
    16.0:1,
    17.0:1,
    20.0:0,
}
filtered_df['WATSUP'] = filtered_df['WATSUP'].replace(watsup_mapping)


# In[11]:


# Mapping dictionary for category reduction. 0 refers to no access to heat, 1 refers to heat being accessible in the household.
heat_mapping = {
    1.0:0,
    2.0:1,
    3.0:1,
    4.0:1,
    5.0:1,
}

filtered_df['HEAT'] = filtered_df['HEAT'].replace(heat_mapping)


# In[12]:


# Mapping dictionary for category reduction. 0 refers to primary education and below, while 1 refers to
# secondary education and above.
edattain_mapping = {
    1.0:0,
    2.0:0,
    3.0:1,
    4.0:1,
}

filtered_df['EDATTAIN'] = filtered_df['EDATTAIN'].replace(edattain_mapping)


# In[13]:


# Mapping dictionary for category reduction. 0 refers to unemployment, while 1 refers to employment.
eempstat_mapping = {
    120:0,
    121:0,
    200:0,
    220:0,
    230:0,
    110:1,
    210:1,
    240:1,
}

filtered_df['EEMPSTAT'] = filtered_df['EEMPSTAT'].replace(eempstat_mapping)


# In[14]:


#Recoding 'MOMLOC' to identify the presence/absence of mothers
column_to_recode = 'MOMLOC'

filtered_df[column_to_recode] = filtered_df[column_to_recode].apply(lambda x: 1 if x > 0 else x)


# In[15]:


#Recoding 'POPLOC' to identify the presence/absence of fathers
column_to_recode = 'POPLOC'

filtered_df[column_to_recode] = filtered_df[column_to_recode].apply(lambda x: 1 if x > 0 else x)


# In[16]:


#Recoding 'SPLOC' to identify the presence/absence of spouses
column_to_recode = 'SPLOC'

filtered_df[column_to_recode] = filtered_df[column_to_recode].apply(lambda x: 1 if x > 0 else x)


# # Univariate Analysis of Ordinal Variables

# In[17]:


#removing scientific notation from axes
mpl.rcParams["axes.formatter.limits"] = (-99, 99)

# Creating subplots for univariate analysis
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(20, 22))
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
    ax.set_yscale('linear')
    filtered_df[column].hist(ax=ax, bins=50, color='pink')

    ax.set_title(title_mapping.get(column, column))

plt.show()


# # Univariate Analysis of Category Variables 

# In[20]:


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
plt.yscale('linear')
plt.show()


# In[18]:


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
plt.yscale('linear')
plt.show()


# In[21]:


# Univariate Analysis for Heat
heat_labels = {
    0:'No heating available',
    1:'Heating available'
}
heat_counts = filtered_df['HEAT'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([heat_labels[code] for code in heat_counts.index], heat_counts.values, color='purple')

plt.xlabel('Heat')
plt.ylabel('Frequency')
plt.title('Frequency of Heat Availability')

plt.xticks(rotation=45, ha='right')
plt.yscale('linear')
plt.show()


# In[19]:


# Univariate Analysis for Water Supply
watsup_labels = {
    0:'No Piped Water',
    1:'Piped Water',
}
watsup_counts = filtered_df['WATSUP'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([watsup_labels[code] for code in watsup_counts.index], watsup_counts.values, color='purple')

plt.xlabel('Water Supply')
plt.ylabel('Frequency')
plt.title('Frequency by Water Supply')

plt.xticks(rotation=45, ha='right')
plt.yscale('linear')

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
plt.yscale('linear')
plt.show()


# In[23]:


# Univariate Analysis for Employment Status
eempstat_labels = {
    0:'Unemployed',
    1:'Employed',
}
eempstat_counts = filtered_df['EEMPSTAT'].value_counts().sort_index()
plt.figure(figsize=(12, 6)) 
plt.bar([eempstat_labels[code] for code in eempstat_counts.index], eempstat_counts.values, color='purple')
plt.xlabel('Employment Status')
plt.ylabel('Frequency')
plt.title('Frequency of Employment Status')

plt.xticks(rotation=45, ha='right')
plt.yscale('linear')
plt.show()


# In[25]:


# Univariate Analysis for Labour Force Participation
labforce_labels = {
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
plt.yscale('linear')
plt.show()


# In[24]:


# Univariate Analysis for Educational Attainment
edattain_labels = {
    0: 'Primary or Less Completed',
    1: 'At Least Secondary Completed',
}
edattain_counts = filtered_df['EDATTAIN'].value_counts().sort_index()
plt.figure(figsize=(15, 6)) 
plt.bar(edattain_counts.index, edattain_counts.values, color='purple')

plt.xlabel('Educational Attainment')
plt.ylabel('Frequency')
plt.title('Frequency by Educational Attainment')
plt.xticks(ticks=edattain_counts.index, labels=[edattain_labels[label] for label in edattain_counts.index], rotation=45, ha='right')
plt.yscale('linear')
plt.show()


# In[26]:


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
plt.yscale('linear')
plt.show()


# In[27]:


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
plt.yscale('linear')
plt.show()


# In[28]:


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
plt.yscale('linear')
plt.show()


# In[29]:


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
plt.yscale('linear')
plt.show()


# # Pearson's Correlation Coefficient Matrix

# In[30]:


# Pearson's Correlation Coefficient in Matrix form
df_corr = filtered_df.corr().transpose()
mask = np.triu(np.ones_like(df_corr))
f, ax = plt.subplots(figsize = (30, 20))
sns.heatmap(df_corr,mask=mask,fmt=".2f",annot=True)
plt.show()


# # Bivariate Analysis

# In[40]:


country_mapping = {
    40: 'Austria',
    300: 'Greece',
    348: 'Hungary',
    616: 'Poland',
    620: 'Portugal',
    642: 'Romania',
    724: 'Spain',
}

filtered_country = filtered_df.loc[filtered_df['HEAT'] == 0, 'COUNTRY']

# Mapping
filtered_country_mapped = filtered_country[filtered_country.isin(country_mapping.keys())]
country_names = filtered_country_mapped.map(country_mapping)
country_counts = country_names.value_counts()

# Contingency Table
if not country_counts.empty:
    
    contingency_table = pd.DataFrame({'COUNTRY': country_counts.index, 'Count': country_counts.values})

    # Bar chart
    plt.figure(figsize=(15, 6))
    sns.barplot(x='COUNTRY', y='Count', data=contingency_table, palette='YlGnBu')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.title('Count of Households without Heat (by Country)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No data available in the specified countries.")


# In[39]:


#Filtering records with no water access
filtered_country = filtered_df.loc[filtered_df['WATSUP'] == 0, 'COUNTRY']
filtered_country_mapped = filtered_country[filtered_country.isin(country_mapping.keys())]

# Mapping
country_codes = filtered_country_mapped.map(country_mapping)
country_counts = country_codes.value_counts()

# Contingency Table
if not country_counts.empty:

    contingency_table = pd.DataFrame({'COUNTRY': country_counts.index, 'Count': country_counts.values})

    # Bar chart
    plt.figure(figsize=(15, 6))
    sns.barplot(x='COUNTRY', y='Count', data=contingency_table, palette='YlGnBu')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.title('Count of Households without Water Supply (by Country)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No data available in the specified countries.")


# In[42]:


# Homemakers x Family Size
filtered_data = df[df['EEMPSTAT'] == 230]

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
y_labels=['Primary and Below Completed','Secondary or Above Completed']


plt.figure(figsize=(20, 8))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu',xticklabels=x_labels,yticklabels=y_labels)

# Labeling x and y axes
plt.xlabel('Country')
plt.ylabel('Educational Attainment')
plt.title("Educational Attainment by Country")

plt.tight_layout()
plt.show()


# In[44]:


# Employment Status x Sex
contingency_table = pd.crosstab(filtered_df['SEX'], filtered_df['LABFORCE'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:, :]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

#Labeling x and y axis ticks
x_labels=['Unemployed','Employed']
y_labels = ['Male','Female']

plt.figure(figsize=(8, 6))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=x_labels, yticklabels=y_labels)

# Labeling x and y axes
plt.xlabel('Employment Status')
plt.ylabel('Sex')
plt.title("Employment Status Versus Sex")

plt.tight_layout()
plt.show()


# In[47]:


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


# In[43]:


# Heat x Water Supply
contingency_table = pd.crosstab(index=filtered_df['WATSUP'], columns=filtered_df['HEAT'])

percentage_table = contingency_table.apply(lambda r: r/r.sum(), axis=1) * 100

watsup_labels = ['No Piped Water','Piped Water']
heat_labels = ['No Heating Available', 'Heating Available']

plt.figure(figsize=(8, 6))
sns.heatmap(percentage_table, annot=True, fmt='.1f', cmap='YlGnBu',xticklabels=heat_labels, yticklabels=watsup_labels)
plt.xlabel('Heat')
plt.ylabel('Water Supply')
plt.title('Contingency Table Heatmap: Water Supply vs. Heat (Percentages)')
plt.tight_layout()
plt.show()


# In[145]:


# Define country mapping
country_mapping = {
    40: 'Austria',
    300: 'Greece',
    348: 'Hungary',
    616: 'Poland',
    620: 'Portugal',
    642: 'Romania',
    724: 'Spain',
}

# Define labels for education categories
education_labels = {
    1: 'Primary and Below Completed',
    2: 'Secondary or Above Completed'
}

# Define labels for labor force categories
labforce_labels = {
    1: 'Unemployed',
    2: 'Employed'
}

# Create separate contingency tables and heatmaps for each country
for country_code, country_name in country_mapping.items():
    # Filter data for the specific country
    country_df = filtered_df[filtered_df['COUNTRY'] == country_code]
    
    # Create contingency table for education vs. labor force
    contingency_table = pd.crosstab(index=country_df['EDATTAIN'], columns=country_df['LABFORCE'])
    
    # Calculate proportions
    proportions_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)*100
    proportions_table = proportions_table.round(2)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(proportions_table, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=labforce_labels.values(), yticklabels=education_labels.values())
    plt.xlabel('Labor Force')
    plt.ylabel('Educational Attainment')
    plt.title(f'Educational Attainment vs. Labor Force Participation in {country_name}')
    plt.tight_layout()
    plt.show()


# In[45]:


# Educational Attainment x Marital Status
contingency_table = pd.crosstab(filtered_df['MARST'], filtered_df['EDATTAIN'])

# Narrowing down the rows/columns based on relevance
contingency_table = contingency_table.iloc[:, :]

# Calculating the grand total of the contingency table and obtaining percentages
grand_total = contingency_table.values.sum()
proportions_table = contingency_table / grand_total
proportions_percent = (proportions_table * 100).round(2)

# Labeling x and y axis ticks
x_labels=['Up to Primary Completed','Secondary or Above Completed']
y_labels=['Single/never married','Married/in union','Separated/divorced/spouse absent','Widowed']

plt.figure(figsize=(20, 8))
sns.heatmap(proportions_table, annot=True, fmt='.2%', cmap='YlGnBu',xticklabels=x_labels,yticklabels=y_labels)

# Labeling x and y axes
plt.xlabel('Educational Attainment')
plt.ylabel('Marital Status')
plt.title("Educational Attainment Versus Marital Status")

plt.tight_layout()
plt.show()


# # Measuring Household Density

# In[48]:


#Confirming the proportion of single-family households is a majority
nfams_values = filtered_df['NFAMS'].value_counts()

(nfams_values/np.sum(nfams_values))*100


# In[49]:


# # Filtering out households with over 1 family
nfams_conditions = [
    (filtered_df['NFAMS'].isin([2,3,4,5,6,7,8,9]))
]

filtered_df = filtered_df[~pd.concat(nfams_conditions, axis=1).any(axis=1)]


# In[50]:


filtered_df['NFAMS'].value_counts()


# In[51]:


filtered_df['DENSITY'] = filtered_df['ROOMS'] / filtered_df['FAMSIZE']
filtered_df['DENSITY'].value_counts()


# In[52]:


#checking for division by zero error
filtered_df['DENSITY'].max()


# In[53]:


density_counts = filtered_df['DENSITY'].value_counts().sort_index()
plt.figure(figsize=(15, 5)) 
plt.bar(density_counts.index,density_counts.values, color='purple')

plt.xlabel('Household Density')
plt.ylabel('Frequency')
plt.title('Frequency by Household Density')

plt.xticks(range(0, 11, 1), rotation=45, ha='right')

plt.yscale('linear')

plt.show()


# # Creating a new Water/Heat variable

# In[54]:


#Creating a new variable that combines water and heat

filtered_df['WH'] = ((filtered_df['WATSUP'] == 1) & (filtered_df['HEAT'] == 1)).astype(int)

print(filtered_df[['WATSUP', 'HEAT', 'WH']])


# In[55]:


# Looking at the new 'WH' variable in each country
def show_value_counts_percentage(data, countries):
    country_mapping = {
        40: 'Austria',
        300: 'Greece',
        348: 'Hungary',
        616: 'Poland',
        620: 'Portugal',
        642: 'Romania',
        724: 'Spain'
    }
    
    for country_code in countries:
        
        country_name = country_mapping.get(country_code, 'Unknown')
        
        country_data = data[data['COUNTRY'] == country_code]
        
        # Percentage form
        value_counts_percentage = country_data['WH'].value_counts(normalize=True) * 100
        
        print(f"Value counts for Water and Heat Access in {country_name} (in percentage):")
        print(value_counts_percentage)
        print("\n")

countries_to_analyze = [40, 300, 348, 616, 620, 642, 724]

show_value_counts_percentage(filtered_df, countries_to_analyze)


# # Feature Selection

# In[57]:


filtered_df5 = filtered_df.copy()
filtered_df6 = filtered_df.copy()


# # Polychoric PCA

# In[58]:


import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import FactorAnalyzer, Rotator
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo


ordinal_columns = ['WH','EDATTAIN', 'LABFORCE', 'DENSITY']

ordinal_data = filtered_df5[ordinal_columns]

# Calculating Bartlett's Sphericity test
chi_square_value, p_value = calculate_bartlett_sphericity(ordinal_data)
print(f"Bartlett's test statistic: {chi_square_value:.4f}")
print(f"P-value: {p_value:.4f}")

# Calculating Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy
kmo_all, kmo_model = calculate_kmo(ordinal_data)
print(f"KMO for each variable:\n{kmo_all}")
print(f"KMO overall: {kmo_model}")


# In[59]:


# Performing Polychoric PCA
n_factors = 2
fa = FactorAnalyzer(n_factors, rotation='oblimin') #non-orthogonal rotations: oblimin rotation, oblique rotation
fa.fit(ordinal_data)

# Obtaining Factor Loadings
print("Factor Loadings:")
print(fa.loadings_)


# In[60]:


# Displaying Factor loadings via DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])]

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[61]:


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


# In[62]:


explained_variance


# # Only keeping the first 2 factors

# In[63]:


# Reducing the number of factors
n_factors = 2
fa = FactorAnalyzer(n_factors, rotation='oblimin')
fa.fit(ordinal_data)

# Obtaining Factor Loadings
print("Factor Loadings:")
print(fa.loadings_)


# In[64]:


# Displaying factor loadings via DataFrame
factor_loadings = fa.loadings_
observed_variables = ordinal_columns
factor_names = [f"Factor {i+1}" for i in range(factor_loadings.shape[1])]

loadings_df = pd.DataFrame(factor_loadings, index=observed_variables, columns=factor_names)

print(loadings_df)


# In[65]:


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


# In[66]:


# Calculating absolute values of factor loadings (Formula 6)
abs_factor_loadings = np.abs(fa.loadings_)

# Calculating the sum of absolute values for each factor
sum_abs_loadings = abs_factor_loadings.sum(axis=0)

# Calculating the weights proportional to absolute values of factor loadings (Formula 7)
factor_weights = sum_abs_loadings / sum_abs_loadings.sum()

print("Factor Weights:")
print(factor_weights)


# In[67]:


# Calculating the composite poverty indicator for each observation (Formula 7)
composite_indicator = np.dot(normalized_factor_scores, factor_weights)

min_value = composite_indicator.min()
max_value = composite_indicator.max()
normalized_indicator = (composite_indicator - min_value) / (max_value - min_value)

filtered_df5['Composite_Indicator'] = normalized_indicator


# In[68]:


mpl.rcParams["axes.formatter.limits"] = (-99, 99)

# Histogram of composite scores
plt.figure(figsize=(12, 6))
sns.histplot(data=filtered_df5, x='Composite_Indicator', bins=30, kde=True)
plt.xlabel('Composite Indicator')
plt.ylabel('Frequency')
plt.title('Histogram of Composite Indicator Scores')
plt.show()


# In[69]:


# Violin Plot
plt.figure(figsize=(20, 6))
sns.violinplot(x=filtered_df5['Composite_Indicator'], color='blue', inner='quartiles')
plt.xlabel('Composite Indicator')
plt.ylabel('Density')
plt.xticks(np.arange(0,1.03, 0.03))
plt.title('Distribution of Composite Indicator Scores')
plt.grid(True)
plt.show()


# In[70]:


#Setting thresholds
filtered_df5['Deprivation'] = 'Not Deprived'

filtered_df5.loc[filtered_df5['Composite_Indicator'] <= 0.18, 'Deprivation'] = 'Most Deprived'
filtered_df5.loc[(filtered_df5['Composite_Indicator'] > 0.18) & (filtered_df5['Composite_Indicator'] <= 0.35), 'Deprivation'] = 'Less Deprived'


# In[71]:


filtered_df5['Deprivation'].value_counts()


# In[72]:


pd.set_option('display.float_format', '{:.4f}'.format)
deprived_summary = filtered_df5[filtered_df5['Deprivation'] == 'Most Deprived']['Composite_Indicator'].describe()
print(deprived_summary)


# In[73]:


pd.set_option('display.float_format', '{:.4f}'.format)
deprived_summary = filtered_df5[filtered_df5['Deprivation'] == 'Less Deprived']['Composite_Indicator'].describe()
print(deprived_summary)


# In[74]:


pd.set_option('display.float_format', '{:.4f}'.format)
deprived_summary = filtered_df5[filtered_df5['Deprivation'] == 'Not Deprived']['Composite_Indicator'].describe()
print(deprived_summary)


# In[75]:


pd.set_option('display.float_format', '{:.4f}'.format)

# Describing 'Not Deprived'
affluent_describe = filtered_df5[filtered_df5['Deprivation'] == 'Not Deprived'][['WH', 'EDATTAIN', 'LABFORCE', 'DENSITY']].describe()

print("\nSummary for Affluent Category:")
print(affluent_describe)


# In[76]:


pd.set_option('display.float_format', '{:.4f}'.format)

# Describing 'Less Deprived'
not_deprived_describe = filtered_df5[filtered_df5['Deprivation'] == 'Less Deprived'][['WH', 'EDATTAIN', 'LABFORCE', 'DENSITY']].describe()

print("\nSummary for Not Deprived Category:")
print(not_deprived_describe)


# In[77]:


deprived_describe = filtered_df5[filtered_df5['Deprivation'] == 'Most Deprived'][['WH', 'EDATTAIN', 'LABFORCE', 'DENSITY']].describe()

# Describing 'Most Deprived'
print("\nSummary for Deprived Category:")
print(deprived_describe)


# In[78]:


# Histogram
plt.figure(figsize=(12, 6))
sns.histplot(data=filtered_df5, x='Composite_Indicator', kde=True)
plt.xlabel('Composite Indicator')
plt.ylabel('Frequency')
plt.title('Histogram of Composite Indicator Scores')
plt.grid(True)
plt.show()


# In[79]:


# Investigating the Top 10% and Bottom 10% of Indicator Scores
top_10_cutoff = np.percentile(normalized_indicator, 90)
top_5_cutoff = np.percentile(normalized_indicator, 95)
bottom_10_cutoff = np.percentile(normalized_indicator, 10)

# Sectioning off the data
top_10_subset = filtered_df5[filtered_df5['Composite_Indicator'] >= top_10_cutoff]
top_5_subset = filtered_df5[filtered_df5['Composite_Indicator'] >= top_5_cutoff]
bottom_10_subset = filtered_df5[filtered_df5['Composite_Indicator'] <= bottom_10_cutoff]


top_10_subset['Composite_Indicator'].describe()


# In[80]:


bottom_10_subset['Composite_Indicator'].describe()


# # Exploring Invariance

# In[81]:


country_mapping = {
    40: 'Austria',
    300: 'Greece',
    348: 'Hungary',
    616: 'Poland',
    620: 'Portugal',
    642: 'Romania',
    724: 'Spain',
}


# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer.factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

def run_measurement_invariance(data, countries):
    results = {}

    for country in countries:
        
        country_data = data[data['COUNTRY'] == country]

        # Reset index to avoid any issues with indexing
        country_data = country_data.reset_index(drop=True)

        # Ordinal columns
        ordinal_columns = ['WH', 'EDATTAIN', 'LABFORCE', 'DENSITY']

        ordinal_data = country_data[ordinal_columns].reset_index(drop=True)

        # Bartlett's Sphericity
        chi_square_value, p_value = calculate_bartlett_sphericity(ordinal_data)
        print(f"Bartlett's test statistic: {chi_square_value:.4f}")
        print(f"P-value: {p_value:.4f}")

        # KMO
        kmo_all, kmo_model = calculate_kmo(ordinal_data)
        print(f"KMO for each variable:\n{kmo_all}")
        print(f"KMO overall: {kmo_model}")

        # Polychoric PCA
        n_factors = 2
        fa = FactorAnalyzer(n_factors, rotation='oblimin')
        fa.fit(ordinal_data)

        # Factor Loadings
        print(f"{country} - Factor Loadings:")
        print(fa.loadings_)

        # Factor scores (regression-based method)
        factor_scores = np.dot(ordinal_data, fa.loadings_)

        # Standard deviations of factor scores
        factor_scores_std = factor_scores.std(axis=0)

        # Checking for zero standard deviations
        non_zero_std_indices = factor_scores_std > 0

        # Normalizing factor scores
        normalized_factor_scores = np.zeros_like(factor_scores)
        normalized_factor_scores[:, non_zero_std_indices] = (factor_scores[:, non_zero_std_indices] - factor_scores[:, non_zero_std_indices].mean(axis=0)) / factor_scores_std[non_zero_std_indices]

        print(f"{country} - Factor Scores:")
        print(normalized_factor_scores)

        # Absolute values of factor loadings
        abs_factor_loadings = np.abs(fa.loadings_)

        # Sum of absolute values for each factor
        sum_abs_loadings = abs_factor_loadings.sum(axis=0)

        # Factor Weights
        factor_weights = sum_abs_loadings / sum_abs_loadings.sum()

        print(f"{country} - Factor Weights:")
        print(factor_weights)

        # Composite indicator for each observation
        composite_indicator = np.dot(normalized_factor_scores, factor_weights)

        # Normalizing composite indicator
        min_value = composite_indicator.min()
        max_value = composite_indicator.max()
        normalized_indicator = (composite_indicator - min_value) / (max_value - min_value)

        country_data['Composite_Indicator'] = normalized_indicator

        results[country] = {
            'Bartlett_Test_Statistic': chi_square_value,
            'P_Value_Bartlett': p_value,
            'KMO_All': kmo_all,
            'KMO_Overall': kmo_model,
            'Factor_Loadings': fa.loadings_,
            'Factor_Scores': normalized_factor_scores,
            'Factor_Weights': factor_weights,
            'Composite_Score': normalized_indicator
        }

        # Histogram of composite scores
        plt.figure(figsize=(12, 6))
        sns.histplot(data=country_data, x='Composite_Indicator', bins=30, kde=True)
        plt.xlabel('Composite Indicator Score')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Composite Indicator Scores - {country}')
        plt.show()

    return results

# Specify the countries for analysis
countries_to_analyze = [40, 300, 348, 616, 620, 642, 724]

results_dict = run_measurement_invariance(filtered_df4, countries_to_analyze)


# # Eigenvectors and Between-factor Correlations

# In[83]:


#Eigenvectors for each factor

# Factor weights matrix for two factors
factor_weights = np.array([
    [0.5672, 0.5544, 0.5891, 0.5333, 0.5283, 0.6175, 0.5304],  # Factor 1
    [0.4328, 0.4456, 0.4109, 0.4666, 0.4717, 0.3824, 0.4696]   # Factor 2
])

# Normalize the factor weights
normalized_factor_weights = factor_weights / np.linalg.norm(factor_weights, axis=1)[:, np.newaxis]

# The normalized_factor_weights matrix now contains the eigenvectors of each factor
# Each row of the matrix represents the eigenvector of the corresponding factor
for i, factor_eigenvector in enumerate(normalized_factor_weights, start=1):
    print(f"Eigenvector of Factor {i}: {factor_eigenvector}")


# In[84]:


# Factor loading scores. Poland and Portugal's factor orders were switched to align with the rest of the factor structures
factor_loading_scores = {
    'Austria': np.array([[0.0249,0.5365,0.6086,-0.0167], [0.0429,0.1794,-0.1275,0.5559]]),
    'Greece': np.array([[0.0632,0.5111,0.5352,-0.0083], [0.1477,0.1818,-0.1539,0.4152]]),
    'Hungary': np.array([[0.1235,0.5685,0.6136,-0.0026], [0.2608,0.1413,-0.1017,0.4087]]),
    'Poland': np.array([[0.1140,0.3711,0.5071,-0.0444],[0.1918,0.2984,-0.1101,0.5847]]),
    'Portugal': np.array([[0.1238,0.4003,0.5154,-0.0309],[0.2926,0.2276,-0.1007,0.5777]]),
    'Romania': np.array([[0.3713,0.9825,0.0004,0.1209], [-0.0176,0.0012,0.7927,-0.1022]]),
    'Spain': np.array([[0.1196,0.4845,0.5158,-0.0249], [0.2280,0.1969,-0.1581,0.4305]])
}

# Calculating between-factor correlations
factor_correlations = {}
for country, scores in factor_loading_scores.items():
    factor1_scores = scores[0]
    factor2_scores = scores[1]
    correlation = np.corrcoef(factor1_scores, factor2_scores)[0, 1]
    factor_correlations[country] = correlation


for country, correlation in factor_correlations.items():
    print(f"Correlation for {country}: {correlation:.4f}")


# In[ ]:




