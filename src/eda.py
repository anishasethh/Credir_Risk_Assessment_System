# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df: pd.DataFrame)-> None:

    # Distribution of Target Variable
    plt.figure(figsize=(12,8))
    sns.countplot(x="TARGET", data=df)
    plt.title('Target Classification')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.show()

    # Gender based distribution
    male  = df[df['CODE_GENDER']=='M']
    female = df[df['CODE_GENDER']=='F']
    
    labels = ['Male', 'Female']
    sizes = [len(male), len(female)]
    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue','pink'])
    plt.title('Gender Distribution')
    plt.axis('equal')
    plt.show()

    # Applications across different age groups
    plt.figure(figsize=(12,8))
    sns.histplot(df['DAYS_BIRTH']/-365, bins=25, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Applications')
    plt.show()

    # Target insights across different income types
    inc = df.groupby('NAME_INCOME_TYPE')['TARGET'].sum().reset_index()
    plt.figure(figsize=(17,8))
    sns.barplot(data=inc, x='NAME_INCOME_TYPE', y='TARGET', color='darkblue')
    plt.title('Defaulted Users across different income types')
    plt.xlabel('Income Types')
    plt.ylabel('Defaulted Users')
    plt.xticks(ticks=range(len(inc)), labels=inc['NAME_INCOME_TYPE'].astype(str), rotation=45)
    plt.show()

    # Target insights across different occupations
    occ1 = df.groupby('OCCUPATION_TYPE')['TARGET'].sum().reset_index()
    occ = occ1[occ1['OCCUPATION_TYPE'] != 0]
    plt.figure(figsize=(17,8))
    sns.barplot(data=occ, x='OCCUPATION_TYPE', y='TARGET', color='green')
    plt.title('Defaulted Users across different Occupations')
    plt.xlabel('Occupation Types')
    plt.ylabel('Defaulted Users')
    plt.xticks(ticks=range(len(occ)), labels=occ['OCCUPATION_TYPE'].astype(str), rotation=45)
    plt.show()

    # Count of 0's and 1's across luxury commodity owners
    vehicle = df.groupby(['FLAG_OWN_CAR', 'TARGET']).size().reset_index(name='count')
    vehicle['FLAG_OWN_CAR'] = vehicle['FLAG_OWN_CAR'].replace({'Y': 'Owns Car', 'N': 'No Car'})
    vcl_pivot = vehicle.pivot(index='FLAG_OWN_CAR', columns='TARGET', values='count')
    vcl_pivot.plot(kind='bar', stacked=True, figsize=(15,6),
                   color={0: 'skyblue', 1: 'darkblue'})
    plt.title('Loan Default Distribution Across Non-Car & Car owners')
    plt.xlabel('Type of Owner')
    plt.ylabel('Number of Users')
    plt.legend(title='TARGET', labels=['Non-Default (0)', 'Default (1)'])
    plt.xticks(rotation=0)
    plt.show()

    # Count of 0's and 1's across house owners
    home = df.groupby(['FLAG_OWN_REALTY', 'TARGET']).size().reset_index(name='count')
    home['FLAG_OWN_REALTY'] = home['FLAG_OWN_REALTY'].replace({'Y': 'Owns House', 'N': 'No House'})
    hm_pivot = home.pivot(index='FLAG_OWN_REALTY', columns='TARGET', values='count')
    hm_pivot.plot(kind='bar', stacked=True, figsize=(15,6),
                  color={0: 'lightgreen', 1: 'darkgreen'})
    plt.title('Loan Default Distribution Across Real Estate Ownership')
    plt.xlabel('Type of Owner')
    plt.ylabel('Number of Users')
    plt.legend(title='TARGET', labels=['Non-Default (0)', 'Default (1)'])
    plt.xticks(rotation=0)
    plt.show()

    # Target insights across different contract types
    occ = df.groupby('NAME_CONTRACT_TYPE')['TARGET'].sum().reset_index()
    plt.figure(figsize=(10,8))
    sns.barplot(data=occ, x='NAME_CONTRACT_TYPE', y='TARGET', color='darkred')
    plt.title('Defaulted Users across different types of loans')
    plt.xlabel('Contract Types')
    plt.ylabel('Defaulted Users')
    plt.xticks(ticks=range(len(occ)), labels=occ['NAME_CONTRACT_TYPE'].astype(str))
    plt.show()
