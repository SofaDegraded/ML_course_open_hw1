import numpy as np
import pandas as pd
df = pd.read_csv('mlbootcamp5_train.csv', sep=';', 
                 index_col='id')
#print(df[df['gender'] == 1]['height'].mean())
#print(df[df['gender'] == 2]['height'].mean())
#print(df.groupby(['gender'])['alco'].describe())
#a = df.groupby(['gender'])['smoke'].value_counts(normalize=True)
#perc = round(a.values[2] / a.values[3])
#def get_age(col):
#    new_age = np.array([round((i/365.)) for i in col])
#    return new_age
def get_BMI(col_w, col_h):
    new_age = np.array([w / ((h/100)**2) for w, h in zip(col_w, col_h)])
    return new_age

#n_bmi = get_BMI(df['weight'], df['height'])
#new_bmi = pd.DataFrame({'bmi':n_bmi})
#df = df.join(new_bmi)
#print(df['bmi'].median())
#print(df[df['gender']== 1]['bmi'].mean())
#print(df[df['gender']== 2]['bmi'].mean())
#print(df[df['cardio']== 0]['bmi'].mean())
#print(df[df['cardio']== 1]['bmi'].mean())
#print(df[(df['cardio']== 0)&(df['alco']== 0)&(df['gender']== 1)]['bmi'].mean())
#print(df[(df['cardio']== 0)&(df['alco']== 0)&(df['gender']== 2)]['bmi'].mean())
#n_age = get_age(df['age'])
#new_age = pd.DataFrame({'age_years':n_age})
#df = df.join(new_age)
#df = df.drop(['age'], axis=1)
#smoke_man = df[(df['gender'] == 2)&(df['smoke'] == 1)&(df['age_years'] >= 60)&(df['age_years'] < 65)]
#smoke_man_1 = smoke_man[(smoke_man['ap_hi'] < 120)&(smoke_man['cholesterol'] == 1)]
#sm1_il = smoke_man_1[smoke_man_1['cardio'] == 1].shape[0]  / smoke_man_1.shape[0]
#smoke_man_2 = smoke_man[(smoke_man['ap_hi'] >= 160)&(smoke_man['ap_hi'] < 180)&(smoke_man['cholesterol'] == 3)]
#sm2_il = smoke_man_2[smoke_man_2['cardio'] == 1].shape[0] / smoke_man_2.shape[0]
#sm_il = round(sm2_il / sm1_il)
#print(df.head())
#a = df.groupby(['smoke'])['age'].agg(np.median)
#b1 = round(a[0]/365*12)
#b2 = round(a[1]/365*12)
#b = b1 - b2
f1 = df[df['ap_hi'] > df['ap_lo']]
f2 = f1[(f1['height']>f1['height'].quantile(0.025)) | (f1['height']<f1['height'].quantile(0.975))]
f3 = f2[(f2['weight']>f2['weight'].quantile(0.025)) | (f2['weight']<f2['weight'].quantile(0.975))]
f = df[(df['ap_hi'] > df['ap_lo'])&((df['height']>=df['height'].quantile(0.025)) & (df['height']<=f1['height'].quantile(0.975)))&\
    ((df['weight']>=df['weight'].quantile(0.025))& (df['weight']<=df['weight'].quantile(0.975)))]
print(1)