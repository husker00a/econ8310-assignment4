import pandas as pd

df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv")
df['retention_1'] = df['retention_1'].astype(int)
df['retention_7'] = df['retention_7'].astype(int)
df['gate30'] = (df['version'] == 'gate_30').astype(int)
df = df[['gate30','retention_1','retention_7']]

import pymc as pm

with pm.Model() as model:
    alpha = 1/df['retention_1'].mean() # enter retention_7 if modeling it

    lambda_1 = pm.Exponential("lambda_1", alpha) # retained
    lambda_2 = pm.Exponential("lambda_2", alpha) # not retained

     # Switch to assign lambda_1 where gate30==1 otherwise lambda_2
    lambda_ = pm.math.switch(df['gate30']==1, lambda_1, lambda_2)

    observation = pm.Exponential("obs", lambda_, observed=df['retention_1']) # enter retention_7 if modeling it

with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step, return_inferencedata=False)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']

print(lambda_1_samples.mean(), lambda_2_samples.mean())

print(lambda_1_samples.max(), lambda_2_samples.min())
