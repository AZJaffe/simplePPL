import pymc3 as pm
import numpy as np

N = 40
z_obs = np.random.binomial(1, 0.2, size=N)
n1 = np.random.normal(loc=1., scale=1., size=N)
n2 = np.random.normal(loc=-1., scale=1., size=N)
y_obs = z_obs * n1 + (1-z_obs) * n2
print(y_obs)
with pm.Model() as model:
  a = pm.Beta('a', 2,2)
  p = pm.Bernoulli('p', a, shape=N)
  x1 = pm.Normal('x1', 1,1,shape=N)
  x2 = pm.Normal('x2', -1,1,shape=N)

model.sample()
