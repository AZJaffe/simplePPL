import simplePPL

p = simplePPL.load('./examples/bern.ppl')
s = simplePPL.run(p)
assert str(s['x']) == 'x ~ Bernoulli'

p = simplePPL.load('./examples/mixture_of_gaussians.ppl')
s = simplePPL.run(p)
assert str(s['z']) == 'z ~ Bernoulli'
assert str(s['x1']) == 'x1 ~ Normal'
assert str(s['x2']) == 'x2 ~ Normal'

s = simplePPL.run('x ~ Unif(0,1)')
assert str(s['x']) == 'x ~ Uniform'
s = simplePPL.run('x ~ Bern(0.5)')
assert str(s['x']) == 'x ~ Bernoulli'
s = simplePPL.run('x ~ N(0,1)')
assert str(s['x']) == 'x ~ Normal'
s = simplePPL.run('x ~ Beta(5,2)')
assert str(s['x']) == 'x ~ Beta'
s = simplePPL.run('x ~ Pois(5)')
assert str(s['x']) == 'x ~ Poisson'
s = simplePPL.run('x ~ DUnif(0,1)')
assert str(s['x']) == 'x ~ DiscreteUniform'
s = simplePPL.run('x ~ Binom(5,0.5)')
assert str(s['x']) == 'x ~ Binomial'
s = simplePPL.run('x ~ Geometric(0.5)')
assert str(s['x']) == 'x ~ Geometric'
s = simplePPL.run('x ~ Exp(5)')
assert str(s['x']) == 'x ~ Exponential'
s = simplePPL.run('x ~ Gamma(5,2)')
assert str(s['x']) == 'x ~ Gamma'

try:
  s = simplePPL.run('x ~ NotADistribution(5,2)')
  assert False
except simplePPL.UndefinedDistribution:
  print('Good job')

try:
  s = simplePPL.run('x ~ Bern()')
  assert False
except simplePPL.WrongArity:
  print('Good job')

try:
  s = simplePPL.run('x ~ Bern(0.1, 0.2)')
  assert False
except simplePPL.WrongArity:
  pass