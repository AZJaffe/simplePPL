import simplePPL

p = simplePPL.load('./examples/bern.ppl')
s = simplePPL.run(p)
assert str(s['x']) == 'x ~ Bernoulli'

p = simplePPL.load('./examples/mixture_of_gaussians.ppl')
s = simplePPL.run(p)
assert str(s['z']) == 'z ~ Bernoulli'
assert str(s['x1']) == 'x1 ~ Normal'
assert str(s['x2']) == 'x2 ~ Normal'