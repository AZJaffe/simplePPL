import simplePPL
import unittest
import numpy as np
from lark import Lark

class Test(unittest.TestCase):

  def setUp(self):
    self.parser = Lark.open('./grammar.lark', start='simpleppl')

  def test_distributions(self):
    s = simplePPL.run('x ~ Unif(0,1)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Uniform')
    s = simplePPL.run('x ~ Bern(0.5)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Bernoulli')
    s = simplePPL.run('x ~ N(0,1)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Normal')
    s = simplePPL.run('x ~ Beta(5,2)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Beta')
    s = simplePPL.run('x ~ Pois(5)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Poisson')
    s = simplePPL.run('x ~ DUnif(0,1)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ DiscreteUniform')
    s = simplePPL.run('x ~ Binom(5,0.5)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Binomial')
    s = simplePPL.run('x ~ Geometric(0.5)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Geometric')
    s = simplePPL.run('x ~ Exp(5)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Exponential')
    s = simplePPL.run('x ~ Gamma(5,2)')
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Gamma')

  def test_examples(self):
    p = simplePPL.load('./examples/bayesian_bern.ppl')
    s = simplePPL.run(p)
    print(type(s['x']))

    p = simplePPL.load('./examples/bern.ppl')
    s = simplePPL.run(p)
    self.assertEqual(str(s.lookup_rv('x')), 'x ~ Bernoulli')

    p = simplePPL.load('./examples/mixture_of_gaussians.ppl')
    s = simplePPL.run(p)
    self.assertEqual(str(s.lookup_rv('z')), 'z ~ Bernoulli')
    self.assertEqual(str(s.lookup_rv('x1')), 'x1 ~ Normal')
    self.assertEqual(str(s.lookup_rv('x2')), 'x2 ~ Normal')


  def test_distribution_errors(self):
    try:
      s = simplePPL.run('x ~ NotADistribution(5,2)')
      self.assertTrue(False)
    except simplePPL.UndefinedDistribution:
      pass
    try:
      s = simplePPL.run('x ~ Bern()')
      self.assertTrue(False)
    except simplePPL.WrongArity:
      pass
    try:
      s = simplePPL.run('x ~ Bern(0.1, 0.2)')
      self.assertTrue(False)
    except simplePPL.WrongArity:
      pass

  def test_data_parse(self):
    p = self.parser.parse('x = [1,2,3]')
    d = p.children[0].children[1]
    a = simplePPL.parse_data(d)
    self.assertTrue((np.array([1,2,3]) == a).all())

    p = self.parser.parse('x = [1,[2],3]')
    d = p.children[0].children[1]
    try:
      a = simplePPL.parse_data(d)
      self.assertTrue(False)
    except simplePPL.InvalidDataLiteral:
      pass

if __name__ == '__main__':
    unittest.main()
