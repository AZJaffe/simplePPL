from lark import Lark
import sys
import pymc3 as pm
import numpy as np
import arviz as az

class UnitializedVariable(Exception):
  def __init__(self, var):
    super().__init__(f'Random variable {var} used before declared')

class DuplicateVariable(Exception):
  def __init__(self, var):
    super().__init__(f'Random variable {var} declared twice')

class AssignAfterDistributed(Exception):
  def __init__(self, var):
    super().__init__(f'Random variable {var} assigned data after distributed statement')

class WrongArity(Exception):
  def __init__(self, dist, actual, expected):
    super().__init__(f'Distribution {dist} expected {expected} arguments but received {actual}')

class UndefinedDistribution(Exception):
  def __init__(self, dist):
    super().__init__(f'Unknown distribution {dist}')

class InvalidDataLiteral(Exception):
  def __init__(self, var=''):
    super().__init__()

  def set_var(self, var):
    self.var = var

  def __str__(self):
    if self.var == '':
      return f'Invalid data literal'
    else:
      return f'Invalid data literal for variable {var}'

class Store():
  def __init__(self, model):
    self.rvs = {}
    self.data = {}
    self.model = model

  def add_rv(self, var, rv):
    if var in self.rvs:
      raise AssignAfterDistributed(var)
    self.rvs[var] = rv

  def add_data(self, var, data):
    if var in self.rvs:
      raise DuplicateVariable(var)
    self.data[var] = data

  def lookup_data(self, var):
    if var in self.data:
      return self.data[var]
    else:
      return None

  def lookup_rv(self, var):
    if var not in self.rvs:
      raise UnitializedVariable(var)
    return self.rvs[var]  

def check_arity(dist, nargs):
  dists_to_arity = {
    'Bern': 1,
    'N': 2,
    'Unif': 2,
    'Beta': 2,
    'Pois': 1,
    'DUnif': 2,
    'Dir': 1,
    'Gamma': 2,
    'Exp': 1,
    'Geometric': 1,
    'Binom': 2,
  }
  if dist not in dists_to_arity:
    raise UndefinedDistribution(dist)
  if dists_to_arity[dist] != nargs:
    raise WrongArity(dist, nargs, dists_to_arity[dist])

def run(program):
  parser = Lark.open('./grammar.lark', start='simpleppl')
  p = parser.parse(program)
  store = Store(pm.Model())
  for stmt in p.children:
    if stmt.data == 'distributed':
      distributed_stmt(store, stmt)
    if stmt.data == 'dataassign':
      data_assign_stmt(store, stmt)
  return store

def distributed_stmt(store, stmt):
  var = stmt.children[0].value
  dist_stmt = stmt.children[1]
  dist = dist_stmt.children[0].value
  args = [process_numexpr(store, arg) for arg in dist_stmt.children[1:]]
  check_arity(dist, len(args))
  data = store.lookup_data(var)
  with store.model:
  # Discrete
    if dist == 'Bern':
      store.add_rv(var, pm.Bernoulli(var, p=args[0], observed=data))
    elif dist == 'Unif':
      store.add_rv(var, pm.Uniform(var, lower=args[0], upper=args[1], observed=data))
    elif dist == 'Beta':
      store.add_rv(var, pm.Beta(var, alpha=args[0], beta=args[1], observed=data))
    elif dist == 'Pois':
      store.add_rv(var, pm.Poisson(var, mu=args[0], observed=data))
    elif dist == 'DUnif':
      store.add_rv(var, pm.DiscreteUniform(var, lower=args[0], upper=args[1], observed=data))
    elif dist == 'Binom':
      store.add_rv(var, pm.Binomial(var, n=args[0], p=args[1], observed=data))
    elif dist == 'Geometric':
      store.add_rv(var, pm.Geometric(var, p=args[0], observed=data))
    # Continuous
    elif dist == 'N':
      store.add_rv(var, pm.Normal(var, mu=args[0], sigma=args[1], observed=data))
    # elif dist == 'Dir':
    #   store.add_rv(var, var, m.Var(var, pm.Dirchlet.dist(a=args[0]), data=data))
    elif dist == 'Exp':
      store.add_rv(var, pm.Exponential(var, lam=args[0], testval=0, observed=data))
    # Multivariate
    elif dist == 'Gamma':
      store.add_rv(var, pm.Gamma(var, alpha=args[0], beta=args[1], observed=data))

def data_assign_stmt(store, stmt):
  var = stmt.children[0].value
  data = stmt.children[1]
  a = parse_data(data)
  store.add_data(var, a)

def parse_data(tree):
  if tree.data == 'number':
    return tree.children[0].value
  if len(data.children) == 0:
    return np.array([])
  if tree.children[0].data == 'number':
    a = np.zeros((len(tree.children)))
    for i,child in enumerate(tree.children):
      if child.data != 'number':
        raise InvalidDataLiteral()
      a[i] = float(child.children[0])
    return a
  else:
    a = [parse_data(child) for child in tree.children]
    for subarray in a:
      if a[0].shape != subarray.shape:
        raise InvalidDataLiteral()
    return np.stack(a)

def process_numexpr(store, numexp):
  # Base Cases
  if numexp.data == 'number':
    return float(numexp.children[0].value)
  if numexp.data == 'id':
    return store.lookup_rv(numexp.children[0].value)

  # Recursive cases
  processed_children = [None]*len(numexp.children)
  for i,child in enumerate(numexp.children):
    processed_children[i] = process_numexpr(store, child)
  if numexp.data == 'sum':
    return processed_children[0] + processed_children[1]
  elif numexp.data == 'difference':
    return processed_children[0] - processed_children[1]
  elif numexp.data == 'product':
    return processed_children[0] * processed_children[1]
  elif numexp.data == 'quotient':
    return processed_children[0] / processed_children[1]
  elif numexp.data == 'negation':
    return -processed_children[0]
  elif numexp.data == 'parantheses':
    return processed_children[0]

def load(path):
  with open(path, 'r') as file:
    return file.read()

if __name__ == '__main__':
  if len(sys.argv) < 2:
    raise RuntimeError('Expected a .ppl file to run')
  s = run(load(sys.argv[1]))
  trace = pm.sample(model=s.model, return_inferencedata=True)
  print(az.summary(trace))
