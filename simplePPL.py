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
    return self.data.get(var,None)

  def lookup_shape(self, var):
    if var in self.data:
      return self.data[var].shape
    if var in self.rvs:
      return self.rvs[var].dshape
    raise UnitializedVariable(var)

  def lookup_rv(self, var):
    return self.rvs.get(var,None)

def check_arity(dist, nargs):
  dists_to_arity = {
    'Bern': 1,
    'N': 2,
    'Unif': 2,
    'Beta': 2,
    'Pois': 1,
    'DUnif': 2,
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
    if stmt.data == 'assign':
      assign_stmt(store, stmt)
  return store

def distributed_stmt(store, stmt):
  var = stmt.children[0].value
  if len(stmt.children) == 2:
    shape = ()
    dist_stmt = stmt.children[1]
  else:
    shape = parse_shape(store, stmt.children[1])
    dist_stmt = stmt.children[2]
  dist = dist_stmt.children[0].value
  args = [process_numexpr(store, arg) for arg in dist_stmt.children[1:]]
  check_arity(dist, len(args))
  data = store.lookup_data(var)
  with store.model:
  # Discrete
    if dist == 'Bern':
      store.add_rv(var, pm.Bernoulli(var, p=args[0], observed=data, shape=shape))
    elif dist == 'Unif':
      store.add_rv(var, pm.Uniform(var, lower=args[0], upper=args[1], observed=data, shape=shape))
    elif dist == 'Beta':
      store.add_rv(var, pm.Beta(var, alpha=args[0], beta=args[1], observed=data, shape=shape))
    elif dist == 'Pois':
      store.add_rv(var, pm.Poisson(var, mu=args[0], observed=data, shape=shape))
    elif dist == 'DUnif':
      store.add_rv(var, pm.DiscreteUniform(var, lower=args[0], upper=args[1], observed=data, shape=shape))
    elif dist == 'Binom':
      store.add_rv(var, pm.Binomial(var, n=args[0], p=args[1], observed=data, shape=shape))
    elif dist == 'Geometric':
      store.add_rv(var, pm.Geometric(var, p=args[0], observed=data, shape=shape))
    # Continuous
    elif dist == 'N':
      store.add_rv(var, pm.Normal(var, mu=args[0], sigma=args[1], observed=data, shape=shape))
    elif dist == 'Gamma':
      store.add_rv(var, pm.Gamma(var, alpha=args[0], beta=args[1], observed=data, shape=shape))
    elif dist == 'Exp':
      store.add_rv(var, pm.Exponential(var, lam=args[0], testval=0, observed=data, shape=shape))

def data_assign_stmt(store, stmt):
  var = stmt.children[0].value
  data = stmt.children[1]
  a = parse_data(data)
  store.add_data(var, a)

def assign_stmt(store, stmt):
  var = stmt.children[0].value
  data = stmt.children[1]
  a = process_numexpr(store, data)
  with store.model:
    store.add_rv(var, pm.Deterministic(var, a))

def parse_data(tree):
  if tree.data == 'number':
    return tree.children[0].value
  if len(tree.children) == 0:
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

def parse_shape(store, shape_expr):
  shape = ()
  for shapearg in shape_expr.children:
    shape = shape + parse_shapearg(store, shapearg)
  return shape

def parse_shapearg(store, shapearg):
  if shapearg.data == 'shapearg':
    val = int(shapearg.children[0].value)
    assert val >= 0
    return (val,)
  elif shapearg.data == 'likeother':
    var = shapearg.children[0].value
    varshape = store.lookup_shape(var)
    if len(shapearg.children) == 2:
      part = int(shapearg.children[1].value)
      return (varshape[part],)
    return varshape

def process_numexpr(store, numexp):
  # Base Cases
  if numexp.data == 'number':
    return float(numexp.children[0].value)
  if numexp.data == 'id':
    var = numexp.children[0].value
    rv = store.lookup_rv(var)
    if rv != None:
      return rv
    data = store.lookup_data(var)
    if data is not None:
      # Data is fixed, not a rv, but hack it into rv because it should work
      with store.model:
        d = pm.Data(var, data)
      store.add_rv(var, d)
      return d
  if numexp.data == 'call':
    # In this case, the first child is the identifier of the fn to call, the rest are the args
    children = numexp.children[1:]
  else:
    children = numexp.children

  # Recursive cases
  processed_children = [None]*len(children)
  for i,child in enumerate(children):
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
  elif numexp.data == 'matmul':
    return processed_children[0] @ processed_children[1]
  elif numexp.data == 'call':
    f_name = numexp.children[0].value
    # Will throw if f_name is not a tt function
    return getattr(pm.math, f_name)(*processed_children)
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
