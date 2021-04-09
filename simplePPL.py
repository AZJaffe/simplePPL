from lark import Lark
import sys
import pymc3 as pm

class UnitializedVariable(Exception):
  def __init__(self, var):
    super().__init__(f'Random variable {var} used before declared')

class WrongArity(Exception):
  def __init__(self, dist, actual, expected):
    super().__init__(f'Distribution {dist} expected {expected} arguments but received {actual}')

class UndefinedDistribution(Exception):
  def __init__(self, dist):
    super().__init__(f'Unknown distribution {dist}')

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
  print(p)
  print(p.pretty())
  store = {}
  m = pm.Model()
  for stmt in p.children:
    if stmt.data == 'distributed':
      distributed_stmt(m, store, stmt)
  return store

def distributed_stmt(m, store, stmt):
  var = stmt.children[0].children[0].value
  dist_stmt = stmt.children[1]
  dist = dist_stmt.children[0].children[0].value
  args = [process_numexpr(store, arg) for arg in dist_stmt.children[1:]]
  check_arity(dist, len(args))
  # Discrete
  if dist == 'Bern':
    store[var] = m.Var(var, pm.Bernoulli.dist(p=args[0], testval=0.5))
  elif dist == 'Unif':
    store[var] = m.Var(var, pm.Uniform.dist(lower=args[0], upper=args[1], testval=args[0]))
  elif dist == 'Beta':
    store[var] = m.Var(var, pm.Beta.dist(alpha=args[0], beta=args[1], testval=0))
  elif dist == 'Pois':
    store[var] = m.Var(var, pm.Poisson.dist(mu=args[0], testval=0))
  elif dist == 'DUnif':
    store[var] = m.Var(var, pm.DiscreteUniform.dist(lower=args[0], upper=args[1], testval=0))
  elif dist == 'Binom':
    store[var] = m.Var(var, pm.Binomial.dist(n=args[0], p=args[1], testval=0))
  elif dist == 'Geometric':
    store[var] = m.Var(var, pm.Geometric.dist(p=args[0], testval=0))
  # Continuous
  elif dist == 'N':
    store[var] = m.Var(var, pm.Normal.dist(mu=args[0], sigma=args[1],testval=0))
  # elif dist == 'Dir':
  #   store[var] = m.Var(var, pm.Dirchlet.dist(a=args[0]))
  elif dist == 'Exp':
    store[var] = m.Var(var, pm.Exponential.dist(lam=args[0], testval=0))
  # Multivariate
  elif dist == 'Gamma':
    store[var] = m.Var(var, pm.Gamma.dist(alpha=args[0], beta=args[1], testval=0))

def lookup(store, key):
  if key not in store:
    raise UnitializedVariable(key)
  return store[key]

def process_numexpr(store, numexp):
  # Base Cases
  if numexp.data == 'number':
    return float(numexp.children[0])
  if numexp.data == 'identifier':
    return lookup(store, numexp.children[0])

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
