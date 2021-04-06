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
    'Unif': 2
  }
  if dist not in dists_to_arity:
    raise UndefinedDistribution(dist)
  if dists_to_arity[dist] != nargs:
    raise WrongArity(dist, nargs, dists_to_arity[dist])

def get_num_args(arglist):
  if len(arglist.children) == 0:
    return 0
  return 1 + len(arglist.children[1].children)

def process_arglist(store, arglist):
  if len(arglist.children) == 0:
    return []
  args = [process_numexpr(store, arglist.children[0])]
  for a in arglist.children[1].children:
    args.append(process_numexpr(store, a))
  return args

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
  arglist = dist_stmt.children[1]
  args = process_arglist(store, arglist)
  check_arity(dist, len(args))
  if dist == 'bern':
    store[var] = m.Var(var, pm.Bernoulli.dist(p=args[0], testval=0.5))

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
  if numexp.data == 'difference':
    return processed_children[0] - processed_children[1]
  if numexp.data == 'product':
    return processed_children[0] * processed_children[1]
  if numexp.data == 'quotient':
    return processed_children[0] / processed_children[1]
  if numexp.data == 'negation':
    return -processed_children[0]
  if numexp.data == 'parantheses':
    return processed_children[0]

if __name__ == '__main__':
  if len(sys.argv) < 2:
    raise RuntimeError('Expected a .ppl file to run')
  with open(sys.argv[1], 'r') as file:
    program = file.read()
  s = run(program)
