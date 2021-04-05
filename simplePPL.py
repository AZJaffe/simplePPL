from lark import Lark
import sys
import pymc3 as pm

class UnitializedVariable(Exception):
  def __init__(var):
    super().__init__(f'Random varaible {var} used before declared')

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
  var = stmt.children[0].children[0]
  distribution = stmt.children[1].data
  params = stmt.children[1].children
  assert distribution == 'bern'
  assert var == 'x'
  assert len(params) == 1
  processed_params = [None] * len(params)
  for i,child in enumerate(params):
    processed_params[i] = process_numexpr(store, child)
  if distribution == 'bern':
    store[var] = m.Var(var, pm.Bernoulli.dist(p=processed_params[0]))

def lookup(store, key):
  if key not in store:
    raise UnitializedVariable(key)
  return store[key]

def process_numexpr(store, numexp):
  # Base Cases
  if numexp.data == 'number':
    return float(numexp.children[0])
  if numexp.data == 'var':
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
  run(program)
