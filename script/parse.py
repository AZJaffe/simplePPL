from lark import Lark
import sys
import simplePPL

if len(sys.argv) < 2:
  raise RuntimeError('Expected a .ppl file to run')
with open(sys.argv[1], 'r') as file:
  program = file.read()
parser = Lark.open('./grammar.lark', start='simpleppl')
p = parser.parse(program)