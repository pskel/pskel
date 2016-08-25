import argparse
import sys
import re
import sys
import multiprocessing
import os
from sets import Set

import sympy

from pycparser import parse_file, c_parser, c_ast, c_generator
from pycparser.plyparser import PLYParser, Coord, ParseError

from pyparsing import javaStyleComment, dblQuotedString, Suppress

WHITESPACE = ' \t\n\r\f\v'

def findPairChars(source, idx, openChar, closeChar, direction=1):
   count = direction
   while count!=0:
      if source[idx]==openChar:
         count += 1
      elif source[idx]==closeChar:
         count -= 1
      idx += direction
   if direction==-1:
      idx += 1
   return idx

def replaceSympyModToC(expr):
   expr = str(expr)
   while len(re.findall('(\W|^)Mod\s*\(',expr))>0:
      for m in re.finditer('(\W|^)Mod\s*\(',expr):
         begin = m.start(0)
         end = m.end(0)
         argsbeg = end
         end = findPairChars(expr, end+1, '(', ')')
         argsend = end
         args = expr[argsbeg:argsend-1].split(',')
         expr = expr[:begin]+'(('+args[0]+')%('+args[1]+'))'+expr[end:]
         break
   return expr

class KnownIDAnalyser(c_ast.NodeVisitor):
   def __init__(self, ids=[]):
      self.knownIds = ids;
      self.usedIds = []

   def getKnownUsedIDs(self):
      return Set(self.knownIds).intersection(Set(self.usedIds))

   def visit_Decl(self, node):
      self.usedIds.append(node.name)
      self.visit(node.type)
      if node.init:
         self.visit(node.init)

   def visit_ID(self, node):
      self.usedIds.append(node.name)

class PSkelKernelGenerator(c_generator.CGenerator,PLYParser):
   def __init__(self, pinfo):
      self.pinfo = pinfo
      self.finfo = None
      self.forFound = []
      super(PSkelKernelGenerator,self).__init__()

   def visit_ArrayRef(self, node):
      arrref = self._parenthesize_unless_simple(node.name)
      if arrref==self.finfo['in']:
         idxId = '('+self.visit(node.subscript)+')'
         try:
            subtract = '(('+self.finfo['height-id']+')*('+self.finfo['size'][0]+') + '+self.finfo['width-id']+')'
            idxExpr = sympy.simplify(sympy.sympify(idxId+'-'+subtract))
            
            print 'simplified:',idxExpr
            if str(idxExpr).strip()=='0':
               w = self.finfo['width-id']
               h = self.finfo['height-id']
            else:
               w = '('+str(sympy.simplify(sympy.sympify(idxId).subs(self.finfo['size'][0],'0')))+')'
               h = '('+str(sympy.simplify(sympy.sympify(idxId+'-'+w).subs(self.finfo['size'][0],'1')))+')'
               subtract = '('+h+'*('+self.finfo['size'][0]+') + '+w+')'
               idxExpr = sympy.simplify(sympy.sympify(idxId+'-'+subtract))
               if str(idxExpr).strip()!='0':
                  w = idxId+'%'+self.finfo['size'][0]
                  h = '('+idxId+'-('+w+'))/'+self.finfo['size'][0]
         except Exception as e:
            print 'Exception parsing expressions:', str(e)
            w = idxId+'%'+self.finfo['size'][0]
            h = '('+idxId+'-('+w+'))/'+self.finfo['size'][0]
         try:
            sh = sympy.sympify(h)
            sw = sympy.sympify(w)
            h = str(sympy.simplify(sh))
            w = str(sympy.simplify(sw))
            h = replaceSympyModToC(h)
            w = replaceSympyModToC(w)
         except Exception as e:
            print 'Exception parsing expressions:', str(e)
         return self.finfo['in']+'('+h+','+w+')'
      elif arrref==self.finfo['out']:
         idxId = '('+self.visit(node.subscript)+')'
         try:
            subtract = '(('+self.finfo['height-id']+')*('+self.finfo['size'][0]+') + '+self.finfo['width-id']+')'
            idxExpr = sympy.simplify(sympy.sympify(idxId+'-'+subtract))
            
            print 'simplified:',idxExpr
            if str(idxExpr).strip()=='0':
               w = self.finfo['width-id']
               h = self.finfo['height-id']
            else:
               w = '('+str(sympy.simplify(sympy.sympify(idxId).subs(self.finfo['size'][0],'0')))+')'
               h = '('+str(sympy.simplify(sympy.sympify(idxId+'-'+w).subs(self.finfo['size'][0],'1')))+')'
               subtract = '('+h+'*('+self.finfo['size'][0]+') + '+w+')'
               idxExpr = sympy.simplify(sympy.sympify(idxId+'-'+subtract))
               if str(idxExpr).strip()!='0':
                  w = idxId+'%'+self.finfo['size'][0]
                  h = '('+idxId+'-('+w+'))/'+self.finfo['size'][0]
         except Exception as e:
            print 'Exception parsing expressions:', str(e)
            w = idxId+'%'+self.finfo['size'][0]
            h = '('+idxId+'-('+w+'))/'+self.finfo['size'][0]
         try:
            sh = sympy.sympify(h)
            sw = sympy.sympify(w)
            h = str(sympy.simplify(sh))
            w = str(sympy.simplify(sw))
            h = replaceSympyModToC(h)
            w = replaceSympyModToC(w)
         except Exception as e:
            print 'Exception parsing expressions:', str(e)
         return self.finfo['out']+'('+h+','+w+')'
      else:
         return arrref + '[' + self.visit(node.subscript) + ']'

   def visit_ParamList(self, n):
      if self.finfo == None:
         self.finfo = {}
         self.finfo['in'] = n.params[0].name
         self.finfo['out'] = n.params[1].name
         self.finfo['size'] = [n.params[2].name,n.params[3].name]
         self.finfo['iterations'] = n.params[4].name
         print self.finfo
      return ', '.join(self.visit(param) for param in n.params)
   
   def visit_For(self, node):
      knownId = KnownIDAnalyser()
      knownId.visit(node.init)
      initIds = knownId.usedIds
      
      knownId = KnownIDAnalyser()
      knownId.visit(node.cond)
      condIds = knownId.usedIds

      knownId = KnownIDAnalyser()
      knownId.visit(node.next)
      nextIds = knownId.usedIds
      
      if self.finfo['iterations'] in Set(initIds).union(Set(condIds)):
         print 'For about iterations:',self.pinfo['iterations']
         self.forFound.append(self.pinfo['iterations'])
         if isinstance(node.stmt,c_ast.Compound):
            if len(self.forFound)==3:
               s = ''
               if node.stmt.block_items:
                  s += ''.join(self._generate_stmt(stmt,add_indent=False) for stmt in node.stmt.block_items)
               return s
            elif node.stmt.block_items and isinstance(node.stmt.block_items[0], c_ast.For):
               return self._generate_stmt(node.stmt.block_items[0],add_indent=False)
         else:
            return self.visit(node.stmt)
      elif len(Set(self.finfo['size']).intersection(Set(initIds).union(Set(condIds))))==1:
         print 'For about dimension:',Set(self.finfo['size']).intersection(Set(initIds).union(Set(condIds)))
         if list(Set(self.finfo['size']).intersection(Set(initIds).union(Set(condIds))))[0]==self.finfo['size'][0]:
            self.finfo['width-id'] = list(Set(nextIds).intersection(Set(initIds)))[0]
            self.forFound.append(self.finfo['size'][0])
         elif list(Set(self.finfo['size']).intersection(Set(initIds).union(Set(condIds))))[0]==self.finfo['size'][1]:
            self.finfo['height-id'] = list(Set(nextIds).intersection(Set(initIds)))[0]
            self.forFound.append(self.finfo['size'][1])
         if isinstance(node.stmt,c_ast.Compound):
            if len(self.forFound)==3:
               s = ''
               if node.stmt.block_items:
                  s += ''.join(self._generate_stmt(stmt,add_indent=False) for stmt in node.stmt.block_items)
               return s
            elif node.stmt.block_items and isinstance(node.stmt.block_items[0], c_ast.For):
               return self._generate_stmt(node.stmt.block_items[0],add_indent=False)
         else:
            return self.visit(node.stmt)
      else:
         return super(PSkelKernelGenerator,self).visit_For(node)
         
   def visit_FuncDef(self, n):
      decl = self.visit(n.decl)
      self.indent_level = 0
      if n.body.block_items and isinstance(n.body.block_items[0], c_ast.For):
         body = self._generate_stmt(n.body.block_items[0],add_indent=False)
      else:
         body = self.visit(n.body)
      s = 'namespace PSkel{\n'
      s += '__parallel__ void stencilKernel('
      arr_type = 'Array'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')+'<'+self.pinfo['in-type']+'>'
      mask_type = 'Mask'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')+'<'+self.pinfo['in-type']+'>'
      s += arr_type+' '+self.finfo['in']+','
      s += arr_type+' '+self.finfo['out']+','
      s += mask_type+' mask, size_t args,' 
      s += 'size_t '+self.finfo['height-id']+', size_t '+self.finfo['width-id']+')'
      defwidth = '#define  '+self.finfo['size'][0]+' '+self.finfo['in']+'.getWidth()\n'
      defheight = '#define '+self.finfo['size'][1]+' '+self.finfo['in']+'.getHeight()\n'
      undefwidth = '#undef  '+self.finfo['size'][0]+'\n'
      undefheight = '#undef '+self.finfo['size'][1]+'\n'
      return  s + '\n{\n'+defwidth+defheight + body + '\n'+undefwidth+undefheight+'}\n}\n'
            
class PSkelGenerator(c_generator.CGenerator,PLYParser):
   def __init__(self, pinfo, fcall_name):
      self.pinfo = pinfo
      self.fcall_name = fcall_name
      self.initialized = []
      self.defined = []
      self.receivedParameters = []
      super(PSkelGenerator,self).__init__()

   def visit_ParamList(self, node):
      argId = 0
      for param in node.params:
         if param.name == self.pinfo['in']:
             print 'WARNING: received as parameter '+self.pinfo['in']
             self.receivedParameters.append( (self.pinfo['in'], argId) ) #keep track of array
         if param.name == self.pinfo['out']:
             print 'WARNING: received as parameter '+self.pinfo['out']
             self.receivedParameters.append( (self.pinfo['out'], argId) ) #keep track of array
         if param.name in self.pinfo['size']:
             print 'WARNING: received as parameter '+param.name
             self.initialized.append(param.name)
         argId += 1
      return super(PSkelGenerator,self).visit_ParamList(node)

   def visit_ArrayRef(self, node):
      arrref = self._parenthesize_unless_simple(node.name)
      if arrref==self.pinfo['in']:
         idxId = '('+self.visit(node.subscript)+')'
         try:
            w = '('+str(sympy.simplify(sympy.sympify(idxId).subs(self.pinfo['size'][0],'0')))+')'
            h = '('+str(sympy.simplify(sympy.sympify(idxId+'-'+w).subs(self.pinfo['size'][0],'1')))+')'
            subtract = '('+h+'*('+self.pinfo['size'][0]+') + '+w+')'
            idxExpr = sympy.simplify(sympy.sympify(idxId+'-'+subtract))
            if str(idxExpr).strip()!='0':
               w = idxId+'%'+self.pinfo['size'][0]
               h = '('+idxId+'-('+w+'))/'+self.pinfo['size'][0]
         except Exception as e:
            print 'Exception parsing expressions:', str(e)
            w = idxId+'%'+self.pinfo['size'][0]
            h = '('+idxId+'-('+w+'))/'+self.pinfo['size'][0]
         try:
            sh = sympy.sympify(h)
            sw = sympy.sympify(w)
            h = str(sympy.simplify(sh))
            w = str(sympy.simplify(sw))
            h = replaceSympyModToC(h)
            w = replaceSympyModToC(w)
         except Exception as e:
            print 'Exception parsing expressions:', str(e)
         return self.pinfo['in']+'('+h+','+w+')'
      elif arrref==self.pinfo['out']:
         idxId = '('+self.visit(node.subscript)+')'
         try:
            w = '('+str(sympy.simplify(sympy.sympify(idxId).subs(self.pinfo['size'][0],'0')))+')'
            h = '('+str(sympy.simplify(sympy.sympify(idxId+'-'+w).subs(self.pinfo['size'][0],'1')))+')'
            subtract = '('+h+'*('+self.pinfo['size'][0]+') + '+w+')'
            idxExpr = sympy.simplify(sympy.sympify(idxId+'-'+subtract))
            if str(idxExpr).strip()!='0':
               w = idxId+'%'+self.pinfo['size'][0]
               h = '('+idxId+'-('+w+'))/'+self.pinfo['size'][0]
         except Exception as e:
            print 'Exception parsing expressions:', str(e)
            w = idxId+'%'+self.pinfo['size'][0]
            h = '('+idxId+'-('+w+'))/'+self.pinfo['size'][0]
         try:
            sh = sympy.sympify(h)
            sw = sympy.sympify(w)
            h = str(sympy.simplify(sh))
            w = str(sympy.simplify(sw))
            h = replaceSympyModToC(h)
            w = replaceSympyModToC(w)
         except Exception as e:
            print 'Exception parsing expressions:', str(e)
         return self.pinfo['out']+'('+h+','+w+')'
      else:
         return arrref + '[' + self.visit(node.subscript) + ']'

   def visit_FuncCall(self,node):
      fref = self._parenthesize_unless_simple(node.name)
      if fref==self.fcall_name:
         arr_type = 'Array'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')+'<'+self.pinfo['in-type']+'>'
         mask_type = 'Mask'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')+'<'+self.pinfo['in-type']+'>'
         s = mask_type+' '+self.fcall_name+'_mask;\n'+self._make_indent()
         s += 'Stencil'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')                  
         s += '<'+arr_type+', '+mask_type+', size_t> '
         s += self.fcall_name+'('+self.pinfo['in']+', '+self.pinfo['out']+', '+self.fcall_name+'_mask, 0);\n'+self._make_indent()
         if self.pinfo['device']=='gpu':
            s += self.fcall_name+'.runIterativeGPU('+self.pinfo['iterations']+', 0)'
         else:
            s += self.fcall_name+'.runIterativeCPU('+self.pinfo['iterations']+', 0)'
         return s
      else:
         return super(PSkelGenerator,self).visit_FuncCall(node)

   def visit_Decl(self, node, no_type=False):
      lval_str = self.visit(node.init)
      if node.name in self.pinfo['size'] and node.init:
         print 'Initializing attribute:', node.name
         self.initialized.append(node.name)
      if node.name==self.pinfo['in']:
         self.pinfo['in-type'] = self.visit(node.type)
         print 'Initializing attribute:', node.name
         if node.init and sum([(1 if var in self.initialized else 0) for var in self.pinfo['size']])==len(self.pinfo['size']) and (self.pinfo['in'] not in self.defined):
            if lval_str.strip() in ['NULL', '0']:
               print 'WARNING: left side is '+(lval_str.strip())
            self.defined.append(self.pinfo['in'])
            arr_type = 'Array'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')+'<'+self.pinfo['in-type']+'>'
            return '%s %s(%s)' % (arr_type, self.pinfo['in'], ','.join(self.pinfo['size']))
         else:
            return ''
      elif node.name==self.pinfo['out']:
         self.pinfo['out-type'] = self.visit(node.type)
         print 'Initializing attribute:', node.name
         if node.init and sum([(1 if var in self.initialized else 0) for var in self.pinfo['size']])==len(self.pinfo['size']) and (self.pinfo['out'] not in self.defined):
            if lval_str.strip() in ['NULL', '0']:
               print 'WARNING: left side is '+(lval_str.strip())
            self.defined.append(self.pinfo['out'])
            arr_type = 'Array'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')+'<'+self.pinfo['in-type']+'>'
            return '%s %s(%s)' % (arr_type, self.pinfo['out'], ','.join(self.pinfo['size']))
         else:
            return ''
      else:
         return super(PSkelGenerator,self).visit_Decl(node,no_type)

   def visit_Assignment(self, node):
      rval_str = self._parenthesize_if(node.rvalue, lambda node: isinstance(node, c_ast.Assignment))
      lval_str = self.visit(node.lvalue)
      if lval_str.strip() in self.pinfo['size']:
         print 'Assigning value to attribute:', lval_str.strip()
         self.initialized.append(lval_str.strip())
      if lval_str.strip()==self.pinfo['in']:
         if self.pinfo['in'] not in self.defined:
            if len(Set(self.pinfo['size']).intersection(Set(self.initialized)))!=len(self.pinfo['size']):
               print 'WARNING: uninitialized attributes '+Set(self.pinfo['size']).difference(Set(self.initialized))
            arr_type = 'Array'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')+'<'+self.pinfo['in-type']+'>'
            return '%s %s(%s)' % (arr_type, self.pinfo['in'], ','.join(self.pinfo['size']))
         else:
            print 'WARNING: redefining '+lval_str.strip()
            return ''
      elif lval_str.strip()==self.pinfo['out']:
         if self.pinfo['out'] not in self.defined:
            if len(Set(self.pinfo['size']).intersection(Set(self.initialized)))!=len(self.pinfo['size']):
               print 'WARNING: uninitialized attributes '+Set(self.pinfo['size']).difference(Set(self.initialized))
            arr_type = 'Array'+(str(self.pinfo['dim'])+'D' if self.pinfo['dim']>1 else '')+'<'+self.pinfo['in-type']+'>'
            return '%s %s(%s)' % (arr_type, self.pinfo['out'], ','.join(self.pinfo['size']))
         else:
            print 'WARNING: redefining '+lval_str.strip()
            return ''
      return '%s %s %s' % (lval_str, node.op, rval_str)


def splitArgs(args):   
   args = args.strip()[1:-1]
   splittedArgs = []
   idx = 0
   arg = ''
   while idx<len(args):
      if args[idx]==',':
         splittedArgs.append(arg.strip())
         arg = ''
         idx += 1
      else:
         if args[idx]=='(':
            end = findPairChars(args,idx+1,'(',')')
            arg = args[idx:end]
            idx = end
         else:
            arg += args[idx]
            idx += 1
   if len(arg.strip())>0:
      splittedArgs.append(arg.strip())
   return splittedArgs

def readFuncCall(source, idx):
   localsrc = source[idx:]
   call = ''
   for m in re.finditer('[_a-zA-Z][_a-zA-Z0-9]*',localsrc):
      fid = m.group(0)
      print fid
      begin = m.start(0)
      end = m.start(0)
      while localsrc[end]!='(':
         end += 1
      end = findPairChars(localsrc,end+1,'(',')')
      args = localsrc[begin+len(fid):end]
      while localsrc[end]!=';':
         end += 1
      print localsrc[begin:end+1]
      return {'name':fid,'args':splitArgs(args),'begin':idx+begin,'end':idx+end+1,'src':source[idx+begin:idx+end+1]}
   return None

def readOutterFunction(source, idx):
   begin = idx
   done = False
   while not done:
      while source[begin]!='{':
         begin -= 1
      bodyidx = begin
      while source[begin]!=')':
         begin -= 1
      begin = findPairChars(source,begin-1,'(',')',-1)
      begin -= 1
      while source[begin] in WHITESPACE:
         begin -= 1
      idname = ''
      while source[begin].isalnum():
         idname = source[begin]+idname
         begin -= 1
      if len(idname)==0:
         continue
      print 'idname:',idname
      while source[begin] in WHITESPACE:
         begin -= 1
      rettype = ''
      while source[begin].isalnum():
         rettype = source[begin]+rettype
         begin -= 1
      if len(rettype)==0:
         continue
      print 'rettype:',rettype
      done = True
   end = findPairChars(source, bodyidx+1, '{','}')
   return {'name':idname, 'src':source[begin:end],'begin':begin,'end':end}

def findFunctionDef(source, funcname):
   for m in re.finditer('void\s\s*'+funcname+'\s*\(', source):
      print m.group(0)
      begin = m.start(0)
      end = m.end(0)
      end = findPairChars(source, end, '(',')')
      while source[end]!='{':
         if source[end] not in WHITESPACE:
            print 'Wrong char found:',source[end]
            end = -1
            break
      if end<0:
         continue
      end = findPairChars(source, end+1, '{','}')
      return {'src':source[begin:end],'begin':begin,'end':end}
   return None

def parsePragma(pragma):
   if not (pragma.strip().startswith('#pragma') and pragma[len('#pragma'):].strip().startswith('pskel')):
      print 'wrong pragma:',pragma
      return None
   cmds = pragma[len('#pragma'):].strip()[len('pskel'):].strip()
   info = {}
   if cmds.startswith('stencil'):
      print 'stencil'
      cmds = cmds[len('stencil'):]
      info['type'] = 'stencil'
      for m in re.finditer('dim\dd',cmds):
         dim = int(m.group(0)[3])
         info['dim'] = dim
         idx = m.end(0)
         while cmds[idx]!='(':
            idx += 1
         argsbegin = idx
         args = cmds[argsbegin:findPairChars(cmds,argsbegin+1,'(',')')]
         args = [arg.strip() for arg in args.strip()[1:-1].split(',')]
         print args
         if len(args)!=dim:
            print 'Wrong argument for dimension'
            return None
         info['size'] = args
      for m in re.finditer('inout',cmds):
         idx = m.end(0)
         while cmds[idx]!='(':
            idx += 1
         argsbegin = idx
         args = cmds[argsbegin:findPairChars(cmds,argsbegin+1,'(',')')]
         args = [arg.strip() for arg in args.strip()[1:-1].split(',')]
         print args
         if len(args)!=2:
            print 'Wrong argument for inout'
            return None
         info['in'] = args[0]
         info['out'] = args[1]
      info['iterations'] = 1
      for m in re.finditer('iterations',cmds):
         idx = m.end(0)
         while cmds[idx]!='(':
            idx += 1
         argsbegin = idx
         args = cmds[argsbegin:findPairChars(cmds,argsbegin+1,'(',')')]
         args = [arg.strip() for arg in args.strip()[1:-1].split(',')]
         print args
         if len(args)!=1:
            print 'Wrong argument for iterations'
            return None
         info['iterations'] = args[0]
      info['device'] = 'cpu'
      for m in re.finditer('device',cmds):
         idx = m.end(0)
         while cmds[idx]!='(':
            idx += 1
         argsbegin = idx
         args = cmds[argsbegin:findPairChars(cmds,argsbegin+1,'(',')')]
         args = [arg.strip() for arg in args.strip()[1:-1].split(',')]
         print args
         if len(args)!=1:
            print 'Wrong argument for iterations'
            return None
         info['device'] = args[0]
      return info
   return None

def parseExtraArgs(args, pinfo):
   extraArgs = []
   idx = 0
   for arg in args:
      if arg==pinfo['in']:
         #print 'arg:',idx,arg,'input array'
         pass
      elif arg==pinfo['out']:
         #print 'arg:',idx,arg,'output array'
         pass
      elif arg in pinfo['size']:
         #print 'arg:',idx,arg,'array dimension'
         pass
      elif arg==pinfo['iterations']:
         #print 'arg:',idx,arg,'stencil iterations'
         pass
      else:
         #print 'arg:',idx,arg,'extra stencil argument'
         extraArgs.append( (arg,idx) )
      idx += 1
   return extraArgs

def main():
   filename = sys.argv[1]
   source = open(filename).read()
   commentFilter = Suppress( javaStyleComment ).ignore( dblQuotedString )
   source = commentFilter.transformString(source)
   
   source = source.replace('\r\n', '\n')
   source = source.replace('\r', '')
   
   for m in re.finditer('#pragma[ \t][ \t]*pskel.*\n', source):
      pragma = m.group(0)
      begin = m.start(0)
      end = m.end(0)+1
      while pragma.strip()[-1]=='\\':
         pragma = pragma.strip()[:-1].strip()
         tmp = ''
         while source[end]!='\n':
            tmp += source[end]
            end += 1
         pragma += ' '+tmp.strip()
         end += 1
      pragma = pragma.strip()
      print begin, end
      print pragma
      print source[begin:end]
      pinfo = parsePragma(pragma)
      print pinfo
      fcall = readFuncCall(source,end)
      print fcall
      extraArgs = parseExtraArgs(fcall['args'],pinfo)
      print 'ExtraArgs:',extraArgs
      if len(extraArgs)>0:
         print 'TODO: create a struct for passing the extra arguments to the PSkel stencil kernel'
         print 'gather the types of the extra arguments from the original kernel function declaration'

      kfunc = findFunctionDef(source,fcall['name'])
      
      print fcall['src']
      mainf = readOutterFunction(source, begin)
      #print mainf
      fcall_name = '_pskelcc_stencil_'+str(fcall['begin'])+'_'+str(fcall['end'])
      tmainf = mainf['src'].replace(source[begin:end],'\n').replace(fcall['src'],fcall_name+'();')
      print tmainf
      parser = c_parser.CParser()
      ast = parser.parse(tmainf)
      pgen = PSkelGenerator(pinfo,fcall_name)
      new_mainf = pgen.visit(ast)
      receivedParameters = pgen.receivedParameters
         
      parser = c_parser.CParser()
      ast = parser.parse(kfunc['src'])
      pgen = PSkelKernelGenerator(pinfo)
      new_kfunc = pgen.visit(ast)
      
      src = source.replace(mainf['src'],new_mainf).replace(kfunc['src'],new_kfunc)

      while len(receivedParameters)>0:
         print 'TODO: Trace arguments back to original definition',pgen.receivedParameters
         print 'Searching for caller of function',mainf['name']
         #print findFunctionCall(src,mainf['name'])
         receivedParameters = []
         #src = src.replace(...)

      f = open(sys.argv[2],'w')
      
      f.write('#define PSKEL_OMP\n')
      if pinfo['device']=='gpu':
         f.write('#define PSKEL_CUDA\n')
      f.write('''
#include "PSkel.h"
using namespace PSkel;\n\n''')
      f.write(src)
      f.close()
      
main()
