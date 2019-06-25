import linear_combination.linear_combination as lc
import operator
###### import linear_combination.words as words
###### import sympy
###### import scipy as sp
import numpy as np
from . import divided_powers as dp # XXX ???
from functools import reduce

def id(x):
    return x


def terminal_values(s):
    return lc.LinearCombination( {v:c[-1] for v,c in s.items()} )


def signature(x, upto_level):
    # TODO Using Chen might be faster.
    ONE = dp.CompositionConcatenation()
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x[np.newaxis,:]

    dim, N = x.shape
    x = np.vstack( tuple(x[i] - x[i,0] for i in range(dim)) )
    prevz = [ ONE ]
    sig = lc.LinearCombination( { ONE : np.ones( N ) } )

    for level in range(1, upto_level+1):
        currentz = []
        for ell in range(0,dim):
            for prev in prevz:
                if not prev == ONE:
                    concat_v = dp.CompositionConcatenation( tuple( p for p in prev ) + (dp.Monomial({ell:1}),) )
                    concat_s = np.hstack( (np.array( [0.] ), np.cumsum( sig[prev][0:-1] * ( x[ell][1:] - x[ell][0:-1] ) ) ) )
                    currentz.append( concat_v )
                    sig[ concat_v ] = concat_s
                if prev == ONE:
                    last_monomial = dp.Monomial()
                else:
                    last_monomial = dp.Monomial( prev[-1] )
                dp.safe_add( last_monomial, ell, +1 )
                bullet_v = dp.CompositionConcatenation( tuple( p for p in prev[0:-1] ) + (last_monomial, ) )
                bullet_s = np.hstack( (np.array( [0.] ), np.cumsum( sig[prev][1:] * ( x[ell][1:] - x[ell][0:-1] )) ) )
                if not prev == ONE:
                    bullet_s = bullet_s - concat_s
                currentz.append( bullet_v )
                sig[ bullet_v ] = bullet_s
        prevz = currentz
    return sig

###### from sympy import Rational
###### import functools
###### import signature_invariants.invariants as invariants
###### import itertools
###### from functools import reduce
###### 
###### import seaborn as sns
###### import matplotlib.pyplot as plt
###### 
###### # TODO
###### # * Tex nice table like in areas-of-areas.
###### # * Eulerian idempotent / coordinates of the first kind.
###### # * Do Lyndon words shuffle generate?
###### 
###### # TODO
###### # * Signature -> EXP^* .. ?=? stratonovich signature
###### # * usual exp and log
###### # * tex
###### # * \sum w \otimes w
###### 
###### # TODAY
###### # * primitive elements (Thm 4.2)
###### # * quasi-shuffle identity
###### # * Chen
###### # * Euler scheme something something?
###### #
###### # * signature as fixpoint
###### #
###### #     DS = e + DS > x + DS \bullet x
###### 
###### # DS = e
###### # DS = e + e > 1 + e \bullet 1 = e + 1
###### # DS = e + (e + 1) > 1 + (e+1) \bullet 1
###### #    = e + 1 + 11 + 2
###### # DS = e + (e+1+11+2) > 1 + (e+1+11+2) \bullet 1
###### #    = e + 1 + 11 + 111 + 21 + 2 + 12 + 3
###### 
###### # PUT SOMEWHERE:
###### #   - DTW(X,Y) = 0 => DS(X) = DS(Y)
###### #   - |DS_{N,n}(X) - DS_{N,n}(Y)| \le C(N,n) * DTW(X,Y) ?
###### 
###### 
###### 
###### 
###### ###################
###### ## HALL BASIS #####
###### ###################
###### 
###### def foliage(x):
######     """Flattens a tuple of tuples, i.e. gives a list of the leaves of the tree."""
######     if isinstance(x, dp.Monomial):
######         yield x
######         return
######     assert type(x) in [tuple,]
######     for i in x:
######         for j in foliage(i):
######             yield j
###### 
###### def less_expression_lyndon(a,b):
######     ta = tuple(foliage(a))
######     tb = tuple(foliage(b))
######     #print('ta=', ta)
######     #print('tb=', tb)
######     return ta < tb
###### 
###### class HallBasis: # modified from words.py
######     # TODO use BinaryPlanarTree
######     def __init__(self, variables, upto_level, less_expression=less_expression_lyndon):
######         #assert dim>1
######         #assert upto_level>0
######         self.variables = variables
######         self.dim = len(variables)
######         self.upto_level = upto_level
######         ms = [ dp.Monomial(m) for m in monomials( variables, upto_level ) ]
######         ms_by_degree = {}
######         for m in ms:
######             deg = m.deg()
######             if deg in ms_by_degree:
######                 ms_by_degree[deg].append( m )
######             else:
######                 ms_by_degree[deg] = [m]
######         out = [ [(m,) for m in ms_by_degree[1]] ]
###### 
######         for current_level in range(2,upto_level+1):
######             out.append( [(m,) for m in ms_by_degree[current_level]] )
######             for first_level in range(1,current_level):
######                 for x in out[first_level-1]:
######                     for y in out[current_level-first_level-1]:
######                         if less_expression(x,y) and (len(x)==1 or not less_expression(x[1],y)):
######                             out[-1].append((x,y))
######         self.data = out
######         self.less = less_expression
###### 
######     def find_as_foliage_of_hall_word(self, w): # XXX name
######         assert type(w) in (tuple,), w
######         assert 0<len(w)<=self.upto_level
######         for i in self.data[dp.CompositionConcatenation(w).deg()-1]: # XXX
######             if w == tuple(foliage(i)):
######                 return i
######         return None
######         
######     def factor_into_hall_words(self,w):
######         assert type(w) in (tuple,), w
######         assert 0 < len(w) <= self.upto_level
######         l = len(w)
######         if l==1:
######             #assert 1<=w[0]<=self.dim, str(w[0])+" is not in my alphabet" XXX
######             return [w]
######         best = (w[-1],)
######         best_prefix_length = l-1
######         for prefix_length in range(0,l-1):
######             end = w[prefix_length:]
######             endH = self.find_as_foliage_of_hall_word(end)
######             #print('endH=', endH, type(endH))
######             #print('best=', best, type(best))
######             #if endH is not None:
######             #    print('self.less(endH,best)=', self.less(endH, best))
######             if endH is not None and self.less(endH,best):
######                 best = endH
######                 #print('best=', best, type(best))
######                 best_prefix_length = prefix_length
######         if best_prefix_length == 0:
######             return [best]
######         return self.factor_into_hall_words(w[:best_prefix_length])+[best]
###### 
###### def test_hall_basis():
######     # Melancon, Reutenauer - Lyndon words, free algebras and shuffles
###### 
######     # The PBW and dual PBW in the SHUFFLE algebra. 
######     hb = HallBasis(['a','b'], 3)
###### 
###### 
######     assert not dp.Monomial({'a':2}) < dp.Monomial({'a':1})
######     assert dp.Monomial({'a':1}) < dp.Monomial({'a':2})
######     assert dp.Monomial({'a':1}) < dp.Monomial({'b':1})
######     assert not dp.Monomial({'b':1}) < dp.Monomial({'a':1})
######     assert dp.Monomial({'b':1}) < dp.Monomial({'a':2})
######     assert not dp.Monomial({'a':2}) < dp.Monomial({'b':1})
###### 
######     assert not less_expression_lyndon( ( dp.Monomial({'a':2}), dp.Monomial({'b':1}) ),  ( dp.Monomial({'b':1}), ) )
######     assert less_expression_lyndon( ( dp.Monomial({'b':1}), ), ( dp.Monomial({'a':2}), dp.Monomial({'b':1}) ) )
###### 
######     # XXX Careful !!!
######     assert dp.CompositionShuffle( (77,) ) == dp.CompositionQuasiShuffle( (77,) ) == dp.CompositionConcatenation( (77,) )
###### 
######     for b in hb.data:
######         for t in b:
######             for bb in hb.data:
######                 for tt in bb:
######                     w = tuple(foliage( t ))
######                     ww = tuple(foliage( tt ))
######                     dw = dual_PBW(w,hb)
######                     pww = primal_PBW(ww,hb)
######                     if w == ww:
######                         assert 1 == lc.LinearCombination.inner_product( dw, pww )
######                         assert 1 == lc.LinearCombination.inner_product( dp.hoffman_EXP(dw), dp.hoffman_LOG_dual(pww) )
######                     else:
######                         assert 0 == lc.LinearCombination.inner_product( dw, pww )
######                         assert 0 == lc.LinearCombination.inner_product( dp.hoffman_EXP(dw), dp.hoffman_LOG_dual(pww) )
###### 
###### def _lie_bracket_of_expression( expression ): # With generator now. This _is_ faster.
######     """Lie bracket of an expression like [ [1],[[1],[2]] ]."""
######     def _lie_bracket(cw1,cw2):
######         for prev in cw1*cw2:
######             yield prev
######         for prev in cw2*cw1:
######             yield (prev[0], -prev[1])
######     if len(expression) == 1:
######         yield from M_concat( *expression ).items()
######     else:
######         for x1, c1 in _lie_bracket_of_expression(expression[0]):
######             for x2, c2 in _lie_bracket_of_expression(expression[1]):
######                 for x, c in _lie_bracket(x1,x2):
######                     yield (x, c1*c2*c)
###### 
###### def primal_PBW(w, basis): # XXX This should not be here. The only difference is the implementation of HallBasis ..
######     """Primal PBW basis element for DESHUFFLE algebra."""
######     assert isinstance(basis, HallBasis), basis
######     assert type(w) in (list,tuple), w # XXX
######     if 0 == len(w):
######         return unitElt
######     #assert 0<len(w)<=basis.upto_level XXX
######     a = basis.factor_into_hall_words(w)
######     return functools.reduce( operator.mul, map( lambda x: lc.LinearCombination.from_generator(_lie_bracket_of_expression(x)), a ) )
###### 
###### def _concatenation_product_shuffle_word(self,other):
######     """Concatenation product."""
######     yield (dp.CompositionShuffle(self+other),1)
###### 
###### 
###### shuffle_unit = lc.lift( dp.CompositionShuffle() )
###### def dual_PBW(w, basis): # XXX This should not be here. The only difference is the implementation of HallBasis ..
######     """Dual PBW basis element for DESHUFFLE algebra."""
######     assert isinstance(basis, HallBasis), basis
######     assert type(w) in (tuple,), w
######     #assert len(w)<=basis.upto_level XXX
######     if len(w)==0:
######         return shuffle_unit
######     a = basis.factor_into_hall_words(w)
######     if len(a) == 1:
######         return lc.LinearCombination.apply_bilinear_function( _concatenation_product_shuffle_word,\
######                 lc.lift( dp.CompositionShuffle( (w[0],) ) ), dual_PBW(w[1:], basis) )
######     factor = 1.
######     out = shuffle_unit
######     for i,j in itertools.groupby(a):
######         word = tuple(foliage(i))
######         num = len(tuple(j))
######         factor *= sympy.factorial( num )
######         base = dual_PBW(word,basis)
######         power = functools.reduce(operator.mul,(base for _ in range(num)))
######         out = out * power
######     out = out * sympy.Rational(1,factor)
######     return out
###### 
###### 
###### ###########################################################
###### ###########################################################
######                 
###### def Pstar_m(m):
######     """
######     (2.8) in Malvenuto/Reutenauer.
###### 
######     Parameters
######     ----------
######         m : Monomial
######     """
######     ret = lc.LinearCombination()
######     for C in dp.finer(dp.CompositionConcatenation( (m,) )):
######         ret += sympy.Rational( (-1)**(len(C)+1), len(C) ) * lc.lift(C)
######     return ret
###### 
###### def Pstar_D(D):
######     """Another basis for QSym^* (NOT the PBW basis).
######        See p. 972 in Malvenuto/Reutenauer."""
######     return reduce(operator.mul, map(Pstar_m, D),lc.lift(dp.CompositionConcatenation()))
###### 
###### def P_C(C):
######     """The dual basis to P^*, see (2.12) in Malvenuto/Reutenauer."""
######     ret = lc.LinearCombination()
######     for D in dp.coarser(C):
######         ret += sympy.Rational(1, dp.f(C,D)) * lc.lift(D)
######     return ret
###### 
###### def experiment_malvenuto_reutenauer():
######     assert M_qs({'a':1}) * M_qs({'b':1}) == M_qs({'a':1},{'b':1}) + M_qs({'a':1,'b':1}) + M_qs({'b':1},{'a':1}) # Quasi-shuffle.
###### 
######     def log(s,N):
######         """See (2.8) in Malvenuto/Reutenauer."""
######         x = s - M_concat({})
######         xn = x
######         ret = x
######         for n in range(2,N+1):
######             xn = dp.project_smaller_equal(xn*x,N)
######             ret += sympy.Rational( (-1)**(n+1), n ) * xn
######         return ret
###### 
######     def to_sympy_monomial(d):
######         return reduce(operator.mul, map(lambda i: i[0]**i[1], d.items()))
###### 
######     def power_series_2_8(letters, up_to_degree):
######         """See (2.8) in Malvenuto/Reutenauer."""
######         return M_concat({}) +\
######                  reduce(operator.add,\
######                     map( lambda t: t[0] * t[1],\
######                          zip(map(M_concat,monomials(letters,up_to_degree)), map(to_sympy_monomial, monomials(sympy_vars, up_to_degree))) )) 
###### 
######     def is_number(s):
######         try:
######             float(s)
######             return True
######         except TypeError:
######             return False
###### 
######     N = 3
######     letters = ['a','b','c']
######     sympy_vars = list(map(sympy.symbols,letters))
###### 
######     # They are in duality:
######     for C in compositions(letters, 2):
######         p1 = P_C(C)
######         pstar1 = Pstar_D(C)
######         for C2 in compositions(letters, 2):
######             p2 = P_C(C2)
######             pstar2 = Pstar_D(C2)
######             if C == C2:
######                 assert 1 ==  dp.inner_product_qs_c( p1, pstar1 )
######             else:
######                 assert 0 ==  dp.inner_product_qs_c( p1, pstar2 ) 
######                 assert 0 ==  dp.inner_product_qs_c( p2, pstar1 )
###### 
######     ms = list(map(to_sympy_monomial, monomials(sympy_vars, N)))
######     primitive_candidates = {}
######     for x, c in log(power_series_2_8(letters, N),N).items():
######         for m in ms:
######             if c.coeff( m ) != 0 and is_number(c.coeff(m)):
######                 dp.safe_add(primitive_candidates, m, c.coeff(m) * lc.lift(x))
###### 
######     for m, D in zip(ms, [ dp.CompositionConcatenation( ( dp.Monomial(m), ) ) for m in monomials(letters, N) ] ):
######         # The procedure of (2.8) seems to work.
######         assert dp.is_primitive( primitive_candidates[m] )
######         assert primitive_candidates[m] == Pstar_D(D)
###### 
###### 
###### def experiment_hoffman():
######     letters = ['a','b'] 
######     for C1 in compositions(letters, 3):
######         for C2 in compositions(letters, 3):
######             # Malvenuto/Reutenauer is a special case of Hoffman:
######             assert P_C( C1 ) == dp.hoffman_EXP( lc.lift( C1 ) )
######             assert Pstar_D( C1 ) == dp.hoffman_LOG_dual( lc.lift( C1 ) )
###### 
######             # Duality:
######             assert lc.LinearCombination.inner_product( dp.hoffman_EXP( M_sh(*C1) ), M_concat(*C2) )\
######                    ==\
######                    lc.LinearCombination.inner_product( M_sh(*C1), dp.hoffman_EXP_dual(M_concat(*C2)) )
######             assert lc.LinearCombination.inner_product( dp.hoffman_LOG( M_sh(*C1) ), M_concat(*C2) )\
######                    ==\
######                    lc.LinearCombination.inner_product( M_sh(*C1), dp.hoffman_LOG_dual(M_concat(*C2)) )
###### 
######             # Algebra morphism:
######             assert dp.hoffman_EXP( M_sh(*C1) * M_sh(*C2) )\
######                    == dp.hoffman_EXP( M_sh(*C1) ) * dp.hoffman_EXP( M_sh(*C2) ) 
###### 
######             assert dp.hoffman_EXP_dual( M_concat(*C1) * M_concat(*C2) )\
######                    == dp.hoffman_EXP_dual( M_concat(*C1) ) * dp.hoffman_EXP_dual( M_concat(*C2) ) 
###### 
###### 
######             # Coalgebra morphism:
######             assert M_sh(*C1)\
######                     .apply_linear_function( dp.CompositionShuffle.coproduct )\
######                     .apply_linear_function(lc.Tensor.fn_otimes_linear(dp._hoffman_EXP, dp._hoffman_EXP)) \
######                   == dp.hoffman_EXP( M_sh(*C1) )\
######                     .apply_linear_function( dp.CompositionQuasiShuffle.coproduct )
######             assert M_qs(*C1)\
######                     .apply_linear_function( dp.CompositionQuasiShuffle.coproduct )\
######                     .apply_linear_function(lc.Tensor.fn_otimes_linear(dp._hoffman_LOG, dp._hoffman_LOG)) \
######                   == dp.hoffman_LOG( M_qs(*C1) )\
######                     .apply_linear_function( dp.CompositionShuffle.coproduct )
######             #assert M_concat(*C1)\
######             #        .apply_linear_function( dp.CompositionConcatenation.coproduct )\
######             #        .apply_linear_function(lc.Tensor.fn_otimes_linear(_hoffman_EXP_dual, dp._hoffman_EXP_dual)) \
######             #      == dp.hoffman_EXP( M_sh(*C1) )\
######             #        .apply_linear_function( dp.CompositionQuasiShuffle.coproduct )
###### 
######             # Hopf algebra morphism:
######             assert dp.hoffman_EXP( M_sh(*C1) ).apply_linear_function( dp.CompositionQuasiShuffle.antipode )\
######                    == dp.hoffman_EXP( M_sh(*C1).apply_linear_function( dp.CompositionShuffle.antipode ) )
###### 
######             # Bijection:
######             assert M_sh(*C1) == dp.hoffman_LOG( dp.hoffman_EXP( M_sh(*C1) ) )
###### 
###### def test_antipode():
######     # \Nabla (id \otimes A) \Delta = \unit \circ \counit.
###### 
######     for C in compositions(['a','b'], 3):
######         x = lc.lift(C)
######         assert lc.LinearCombination() \
######                 == lc.lift(C)\
######                      .apply_linear_function(dp.CompositionConcatenation.coproduct)\
######                      .apply_linear_function(lc.Tensor.fn_otimes_linear(lc.id,dp.CompositionConcatenation.antipode))\
######                      .apply_linear_function(lc.Tensor.m12 )
######     assert lc.lift(dp.CompositionConcatenation()) \
######             == lc.lift(dp.CompositionConcatenation())\
######                 .apply_linear_function(dp.CompositionConcatenation.coproduct)\
######                 .apply_linear_function(lc.Tensor.fn_otimes_linear(lc.id,dp.CompositionConcatenation.antipode))\
######                 .apply_linear_function(lc.Tensor.m12)
###### 
######     for C in compositions(['a','b'], 3):
######         x = lc.lift(dp.CompositionQuasiShuffle(C))
######         assert lc.LinearCombination() \
######                 == lc.lift(dp.CompositionQuasiShuffle(C))\
######                      .apply_linear_function(dp.CompositionQuasiShuffle.coproduct)\
######                      .apply_linear_function(lc.Tensor.fn_otimes_linear(lc.id,dp.CompositionQuasiShuffle.antipode))\
######                      .apply_linear_function(lc.Tensor.m12 )
######     assert lc.lift(dp.CompositionQuasiShuffle()) \
######             == lc.lift(dp.CompositionQuasiShuffle())\
######                 .apply_linear_function(dp.CompositionQuasiShuffle.coproduct)\
######                 .apply_linear_function(lc.Tensor.fn_otimes_linear(lc.id,dp.CompositionQuasiShuffle.antipode))\
######                 .apply_linear_function(lc.Tensor.m12)
###### 
###### 
###### 
###### 
###### 
###### 
###### 
###### def conjecture_coordinates_of_the_first_kind():
######     """Conjecture: Just use pi_1^\\bot.
######        OPEN (but seems, and should, be true)."""
######     upto_level = 3
######     #hb = HallBasis(['1'], upto_level)
######     hb = HallBasis(['1','2'], upto_level)
######     Lsh = lc.LinearCombination()
######     Lqs = lc.LinearCombination()
######     duals_sh = []
######     duals_qs = []
######     coefficients = []
######     for b in hb.data:
######         for t in b:
######             w = tuple(foliage( t ))
######             lw = lc.lift( dp.CompositionShuffle(w) )
######             dw = dual_PBW(w,hb)
######             pww = primal_PBW(w,hb)
######             s = str(''.join(map(str,w)).replace('^','^'))
######             a = sympy.symbols('p_{'+s+'}')
######             coefficients.append( a )
######             Lqs += a * dp.hoffman_LOG_dual(pww)
######             Lsh += a * pww
######             duals_qs.append( dp.hoffman_EXP(dw) )
######             duals_sh.append( dw )
###### 
######     g = dp.exp(Lqs,upto_level)
######     for a, dw  in zip(coefficients,duals_qs):
######         print('a=', a)
######         print('dw=', dw)
######         print('..(dw)=', dp.pi1_adjoint(dw,upto_level))
######         assert a == sympy.simplify( lc.LinearCombination.inner_product( dp.pi1_adjoint(dw,upto_level), g ) )
###### 
######     # Classical case as sanity check.
######     g = dp.exp(Lsh,upto_level)
######     for a, dw in zip(coefficients, duals_sh):
######         assert a == sympy.simplify( lc.LinearCombination.inner_product( dp.pi1_adjoint(dw,upto_level), g ) )
###### 
###### 
###### def for_paper():
######     letters = ['a','b'] 
######     for C1 in compositions(letters, 3):
######         phi = M_sh( *C1 )
######         print()
######         print()
######         print('phi=', phi)
######         print('EXP phi=', dp.hoffman_EXP(phi))
###### 
######         x = M_concat( *C1 )
######         print('x=', x)
######         print('LOG^* x=', dp.hoffman_LOG_dual(x))
###### 
######     #hb = HallBasis(['\\letter1','\\letter2'], 4)
######     hb = HallBasis(['\\letter1'], 3)
######     print()
######     print()
######     for b in hb.data:
######         for t in b:
######             w = tuple(foliage( t ))
######             lw = lc.lift( dp.CompositionShuffle(w) )
######             dw = dual_PBW(w,hb)
######             pww = primal_PBW(w,hb)
######             print()
######             print()
######             print('w=', '|'.join(map(str,w)))
######             print('SHUFFLE PBW=', pww)
######             print('SHUFFLE dual PBW=', dw)
######             print('pi_1^\\top SHUFFLE dual PBW=', dp.pi1_adjoint(dw,3))
######             print('QUASI-SHUFFLE PBW=', dp.hoffman_LOG_dual(pww))
######             print('QUASI-SHUFFLE dual PBW=', dp.hoffman_EXP(dw))
######             #if lw != dw:
######             #    print('$'+''.join(map(str,w))+'$ & $' + str(pww) + '$ & $' + str(dw) + '$ & $' + str(dp.pi1_adjoint(dw,3)) + '$ \\\\')
######             #print('$'+''.join(map(str,w))+'$ & $' + str(pww) + '$ & $' + str(dw) + '$ & $' + str(dp.pi1_adjoint(dw,3)) + '$ \\\\')
###### 
######             #print()
######             #print( dp.hoffman_EXP( lw ) )
######             #print( dp.hoffman_LOG_dual(pww) )
######             #print( dp.hoffman_EXP( dw ) )
######             #print('$('+','.join(map(str,w))+')$ & $' + str(dp.hoffman_LOG_dual(pww)) + '$ & $' + str(dp.hoffman_EXP(dw)) + '$ & $..' + '$ \\\\')
###### 
######             #print('$('+','.join(map(str,w))+')$ & $' + str(pww) + '$ & $' + str(dw) + '$ & $' + str(dp.pi1_adjoint(dw,3)) + '$ \\\\')
###### 
###### def test_dynkin():
######     def N( C ):
######         yield (C, C.deg())
###### 
######     def dynkin_dequasishuffle(x):
######         # TODO Compare to right-bracketing.
######         return x.apply_linear_function( dp.CompositionConcatenation.coproduct )\
######                 .apply_linear_function( lc.Tensor.fn_otimes_linear( N, dp.CompositionConcatenation.antipode ) )\
######                 .apply_linear_function( lc.Tensor.m12 )
######     def N_deshuffle( C ):
######         yield (C, len(C))
###### 
######     def dynkin_deshuffle(x):
######         # TODO Compare to right-bracketing.
######         # XXX what N ???
######         return x.apply_linear_function( dp.CompositionConcatenation.coproduct_deshuffle )\
######                 .apply_linear_function( lc.Tensor.fn_otimes_linear( N_deshuffle, dp.CompositionConcatenation.antipode_deshuffle ) )\
######                 .apply_linear_function( lc.Tensor.m12 )
###### 
######     #print( dynkin_dequasishuffle( M_concat({0:1}) ) )
######     #print( dynkin_dequasishuffle( M_concat({0:1}, {0:1}) ) )
######     #print( dynkin_dequasishuffle( M_concat({0:1}, {1:1}) ) )
######     #print( dp.is_primitive( dynkin_dequasishuffle( M_concat({0:2}) ) ) )
######     letters = ['a','b'] 
######     #x = M_concat({'a':1},{'a':1},{'b':1})
######     #print( dynkin_dequasishuffle(x) )
###### 
######     for C in compositions(letters, 3):
######         x = lc.lift( C )
######         #print( dynkin_deshuffle( x ) )
###### 
######         # TODO CONTINUE HERE what is going on?
######         #print( x.apply_linear_function( dp.CompositionConcatenation.coproduct_deshuffle ) )
######         #print( x.apply_linear_function( dp.CompositionConcatenation.coproduct_deshuffle )\
######         #        .apply_linear_function( lc.Tensor.fn_otimes_linear( N_deshuffle, dp.CompositionConcatenation.antipode_deshuffle ) ) )
######         #print( x.apply_linear_function( dp.CompositionConcatenation.coproduct_deshuffle )\
######         #        .apply_linear_function( lc.Tensor.fn_otimes_linear( N_deshuffle, dp.CompositionConcatenation.antipode_deshuffle ) )\
######         #        .apply_linear_function( lc.Tensor.m12 ) )
######         #print( '0)=', dynkin_deshuffle(x) )
######         #print( '1)=', dp.hoffman_LOG_dual( dynkin_deshuffle(x) ) )
######         #print( '2)=', dynkin_dequasishuffle( dp.hoffman_LOG_dual(x) ) )
######         #print( '3)=', dp.hoffman_LOG_dual(x) )
######         #print( '4)=', dynkin_dequasishuffle( x ) )
######         if len(C) == 1:
######             print()
######             print( x )
######             print( '3)=', dp.hoffman_LOG_dual(x) )
######             print( '4)=', dynkin_dequasishuffle( x ) )
######             print( '5)=', C.deg() * dp.hoffman_LOG_dual(x) - dynkin_dequasishuffle( x ) )
###### 
######         #assert dp.hoffman_LOG_dual( dynkin_deshuffle(x) )\
######         #        == dynkin_dequasishuffle( dp.hoffman_LOG_dual(x) )
######         # XXX the grading is OFF
###### 
######         #print( dynkin_dequasishuffle( x ) )
######         #print()
######         #print( x )
######         #print( dynkin_dequasishuffle( x ) )
######         #print( dynkin_dequasishuffle(dynkin_dequasishuffle( x )) )
######         #print( dp.hoffman_LOG_dual( x ) )
######         #if len(C) == 1:
######         #    print( dynkin_dequasishuffle( x ) - 3 *  dp.hoffman_LOG_dual( x ) )
######         #    assert dp.is_primitive( dynkin_dequasishuffle( x ) - 3 *  dp.hoffman_LOG_dual( x ) )
######         assert dynkin_dequasishuffle(dynkin_dequasishuffle( x )) == C.deg() * dynkin_dequasishuffle(x)
######         assert dp.is_primitive(dynkin_dequasishuffle( x ))
######     
###### def area(_x,_y):
######     return bi( dp.CompositionShuffle.half_shuffle, _x, _y )\
######            -\
######            bi( dp.CompositionShuffle.half_shuffle, _y, _x )
###### 
###### def bi(_f,_x,_y):
######     return lc.LinearCombination.apply_bilinear_function(_f,_x,_y)
###### 
###### def area_discrete(_x,_y):
######     return bi( dp.CompositionQuasiShuffle.half_shuffle, _x, _y )\
######            -\
######            bi( dp.CompositionQuasiShuffle.half_shuffle, _y, _x )
###### 
###### def experiment_area():
######     x = M_sh( {'a':1}, {'b':1} )
######     y = M_sh( {'c':1}, {'d':1} )
###### 
###### 
######     assert\
######         M_sh( {'a':1}, {'b':1}, {'c':1}, {'d':1} )\
######       + M_sh( {'a':1}, {'c':1}, {'b':1}, {'d':1} )\
######       + M_sh( {'a':1}, {'c':1}, {'d':1}, {'b':1} )\
######       + M_sh( {'c':1}, {'a':1}, {'b':1}, {'d':1} )\
######       + M_sh( {'c':1}, {'a':1}, {'d':1}, {'b':1} )\
######       + M_sh( {'c':1}, {'d':1}, {'a':1}, {'b':1} )\
######       == x * y
###### 
######     assert x * y\
######            == bi( dp.CompositionShuffle.half_shuffle, x, y )\
######             + bi( dp.CompositionShuffle.half_shuffle, y, x )
###### 
######     x = M_qs( {'a':1}, {'b':1} )
######     y = M_qs( {'c':1}, {'d':1} )
######     assert\
######         M_qs( {'a':1}, {'b':1}, {'c':1}, {'d':1} )\
######       + M_qs( {'a':1}, {'b':1 ,  'c':1}, {'d':1} )\
######       + M_qs( {'a':1}, {'c':1}, {'b':1}, {'d':1} )\
######       + M_qs( {'a':1 ,  'c':1}, {'b':1}, {'d':1} )\
######       + M_qs( {'a':1}, {'c':1}, {'b':1 ,  'd':1} )\
######       + M_qs( {'a':1 ,  'c':1}, {'b':1 ,  'd':1} )\
######       + M_qs( {'a':1}, {'c':1}, {'d':1}, {'b':1} )\
######       + M_qs( {'a':1 ,  'c':1}, {'d':1}, {'b':1} )\
######       + M_qs( {'c':1}, {'a':1}, {'b':1}, {'d':1} )\
######       + M_qs( {'c':1}, {'a':1}, {'b':1 ,  'd':1} )\
######       + M_qs( {'c':1}, {'a':1}, {'d':1}, {'b':1} )\
######       + M_qs( {'c':1}, {'d':1}, {'a':1}, {'b':1} )\
######       + M_qs( {'c':1}, {'d':1 ,  'a':1}, {'b':1} )\
######       == x * y
###### 
######     assert x * y == \
######              bi( dp.CompositionQuasiShuffle.half_shuffle, x, y )\
######            + bi( dp.CompositionQuasiShuffle.half_shuffle, y, x )\
######            + bi( dp.CompositionQuasiShuffle.bullet_last, y, x )
###### 
###### 
###### 
######     a_s = M_sh({'a':1})
######     b_s = M_sh({'b':1})
######     c_s = M_sh({'c':1})
###### 
######     print( M_sh({'a':1},{'b':1},{'c':1}) - M_sh({'a':1},{'c':1},{'b':1}) )
######     print( dp.hoffman_EXP( M_sh({'a':1},{'b':1},{'c':1}) - M_sh({'a':1},{'c':1},{'b':1}) ) )
######     xx
###### 
######     a_q = M_qs({'a':1})
######     b_q = M_qs({'b':1})
######     c_q = M_qs({'c':1})
######     print(area_discrete(a_q,b_q))
######     print(area_discrete(area_discrete(a_q,b_q),c_q))
######     xx
###### 
######     # The Hoffman exponential is nicely compatible
######     # with the area-operation.
######     assert dp.hoffman_EXP( area(area(a_s,b_s), c_s) )\
######            == area_discrete( area_discrete(a_q,b_q), c_q )
######     # TODO do more trees
######     a = M_sh( {'1':1} )
######     b = M_sh( {'2':1} )
######     c = M_sh( {'3':1} )
######     d = M_sh( {'4':1} )
###### 
######     def qhs(_x,_y):
######         return bi( dp.CompositionQuasiShuffle.half_shuffle, _x, _y )
######     def hs(_x,_y):
######         return bi( dp.CompositionShuffle.half_shuffle, _x, _y )
###### 
######     #print( (hs(a,b) - hs(b,a)) )
######     #print( (hs(a,b) - hs(b,a)) * c )
######     #xx
######     phi = hs( (hs(a,b) - hs(b,a)) * c, d )\
######         - hs( (hs(a,b) - hs(b,a)) * d, c )\
######         - hs( (hs(c,d) - hs(d,c)) * a, b )\
######         + hs( (hs(c,d) - hs(d,c)) * b, a )
###### 
######     #print( phi )
######     #print( dp.hoffman_EXP( phi ) )
######  
######     a_q = M_qs( {'1':1} )
######     b_q = M_qs( {'2':1} )
######     c_q = M_qs( {'3':1} )
######     d_q = M_qs( {'4':1} )
###### 
###### 
######     psi = qhs( (qhs(a_q,b_q) - qhs(b_q,a_q)) * c_q, d_q )\
######         - qhs( (qhs(a_q,b_q) - qhs(b_q,a_q)) * d_q, c_q )\
######         - qhs( (qhs(c_q,d_q) - qhs(d_q,c_q)) * a_q, b_q )\
######         + qhs( (qhs(c_q,d_q) - qhs(d_q,c_q)) * b_q, a_q )
######     
######     print( dp.hoffman_EXP(phi) - psi )
###### 
######     phi2 = hs( hs(a,b) - hs(b,a), hs(c,d) - hs(d,c) )\
######          - hs( hs(c,d) - hs(d,c), hs(a,b) - hs(b,a) )
###### 
######     psi2 = qhs( qhs(a_q,b_q) - qhs(b_q,a_q), qhs(c_q,d_q) - qhs(d_q,c_q) )\
######          - qhs( qhs(c_q,d_q) - qhs(d_q,c_q), qhs(a_q,b_q) - qhs(b_q,a_q) )
###### 
######     print( dp.hoffman_EXP(phi2) - psi2 )
###### 
###### 
###### def for_talk():
######     import sys
######     sys.path.append('/Users/joscha/Dropbox/12 - PostDoc/diehl-otter-waas/code')
######     import html_plotter
###### 
######     sys.path.append('/Users/joscha/Dropbox/12 - PostDoc/diehl-ebrahimi-pfeffer-tapia/code')
######     import OLD_conjectures as OC
###### 
######     X = [0.,0.5,1.0,.5,0.,-.5,-1.,-.5,.0]
###### 
######     M = len(X)
######     N = 15
######     i = 0
######     #for beta in OC.NM_time_changes(N, M):
######     plt.yticks([])
######     b = plt.bar( range(M), X )
######     #print(b[0].get_facecolor())
###### 
######     color = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
######     html_plotter.plot(plt)
###### 
######     np.random.seed(123456)
###### 
######     eps = 2.
###### 
######     def rw(_N):
######         return np.cumsum( [0] + [np.random.randn(1) for i in range(_N)] )
###### 
######     for i in [100, 1100, 2648]:
######         beta = list(OC.NM_time_changes(N, M))[i]
######         Y_beta = [ X[ beta[i] -1] for i in range(1,N+1) ]
######         plt.bar( range(N), Y_beta )
######         plt.yticks([])
######         html_plotter.plot(plt)
###### 
######         noise = rw(N)
######         Y_beta_noise = [ Y_beta[i] + eps * noise[i] for i in range(N) ]
######         plt.bar( range(N), Y_beta_noise )
######         plt.yticks([])
######         html_plotter.plot(plt)
###### 
######     # PICTURE FOR STANDING STILL
######     X = [0.,0.5,1.0,.5,0.,-.5,-1.,-.5,.0]
######     X_s = [0.,0.5,1.0,.5,.5,0.,-.5,-1.,-.5,.0]
######     plt.bar(range(len(X)), X)
######     html_plotter.plot(plt)
######     plt.bar(range(len(X_s)), X_s, color= [color] * 4 + [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, .5)] )
######     html_plotter.plot(plt)
###### 
######     diff_color = 'C3'
######     Y = [0.5,0.5,-.5,-.5,-.5,-.5,.5,.5]
######     Y_s = [0.5,0.5,-.5,0., -.5,-.5,-.5,.5,.5]
######     plt.bar(range(len(Y)), Y, color=diff_color)
######     html_plotter.plot(plt)
###### 
######     # (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0)
######     plt.bar(range(len(Y_s)), Y_s, color=diff_color)
######     html_plotter.plot(plt)
###### 
###### def experiment_multiparameter():
######     import sys
######     sys.path.append('/Users/joscha/Dropbox/12 - PostDoc/diehl-otter-waas/code')
######     import html_plotter
###### 
######     np.random.seed(123456)
###### 
######     I = np.random.randn(5,5)
###### 
######     cmap = None #'binary'
######     plt.imshow( I, cmap=cmap )
######     html_plotter.plot( plt )
###### 
######     def stand_still_x(n, I):
######         return np.hstack( ( I[:,0:n+1] , I[:,n:n+1] , I[:,n+1:] ) )
###### 
######     def stand_still_y(n, I):
######         return np.vstack( ( I[0:n+1,:] , I[n:n+1,:] , I[n+1:,:] ) )
###### 
######     print( I.shape )
######     print( stand_still_x(2, I).shape  )
######     print( I )
######     print( stand_still_x(2, I)  )
######     plt.imshow( stand_still_x(2, I), cmap=cmap )
######     html_plotter.plot( plt )
######     plt.imshow( stand_still_y(2, I), cmap=cmap )
######     html_plotter.plot( plt )
######     plt.imshow( stand_still_x( 2, stand_still_y(2, I)), cmap=cmap )
######     html_plotter.plot( plt )
######     plt.imshow( stand_still_y( 2, stand_still_x(2, I)), cmap=cmap )
######     html_plotter.plot( plt )
###### 
######     #    print(  )
######     #for beta in 
######     #    print()
######     #    print('i=', i)
######     #    i +=1
######     #    print('beta=', beta)
######     #    print( [X[ beta[i] -1] for i in range(1,N+1)] )
###### 
###### def conjecture_counting():
######     import collections 
######     t = sympy.symbols('t')
######     def series_coefficients(f, upto_level):
######         ret = []
######         ell = 0
######         while ell <= upto_level:
######             ret.append( f.subs({t:0}) / sympy.factorial(ell) )
######             f = f.diff(t)
######             ell += 1
######         return ret
###### 
######     upto_level = 5
######     for dim in [2,3,4,5]:
######         c = collections.Counter(list(map(dp.CompositionConcatenation.deg, compositions(range(dim),upto_level)) ))
######         H = (1-t)**dim / ( 2*(1-t)**dim - 1 )
######         assert series_coefficients(H, upto_level) == [1] + [c[i] for i in range(1,upto_level+1)]
###### 
###### 
###### def as_vectors(DSs, d, upto_level):
######     Cs = list(map(lambda C: M_qs(*C), compositions(range(d), upto_level)))
######     vs = []
######     for DS in DSs:
######         v = []
######         for C in Cs:
######             v.append(lc.LinearCombination.inner_product(C, DS))
######         vs.append(v)
######     return vs
###### 
###### 
###### def conjecture_linear_span():
######     """Conjecture: For long enough timeseries, the linear span of (projected) signatures is the whole (projected) space.
######        OPEN."""
###### 
######     upto_level = 3
######     N = upto_level + 1
######     paths = []
######     sigs = []
######     #np.random.seed(123124)
######     for i in range(300):
######         x = np.random.randint(-100,100, size=(1,N))
######         #x = np.random.randn(1,N)
######         DS = terminal_values(discrete_signature(x, upto_level))
######         sigs.append( DS )
######     vs = as_vectors(sigs, 1, upto_level)
######     _, sigmas, _ = np.linalg.svd(np.array(vs))
######     #print( vs )
######     print(len(vs[0]))
######     print( sigmas, len(sigmas) )
######     #print( np.linalg.matrix_rank(vs, tol=0.5) )
######     print( np.linalg.matrix_rank(vs) )
######     M = sympy.Matrix( vs )
######     M, y = M.rref()
######     print( M[0:len(vs[0]),:] )
######     print( y )
######     #print( words.rank( as_vectors(sigs, 1, upto_level) ) )
######     # TODO run on server
###### 
###### def misc():
######     aa = M_qs( {1:1}, {1:1} )
######     bb = M_qs( {2:1}, {2:1} )
######     print( aa * bb )
###### 
###### def conjecture_reverse_signature():
######     """Conjecture: DS( \overleftarrow x ) * DS( x ) is 'interesting'.
######        OPEN."""
######     N = 10
######     upto_level = 5
######     x = np.random.randint(-100,100, size=(1,N))
######     #x = np.random.randn(1,N)
######     overleftarrow_x = np.flip(x,axis=1)
######     print( x )
######     print( overleftarrow_x )
######     DS_x = terminal_values(discrete_signature(x, upto_level))
######     DS_overleftarrow_x = terminal_values(discrete_signature(overleftarrow_x, upto_level))
###### 
######     print()
######     print( DS_overleftarrow_x )
######     print()
######     print( DS_x )
######     print()
######     print( dp.project_smaller_equal( DS_overleftarrow_x * DS_x, upto_level) )
######     print()
######     print( dp.project_smaller_equal( DS_overleftarrow_x * DS_x, upto_level).apply_linear_function(dp.CompositionConcatenation.antipode) )
######     #print()
######     #print()
######     #print( dp.project_smaller_equal( DS_overleftarrow_x * DS_x.apply_linear_function(dp.CompositionConcatenation.antipode), upto_level) )
###### 
###### 
###### def one_dim_coordinates_of_the_first_kind():
######     upto_level = 9
######     #hb = HallBasis(['1'], upto_level)
######     hb = HallBasis([1], upto_level)
######     Lsh = lc.LinearCombination()
######     Lqs = lc.LinearCombination()
######     duals_sh = []
######     duals_qs = []
######     coefficients = []
###### 
######     filename = 'coordinates_of_the_first_kind.txt'
######     with open(filename, 'w') as fout:
######         for b in hb.data:
######             for t in b:
######                 w = tuple(foliage( t ))
######                 lw = lc.lift( dp.CompositionShuffle(w) )
######                 dw = dual_PBW(w,hb)
######                 pww = primal_PBW(w,hb)
######                 s = str(''.join(map(str,w)).replace('^','^'))
######                 a = sympy.symbols('p_{'+s+'}')
######                 coefficients.append( a )
######                 Lqs += a * dp.hoffman_LOG_dual(pww)
######                 Lsh += a * pww
######                 duals_qs.append( dp.hoffman_EXP(dw) )
######                 duals_sh.append( dw )
######                 def composition_shuffle_to_integer_composition(cs):
######                     #print('cs=', repr(cs))
######                     t = tuple(map( lambda d: d[1], cs ))
######                     #print('t=', t)
######                     yield t, +1
###### 
######                 #print( repr(dp.pi1_adjoint(dw,upto_level)) )
######                 adj = dp.pi1_adjoint(dw,upto_level).apply_linear_function( composition_shuffle_to_integer_composition )
######                 print('w=', w)
######                 print( repr(adj) )
######                 fout.write( repr(adj) + "\n" )
###### 
###### if __name__ == '__main__':
######     #conjecture_reverse_signature()
######     #misc()
######     #conjecture_linear_span()
######     #test_antipode()
######     #test_monomials()
######     #test_divided_powers()
######     #test_compositions()
######     #experiment_hoffman()
######     #conjecture_coordinates_of_the_first_kind()
######     one_dim_coordinates_of_the_first_kind()
######     #test_dynkin()
######     #for_paper()
######     #conjecture_counting()
######     #experiment_area()
######     #test_signature()
######     #test_f()
######     #test_hall_basis()
######     #test_pi1()
######     #experiment_malvenuto_reutenauer()
######     #experiment_super_catalan_numbers()
######     #for_talk()
######     #experiment_multidim()
