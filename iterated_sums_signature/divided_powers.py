from linear_combination import linear_combination as lc
import linear_combination.words as words
import pyparsing as pp
import sympy
import itertools
from sympy.utilities.iterables import multiset_permutations
from itertools import combinations_with_replacement
import numpy as np
import operator
import functools
from functools import reduce
import math


#####################
# CONVENIENCE METHODS
#####################
def M_concat(*args):
    return reduce(operator.mul,\
            map( lambda d: lc.lift(CompositionConcatenation((Monomial(d),))), args),\
            lc.lift(CompositionConcatenation()))

def M_qs(*args):
    return concatenation_to_quasi_shuffle( M_concat(*args) )

def M_sh(*args):
    return concatenation_to_shuffle( M_concat(*args) )

def compositions(variables, up_to_degree): # XXX move
    for m in monomials(variables, up_to_degree):
        for C in finer(CompositionConcatenation( ( Monomial(m), ) )):
            yield C

def monomials(variables, up_to_degree):
    for n in range(1,up_to_degree+1):
        for p in combinations_with_replacement( variables, n ):
            ret = {}
            for v in p:
                safe_add(ret, v, +1)
            yield ret

def safe_add(d,key,value): # from word_algebras.py
    if key in d:
        d[key] += value
    else:
        d[key] = value
#####################
# /CONVENIENCE METHODS
#####################


class Monomial(dict):

    def __mul__(self,other):
        """Monomial product."""
        yield (Monomial(lc.merge_with_add(self,other)),1)

    def remove_zeros(self):
        return Monomial( dict( filter( lambda x: not x[1] == 0, self.items() ) ) )

    def coproduct(self):
        """Divided power coproduct."""
        if len(self) == 0: # XXX
            yield (lc.Tensor( (Monomial(),Monomial()) ),1)
        else:
            itemz = list(self.items())
            varz = [i[0] for i in itemz]
            powers = [i[1] for i in itemz]
            for pp in itertools.product( *map(lambda p: range(0,p+1), powers) ):
                left = Monomial( { v:p for v,p in zip(varz, pp) } ).remove_zeros()
                right = Monomial( { v:p1-p2 for v,p1,p2 in zip(varz, powers, pp) } ).remove_zeros()
                yield (lc.Tensor( (left,right) ),1)

    def __hash__(self):
        return hash( (Monomial, frozenset(self.items())) ) # XXX

    #def __str__(self):
    #    if len(self) == 0:
    #        return 'e'
    #    else:
    #        def _s(v,p):
    #            if p == 1:
    #                return str(v)
    #            else:
    #                return str(v) + '^' + str(p)
    #        return ' '.join([ _s(v,p)  for v,p in self.items()])

    def __str__(self):
        if len(self) == 0:
            return 'e'
        #elif len(self) == 1 and list(self.items())[0][1] == 1:
        #    return str( list(self.items())[0][0] )
        else:
            def _s(v,p):
                if p == 1:
                    return str(v)
                else:
                    return str(v) + '^' + str(p)
            return '[' + ' '.join([ _s(v,p)  for v,p in self.items()]) + ']'

    def deg(self):
        return reduce(operator.add, [p for _, p in self.items()])

    def __lt__(self,other):
        def flatten(d):
            ret = []
            for x, p in d.items():
                ret += [x] * p
            return ret

        if self.deg() < other.deg():
            return True
        elif self.deg() > other.deg():
            return False
        return sorted(flatten(self)) < sorted(flatten(other))

class CompositionShuffle(tuple):

    def __add__(self,other):
        return CompositionShuffle( super(CompositionShuffle,self).__add__(other) )

    def __repr__(self):
        return 'CompositionShuffle[' + super(CompositionShuffle,self).__repr__() + ']'

    def __mul__(self,other):
        """Shuffle product.

           va * wb = (va * w) b + (v * wb) a"""

        if len(other) == 0:
            yield (self, 1)
        elif len(self) == 0:
            yield (other, 1)
        else:
            v, a = CompositionShuffle(self[:-1]), self[-1]
            w, b = CompositionShuffle(other[:-1]), other[-1]
            for x, c in v * other:
                yield (CompositionShuffle( x + (a,) ), c)
            for x, c in self * w:
                yield (CompositionShuffle( x + (b,) ), c)

    def half_shuffle(self, other):
        if len(other) == 0:
            yield (self, 1)
        elif len(self) == 0:
            yield (other, 1)
        else:
            w, b = CompositionShuffle(other[:-1]), other[-1]
            for x, c in self * w:
                yield (CompositionShuffle( x + (b,) ), c)

    def deg(self):
        if len(self) == 0:
            return 0
        else:
            return reduce(operator.add, map(Monomial.deg, self))

    def __str__(self):
        if len(self) == 0:
            return 'e'
        else:
            #return '(' + ', '.join(map(str,self)) + ')'
            return ''.join(map(str,self))

    def coproduct(self):
        """De-concatenation."""
        for i in range(len(self)+1):
            yield (lc.Tensor( (CompositionShuffle(self[:i]), CompositionShuffle(self[i:])) ), 1)

    def antipode(self):
        rev = CompositionShuffle( reversed(self) )
        yield (rev, (-1)**len(self))
        

class CompositionQuasiShuffle(tuple):

    def __add__(self,other):
        return CompositionQuasiShuffle( super(CompositionQuasiShuffle,self).__add__(other) )
    
    def __repr__(self):
        return 'CompositionQuasiShuffle[' + super(CompositionQuasiShuffle,self).__repr__() + ']'

    def __mul__(self,other):
        """Quasi shuffle product.

           va * wb = (va * w) b + (v * wb) * b + (v*w) (a \bullet b)"""
        if len(other) == 0:
            yield (self, 1)
        elif len(self) == 0:
            yield (other, 1)
        else:
            v, a = CompositionQuasiShuffle(self[:-1]), self[-1]
            w, b = CompositionQuasiShuffle(other[:-1]), other[-1]
            for x, c in v * other:
                yield (CompositionQuasiShuffle( x + (a,) ), c)
            for x, c in self * w:
                yield (CompositionQuasiShuffle( x + (b,) ), c)
            for x, c in v * w:
                bullet = list(a*b)[0][0] # XXX HACK
                yield (CompositionQuasiShuffle( x + (bullet,) ), c )

    def half_shuffle(self,other):
        if len(other) == 0:
            yield (self, 1)
        elif len(self) == 0:
            yield (other, 1)
        else:
            #v, a = CompositionQuasiShuffle(self[:-1]), self[-1]
            w, b = CompositionQuasiShuffle(other[:-1]), other[-1]
            #for x, c in v * other:
            #    yield (CompositionQuasiShuffle( x + (a,) ), c)
            for x, c in self * w:
                yield (CompositionQuasiShuffle( x + (b,) ), c)

    def bullet_last(self,other):
        if len(other) == 0:
            yield (self, 1)
        elif len(self) == 0:
            yield (other, 1)
        else:
            v, a = CompositionQuasiShuffle(self[:-1]), self[-1]
            w, b = CompositionQuasiShuffle(other[:-1]), other[-1]
            for x, c in v * w:
                bullet = list(a*b)[0][0] # XXX HACK
                yield (CompositionQuasiShuffle( x + (bullet,) ), c )
        

    def deg(self):
        if len(self) == 0:
            return 0
        else:
            return reduce(operator.add, map(Monomial.deg, self))

    def __str__(self):
        if len(self) == 0:
            return 'e'
        else:
            #return '(' + ', '.join(map(str,self)) + ')'
            return ''.join(map(str,self))

    def coproduct(self):
        """De-concatenation."""
        for i in range(len(self)+1):
            yield (lc.Tensor( (CompositionQuasiShuffle(self[:i]), CompositionQuasiShuffle(self[i:])) ), 1)

    def antipode(self):
        "Theorem 3.2 in Hoffman."""
        n = len(self)
        rev = CompositionConcatenation( reversed(self) )
        for C in coarser( rev ):
            yield (C, (-1)**n)
        #"""Antipode for the de-quasi-shuffle structure. p.58 in Hoffman"""
        #rev = CompositionConcatenation( reversed(self) )
        #for C in finer( rev ):
        #    yield (C, (-1)**len(C))

class CompositionConcatenation(tuple):

    def __add__(self,other):
        return CompositionConcatenation( super(CompositionConcatenation,self).__add__(other) )

    def __repr__(self):
        return 'CompositionConcatenation[' + super(CompositionConcatenation,self).__repr__() + ']'

    def __mul__(self,other):
        """Concatenation product."""
        yield (CompositionConcatenation(self+other),1)
    
    def __str__(self):
        if len(self) == 0:
            return 'e'
        else:
            #return '(' + ', '.join(map(str,self)) + ')'
            return ''.join(map(str,self))

    def deg(self):
        if len(self) == 0:
            return 0
        else:
            return reduce(operator.add, map(Monomial.deg, self))

    def antipode(self):
        """Antipode for the de-quasi-shuffle structure. p.58 in Hoffman"""
        rev = CompositionConcatenation( reversed(self) )
        for C in finer( rev ):
            yield (C, (-1)**len(C))

    def antipode_deshuffle(self):
        """Antipode for the deshuffle structure."""
        rev = CompositionConcatenation( reversed(self) )
        yield (rev, (-1)**len(self))

    def coproduct(self):
        """De-quasi-shuffle coproduct."""
        if len(self) == 0:
            yield (lc.Tensor( (CompositionConcatenation(),CompositionConcatenation()) ),1)
        elif len(self) == 1:
            # XXX Is this correct?
            for x, c in self[0].coproduct():
                left, right = x
                if len(left) == 0:
                    c_left = CompositionConcatenation()
                else:
                    c_left = CompositionConcatenation( (left,) )
                if len(right) == 0:
                    c_right = CompositionConcatenation()
                else:
                    c_right = CompositionConcatenation( (right,) )
                yield (lc.Tensor( (c_left, c_right) ), 1)
        else:
            tmp = map(lambda c: CompositionConcatenation( (c,) ).coproduct(), self) # XXX
            def mul(a,b):
                #print('a=', a, ' b=', b)
                return ( lc.Tensor( (list(a[0][0] * b[0][0])[0][0], list(a[0][1] * b[0][1])[0][0] ) ), 1 ) # XXX Hack
            for p in itertools.product(*tmp):
                #print('p=', p, list(p))
                x = reduce(mul, p) # XXX Hack
                #print('x=', x)
                yield x

    def coproduct_deshuffle(self):
        """De-shuffle coproduct."""
        for i in range(len(self)+1):
            yield (lc.Tensor( (CompositionConcatenation(self[:i]), CompositionConcatenation(self[i:]))), +1 )
            

def concatenation_to_quasi_shuffle(tau):
    def _cc_to_cqs( cc ):
        yield (CompositionQuasiShuffle(cc), 1)
    return tau.apply_linear_function( _cc_to_cqs )

def concatenation_to_shuffle(tau):
    def _cc_to_cs( cc ):
        yield (CompositionShuffle(cc), 1)
    return tau.apply_linear_function( _cc_to_cs )

def inner_product_qs_c(left,right):
    return lc.LinearCombination.inner_product( concatenation_to_quasi_shuffle(left), right )

def projection_smaller_equal(n):
    def p(t):
        d = t.deg()
        if isinstance(d,int):
            if d <= n:
                yield (t,1)
        else:
            # XXX Tensors not yet supported here.
            fail
        #elif t.weight() == (n,n):
        #    yield (t,1)
    return p

def project_smaller_equal(x, n):
    return x.apply_linear_function(projection_smaller_equal(n))

def projection_equal(n):
    def p(t):
        d = t.deg()
        if isinstance(d,int):
            if d == n:
                yield (t,1)
        else:
            # XXX Tensors not yet supported here.
            fail
        #elif t.weight() == (n,n):
        #    yield (t,1)
    return p

def project_equal(x, n):
    return x.apply_linear_function(projection_equal(n))

##################
def coarser(CC):
    """
    Parameters
    ----------
    CC : CompositionXXX

    Returns
    -------
    compositions : A generator of CompositionXXX of all coarser composititon.

    Examples
    --------
    (a^2,b) -> (a^2,b), (a^2 b)
    (a,b,c) -> (a,b,c), (ab,c), (a,bc), (abc)."""

    clazz = CC.__class__
    #print('CC=',CC,len(CC))
    if len(CC) == 1 or len(CC) == 0:
        yield CC
    elif len(CC) == 2:
        yield CC
        yield clazz( ( list(CC[0] * CC[1])[0][0],) ) # XXX hack
    else:
        for right in coarser(clazz(CC[1:])):
            yield clazz( CC[:1] ) + right
        #yield from coarser( clazz( ( list(CC[0] * CC[1])[0][0],) + CC[2:] ) )
        for x in coarser( clazz( ( list(CC[0] * CC[1])[0][0],) + CC[2:] ) ):
            yield x

def f(C,D):
    """
    p.973 in Malvenuto/Reutenauer

    f( (2,4,2,1,1), (6,4) ) = 2! 3! """
    ells = []
    C_remaining = list(C)
    for d in D:
        total_in_d = d.deg()
        ell = 0
        while total_in_d > 0:
            ell += 1
            total_in_d -= C_remaining[0].deg()
            C_remaining = C_remaining[1:]
        ells.append( ell )
    return reduce(operator.mul, map(sympy.factorial, ells), 1)

def test_f():
    C = D = CompositionShuffle( ( Monomial({'a':1}),Monomial({'a':1}),Monomial({'a':2})) )
    assert 1 == f(C,D)
    C = D = CompositionShuffle( (Monomial({'a':2}), Monomial({'a':1}),Monomial({'a':1})) )
    assert 1 == f(C,D)

    C = CompositionShuffle( tuple( Monomial({'a':n}) for n in [2,4,2,1,1] ) )
    D = CompositionShuffle( tuple( Monomial({'a':n}) for n in [6,4] ) )
    assert sympy.factorial(2) * sympy.factorial(3) == f(C,D)

def g(C,D):
    """
    g( (2,4,2,1,1), (6,4) ) = 2 * 3 """
    ells = []
    C_remaining = list(C)
    for d in D:
        total_in_d = d.deg()
        ell = 0
        while total_in_d > 0:
            ell += 1
            total_in_d -= C_remaining[0].deg()
            C_remaining = C_remaining[1:]
        ells.append( ell )
    return reduce(operator.mul, ells,1)

def id_(x):
    if len(x) > 0:
        yield (x,1)

def finer(DD): # XXX Slow.
    """
    Parameters
    ----------
    DD : CompositionXXX

    Returns
    -------
    compositions : A generator of CompositionConcatenation of all finer composititon.

    Examples
    --------
    (a,b,c) -> (a,b,c)
    (a^2b,c) -> (ab,a,c), (a,ab,c), (a^2,b,c), (b,a^2,c), (a,a,b,c), (a,b,a,c), (b,a,a,c)."""
    # XXX This returns CompositionConcatenation's.
    def finer_monomial(dp_mon):
        yield dp_mon
        lc_dp_mon = lc.lift( dp_mon )\
                         .coproduct()\
                         .apply_linear_function( lc.Tensor.fn_otimes_linear( id_, id_ ) )
        for x in lc_dp_mon.apply_linear_function( lc.Tensor.m ):
            yield x
        for i in range(1,dp_mon.deg()):
            lc_dp_mon = lc_dp_mon\
                         .apply_linear_function( lc.Tensor.fn_otimes_linear( *( [CompositionConcatenation.coproduct] + [id_]*i ) ) )\
                         .apply_linear_function( lc.Tensor.fn_otimes_linear( * [id_] * (i+2)  ) )
            for x in lc_dp_mon.apply_linear_function( lc.Tensor.m ):
                yield x
    if len(DD) == 0:
        yield CompositionConcatenation( DD )
    else:
        for tmp in itertools.product( * map( lambda x: finer_monomial(CompositionConcatenation( (x,)) ), DD ) ):
            yield reduce(operator.add, tmp)
##################

def exp( x, upto_level ):
    xell = lc.lift( CompositionConcatenation() )
    ret = xell
    for ell in range(upto_level+1):
        xell = project_smaller_equal(xell * x,upto_level) 
        ret += 1/sympy.factorial(ell) * xell
    return ret

def pi1_adjoint( x, upto_level ):
    return pi1( x, upto_level )

def pi1( x, upto_level ):
    with words.UseRationalContext():
        ret = words.pi1( x, upto_level )
    return ret

def test_pi1():
    x = lc.lift( CompositionConcatenation( (Monomial({'a':2}),) ) )
    assert is_primitive( pi1( x, 2 ) )
    assert pi1( x, 2 ) == pi1( pi1( x, 2 ), 2 )

def is_primitive(_x):
    e = lc.lift( CompositionConcatenation() )
    return _x.apply_linear_function( CompositionConcatenation.coproduct )\
            == lc.LinearCombination.otimes(_x,e) + lc.LinearCombination.otimes(e, _x)

def _hoffman_EXP( C ):
    for D in coarser(C):
        yield (CompositionQuasiShuffle(D), sympy.Rational(1, f(C,D)))

def hoffman_EXP(phi):
    """p.52 in Hoffman"""
    return phi.apply_linear_function( _hoffman_EXP )

def hoffman_EXP_dual(phi):
    """p.58 in Hoffman"""
    # XXX Slow hack.
    def _EXP_dual_monomial(m):
        _ret = lc.LinearCombination()
        for C in finer(CompositionConcatenation( (m,) )):
            _ret += sympy.Rational( 1, sympy.factorial(len(C)) ) * lc.lift(C)
        return _ret

    ret = lc.LinearCombination()
    for D, c in phi.items():
        ret += c * reduce(operator.mul, map(_EXP_dual_monomial, D))
    return ret

def _hoffman_LOG( C ):
    for D in coarser(C):
        yield (D, sympy.Rational((-1)**( len(C) - len(D) ), g(C,D)))

def hoffman_LOG(psi):
    """p.52 in Hoffman"""
    return psi.apply_linear_function( _hoffman_LOG )

def hoffman_LOG_dual(psi):
    """p.58 in Hoffman"""
    # XXX Slow hack.
    def _LOG_dual_monomial(m):
        _ret = lc.LinearCombination()
        for C in finer(CompositionConcatenation( (m,) )):
            _ret += sympy.Rational( (-1)**(len(C)-1), len(C) ) * lc.lift(C)
        return _ret
    ret = lc.LinearCombination()
    for D, c in psi.items():
        ret += c * reduce(operator.mul, map(_LOG_dual_monomial, D))
    return ret

