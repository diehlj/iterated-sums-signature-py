import iterated_sums_signature as iss
from iterated_sums_signature import divided_powers as dp
from iterated_sums_signature.divided_powers import M_qs, M_concat, M_sh
import pytest
import numpy as np
from linear_combination import linear_combination as lc
from functools import reduce

def test_monomials():
    assert [{'a': 1}, {'b': 1}, {'a': 2}, {'a': 1, 'b': 1}, {'b': 2}] == list(dp.monomials(['a','b'], 2 ))

def test_compositions():
    assert [ dp.CompositionConcatenation( (dp.Monomial({'a':1}),) ), dp.CompositionConcatenation( (dp.Monomial({'b':1}),) ) ]\
            ==\
            list(dp.compositions(['a','b'], 1))

def test_quasi_shuffle():
    X = dp.M_qs( dp.Monomial({'1':1}) )
    Y = dp.M_qs( dp.Monomial({'1':3}), dp.Monomial({'1':7}))
    assert dp.M_qs( dp.Monomial({'1':1}), dp.Monomial({'1':3}), dp.Monomial({'1':7}) )\
         + dp.M_qs( dp.Monomial({'1':3}), dp.Monomial({'1':1}), dp.Monomial({'1':7}) )\
         + dp.M_qs( dp.Monomial({'1':3}), dp.Monomial({'1':7}), dp.Monomial({'1':1}) )\
         + dp.M_qs( dp.Monomial({'1':4}), dp.Monomial({'1':7}) )\
         + dp.M_qs( dp.Monomial({'1':3}), dp.Monomial({'1':8}) )\
         == X * Y

def test_divided_powers():
    def ot(a,b):
        return lc.LinearCombination.otimes(a,b)

    assert M_concat({'a':2}).coproduct()\
           == ot( M_concat(), M_concat({'a':2}) ) + ot( M_concat({'a':1}), M_concat({'a':1}) ) + ot( M_concat({'a':2}), M_concat() )

    assert M_concat({'a':2, 'b':1}).coproduct().apply_linear_function( lc.Tensor.fn_otimes_linear( *[ dp.id_, dp.id_ ] ) )\
            == ot( M_concat({'b':1}), M_concat({'a':2}) )\
             + ot( M_concat({'a':1,'b':1}), M_concat({'a':1}) )\
             + ot( M_concat({'a':1}), M_concat({'a':1,'b':1}) )\
             + ot( M_concat({'a':2}), M_concat({'b':1}) )

