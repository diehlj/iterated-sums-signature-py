import iterated_sums_signature as iss
from iterated_sums_signature import divided_powers as dp
from iterated_sums_signature.divided_powers import M_qs, M_concat, M_sh
import pytest
import numpy as np
from linear_combination import linear_combination as lc
from functools import reduce

def get(z,_DS):
    return lc.LinearCombination.inner_product( z, _DS )

def test_signature():
    x = np.array( [ [0,10,-20,50],
                    [0,-200,100,500]  ] )

    DS = iss.signature( x, 6 )

    #print('DS=', DS)
    #xx

    assert np.array_equal( get( M_qs({0:1},{0:1}), DS ),\
                           np.array( [0, 0, 10 * (-30), 10 * (-30) + (10 - 30) * 70 ] ) )

    assert np.array_equal( get( M_qs({0:2}), DS ),\
                           np.array( [0, 10**2, 10**2  + (-30)**2, 10**2  + (-30)**2 + 70**2] ) )


    ##################
    # Chen's identity.
    ##################
    z = np.array( [ [0,  10,-20,  50, 30,90],
                    [0,-200,100,-800,500,100]  ] )

    x = z[:,:4]
    y = z[:,3:] # XXX
        
    DSx = iss.terminal_values( iss.signature(x,3) )
    DSy = iss.terminal_values( iss.signature(y,3) )
    DSxy = iss.terminal_values( iss.signature( z, 3 ) ).remove_zeros()
    DSxDSy = dp.project_smaller_equal( DSx * DSy, 3 ).remove_zeros()
    assert DSxDSy == DSxy

    ###################
    # Shuffle identity.
    ###################
    comps = list(dp.compositions(range(2), 3))
    for i in range(len(comps)):
        for j in range(i+1):
            C1 = comps[i] 
            C2 = comps[j] 
            phi = M_qs(*C1)
            psi = M_qs(*C2)
            assert np.array_equal( get(phi,DS) * get(psi,DS), get(phi * psi,DS) )

@pytest.mark.slow
def test_lead_lag():
    #################
    # Map to lead-lag.
    #################
    def replace_multidim( m, _dim ):
        # [0,1,2]
        # [0',0'',1',1'',2',2'']
        if len(m) == 0:
            fail
        types = []
        tmp = {}
        for d in range(2 * _dim):
            if m.get( d, 0 ) >= 1:
                go_to = d // 2
                if d % 2 == 0:
                    typ = 'left'
                else:
                    typ = 'right'
                types.append( typ )
                tmp[ go_to ] = m[d]
        if all(map(lambda s: s=='left',types)) or all(map(lambda s: s=='right',types)):
            return types[0], dp.Monomial( tmp )
        else:
            return None, None

    def _ll_to_ds_multidim( _dim ):
        def __ll_to_ds_multidim( cqs ):
            if len(cqs) == 1:
                t, rep = replace_multidim(cqs[0],_dim)
                if not rep is None:
                    yield (dp.CompositionQuasiShuffle( (dp.Monomial(rep),) ),+1)
            else:
                last = cqs[-1]
                t_last, rep_last = replace_multidim(last,_dim)
                if rep_last == None:
                    return
                second_to_last = cqs[-2]
                t_second_to_last, rep_second_to_last = replace_multidim(second_to_last,_dim)
                if rep_second_to_last == None:
                    return
                for x, c in __ll_to_ds_multidim( cqs[0:-1] ):
                    yield (dp.CompositionQuasiShuffle( x + (rep_last,) ), +1)
                    if t_second_to_last == 'left' and t_last == 'right':
                        merged = dp.Monomial( x[-1] )
                        for _x, _p in rep_last.items():
                            dp.safe_add( merged, _x, _p )
                        yield (dp.CompositionQuasiShuffle( x[:-1] + (merged,) ), +1)
        return __ll_to_ds_multidim

    def ll_to_ds_multidim( phi, _dim ):
        return phi.apply_linear_function( _ll_to_ds_multidim(_dim) )

    z = np.random.randint(-100,100, size=(2,10))

    doubles = []
    for d in range(2):
        double = []
        for i in range(0,len(z[d])):
            double.append(z[d,i])
            double.append(z[d,i])
        doubles.append(double)
    xs = []
    for d in range(2):
        xs.append( doubles[d][1:] )
        xs.append( doubles[d][:-1] )
    x = np.vstack( xs )
    DSx = iss.terminal_values( iss.signature( x, 4 ) )
    DSz = iss.terminal_values( iss.signature( z, 4 ) )

    # 1' \mapsto 1
    assert get(M_qs({0:1}), DSx) == get(M_qs({0:1}), DSz)
    assert ll_to_ds_multidim(M_qs({0:1}),2) == M_qs({0:1})
    # 1'' \mapsto 1
    assert get(M_qs({1:1}), DSx) == get(M_qs({0:1}), DSz)
    assert ll_to_ds_multidim(M_qs({1:1}),2) == M_qs({0:1})
    # [1'^2] \mapsto (1^2)
    assert get(M_qs({0:2}), DSx) == get(M_qs({0:2}), DSz)
    assert ll_to_ds_multidim(M_qs({0:2}),2) == M_qs({0:2})
    # [1''^2] \mapsto (1^2)
    assert get(M_qs({1:2}), DSx) == get(M_qs({0:2}), DSz)
    assert ll_to_ds_multidim(M_qs({1:2}),2) == M_qs({0:2})
    # [1'1'']  \mapsto _
    assert get(M_qs({0:1,1:1}), DSx) == 0
    assert ll_to_ds_multidim(M_qs({0:1,1:1}),2) == lc.LinearCombination()
    # 1'1'    \mapsto 11
    assert get(M_qs({0:1},{0:1}), DSx) == get(M_qs({0:1},{0:1}), DSz)
    assert ll_to_ds_multidim(M_qs({0:1},{0:1}),2) == M_qs({0:1},{0:1})
    # 1'1''    \mapsto 11 + [1^2]
    assert get(M_qs({0:1},{1:1}), DSx) == get(M_qs({0:1},{0:1}), DSz) + get(M_qs({0:2}), DSz)
    assert ll_to_ds_multidim(M_qs({0:1},{1:1}),2) == M_qs({0:1},{0:1}) + M_qs({0:2})
    # 1''1'    \mapsto 11
    assert get(M_qs({1:1},{0:1}), DSx) == get(M_qs({0:1},{0:1}), DSz)
    assert ll_to_ds_multidim(M_qs({1:1},{0:1}),2) == M_qs({0:1},{0:1})
    # 1'1''1'   \mapsto  111 + (1^2)1
    assert get(M_qs({0:1},{1:1},{0:1}), DSx) == get(M_qs({0:1},{0:1},{0:1}), DSz) + get(M_qs({0:2},{0:1}), DSz)
    assert ll_to_ds_multidim(M_qs({0:1},{1:1},{0:1}),2) == M_qs({0:1},{0:1},{0:1}) + M_qs({0:2},{0:1})
    # 1''1'1''   \mapsto  111 + 1(1^2)
    assert get(M_qs({1:1},{0:1},{1:1}), DSx) == get(M_qs({0:1},{0:1},{0:1}), DSz) + get(M_qs({0:1},{0:2}), DSz)
    assert ll_to_ds_multidim(M_qs({1:1},{0:1},{1:1}),2) == M_qs({0:1},{0:1},{0:1}) + M_qs({0:1},{0:2})
    # 1''[1'^2] \mapsto  1[1^2]
    assert get(M_qs({1:1},{0:2}), DSx) == get(M_qs({0:1},{0:2}), DSz)
    assert ll_to_ds_multidim(M_qs({1:1},{0:2}),2) == M_qs({0:1},{0:2})
    # 1'[1''^2] \mapsto  1[1^2] + [1^3]
    assert get(M_qs({0:1},{1:2}), DSx) == get(M_qs({0:1},{0:2}), DSz) + + get(M_qs({0:3}), DSz)
    assert ll_to_ds_multidim(M_qs({0:1},{1:2}),2) == M_qs({0:1},{0:2}) + M_qs({0:3})

    # It does the job.
    for C1 in dp.compositions(range(4), 4):
        phi = M_qs(*C1)
        assert get( ll_to_ds_multidim(phi,2), DSz ) == get(phi, DSx )

    # It is a quasi-shuffle morphism.
    assert ll_to_ds_multidim(M_qs({0:1},{1:1}),2) * ll_to_ds_multidim(M_qs({2:1},{3:1}),2) ==  ll_to_ds_multidim( M_qs({0:1},{1:1}) * M_qs({2:1},{3:1}), 2 )
    assert ll_to_ds_multidim(M_qs({0:1},{1:1}),2) * ll_to_ds_multidim(M_qs({2:1},{2:1}),2) ==  ll_to_ds_multidim( M_qs({0:1},{1:1}) * M_qs({2:1},{2:1}), 2 )
    assert ll_to_ds_multidim(M_qs({0:1},{1:1}),2) * ll_to_ds_multidim(M_qs({3:1},{3:1}),2) ==  ll_to_ds_multidim( M_qs({0:1},{1:1}) * M_qs({3:1},{3:1}), 2 )
    assert ll_to_ds_multidim(M_qs({0:1},{0:1}),2) * ll_to_ds_multidim(M_qs({2:1},{2:1}),2) ==  ll_to_ds_multidim( M_qs({0:1},{0:1}) * M_qs({2:1},{2:1}), 2 )
    comps = list(dp.compositions(range(4), 3))
    for i in range(len(comps)):
        for j in range(i+1):
            C1 = comps[i] 
            C2 = comps[j] 
            phi = M_qs(*C1)
            psi = M_qs(*C2)
            assert ll_to_ds_multidim( phi * psi, 2 ) == ll_to_ds_multidim( phi, 2 ) * ll_to_ds_multidim( psi, 2 )
            # NOT a half-shuffle morphism.
            # Fails for phi=2, psi=1.
            #assert ll_to_ds_multidim( bi( dp.CompositionQuasiShuffle.half_shuffle, phi, psi ), 2 )\
            #        == bi( dp.CompositionQuasiShuffle.half_shuffle, ll_to_ds_multidim(phi,2), ll_to_ds_multidim(psi,2) )
