iterated-sums-signature-py
--------------------------

An (inefficient) implementation of the `iterated-sums signature (also called discrete signature) <https://arxiv.org/abs/1906.05823>` (a discrete generalization of Chen's iterated-integrals signature)::

   import iterated_sums_signature as iss 
   x = [1., 10., -99, -50] 
   sig = iss.signature( x, upto_level=5 ) 
   print( sig ) 


Installation
------------

Install with::

   pip install git+https://github.com/diehlj/linear-combination-py
   pip install git+https://github.com/diehlj/iterated-sums-signature-py

Copyright © 2019 Joscha Diehl

Distributed under the `Eclipse Public License <https://opensource.org/licenses/eclipse-1.0.php>`_.
