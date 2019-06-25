import setuptools


def readme():
    with open('README.rst') as f:
        return f.read()

setuptools.setup(name='iterated-sums-signature-py',
      #python_requires='>=3.5.2',
      version='0.1.0',
      packages=['iterated_sums_signature'],
      description='An (inefficient) implementation of the iterated-sums signature.',
      long_description=readme(),
      author='Joscha Diehl',
      #author_email='',
      url='https://github.com/diehlj/iterated-sums-signature-py',
      license='Eclipse Public License',
      install_requires=['numpy', 'scipy', 'sympy'],
      #setup_requires=['setuptools_git >= 0.3', ],
      test_suite='tests'
      )
