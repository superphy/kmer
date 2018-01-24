from distutils.core import setup
setup(name='kmerpredict',
      version='0.1',
      author='boothmanrylan',
      author_email='boothmanrylan@gmail.com',
      url='https://github.com/superphy/kmer',
      py_modules=['kmerprediction'],
      requires=['biopython', 'keras', 'matplotlib', 'numpy', 'pandas', 'pyyaml',
                'tensorflow', 'yaml', 'nose2', 'imblearn'],
      description='Phenotypic predictions based on gene sequence data'
      )
