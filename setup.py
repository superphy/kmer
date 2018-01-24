from distutils.core import setup
setup(name='kmerpredict',
      version='0.1',
      author='boothmanrylan',
      author_email='boothmanrylan@gmail.com',
      url='https://github.com/superphy/kmer',
      py_modules=['kmer_counter', 'constants', 'data_augmentation',
                  'feature_scaling', 'feature_selection', 'get_data',
                  'kmer_counter', 'models', 'run', 'utils'],
      requires=['biopython', 'keras', 'matplotlib', 'numpy', 'pandas', 'pyyaml',
                'tensorflow', 'yaml', 'nose2', 'imblearn'],
      description='Phenotypic predictions based on gene sequence data'
      )
