from distutils.core import setup

setup(name='kmerprediction',
      version='0.0.1',
      author='Chad Laing, Rylan Boothman',
      author_email='chadr.laing@canada.ca, boothmanrylan@gmail.com',
      url='https://github.com/superphy/kmer',
      packages=['kmerprediction'],
      package_data={'kmerprediction': ['Data/*']},
      requires=['biopython', 'keras', 'matplotlib', 'numpy', 'pandas',
                'pyyaml', 'tensorflow', 'yaml', 'nose2', 'imblearn'],
      description='Phenotypic predictions based on gene sequence data')
