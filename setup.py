from distutils.core import setup
setup(name='kmerpredict',
      version='0.2',
      author='Chad Laing, Rylan Boothman',
      author_email='chadr.laing@canada.ca, boothmanrylan@gmail.com',
      url='https://github.com/superphy/kmer',
      packages=['kmerprediction'],
	  package_data={'kmerprediction': ['Data/*']},
	  scripts=['bin/kmerprediction'],
      requires=['biopython', 'keras', 'matplotlib', 'numpy', 'pandas', 'pyyaml',
                'tensorflow', 'yaml', 'nose2', 'imblearn'],
      description='Phenotypic predictions based on gene sequence data'
      )
