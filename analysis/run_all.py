import yaml
import os
import sys
sys.path.append('../../')
from run import main

all_files = [x for x in os.listdir('./config_files/') if '.yml' in x]

for f in all_files:
    main(f, './results/omnilog/%s' % f, f.replace('.yml', ''))




