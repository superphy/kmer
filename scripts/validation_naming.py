def convert_filepath(filepath):
    name = filepath.replace('results/validation/yaml/', '')
    name = name.split('/')
    model = name[0].replace('_', ' ').capitalize()
    if 'genome' in name[1]:
        datatype = 'Genome region'
        fragment = int(name[1].replace('genome', ''))
        k = None
        f = None
    else:
        kmer_info = name[1].split('_')
        datatype = 'Kmer'
        k = int(kmer_info[0].replace('mer', ''))
        f = ' '.join(kmer_info[1:]).capitalize()
        fragment = None
    dataset = name[2].replace('_', ' ').capitalize()
    if dataset == 'Uk' or dataset == 'Us':
        dataset = dataset.upper()
    selection = name[3].replcae('_', ' ').capitalize()

    output = {'model': model, 'datatype': datatype, 'fragment': fragment,
              'k': k, 'filter': f, 'dataset': dataset, 'selection': selection}
    return output
