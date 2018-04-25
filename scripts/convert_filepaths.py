def validation(filepath):
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
    selection = name[3].replace('_', ' ').capitalize()

    output = {'model': model, 'datatype': datatype, 'fragment': fragment,
              'k': k, 'filter': f, 'dataset': dataset, 'selection': selection}
    return output

def omnilog(filepath):
    name = filepath.replace('results/omnilog/yaml/', '')
    name = name.split('/')
    model = name[0].replace('_', ' ').capitalize()
    if name[1] == 'omnilog':
        datatype = 'Omnilog'
        k = None
        f = None
    else:
        kmer_info = name[1].split('_')
        datatype = 'Kmer'
        k = int(kmer_info[0].replace('mer', ''))
        f = kmer_info[1].replace('_', ' ').capitalize()
    selection = name[2].replace('_', ' ').capitalize()
    prediction = name[3].replace('_', ' ').capitalize()
    ova = name[4].replace('_', ' ').capitalize()

    output = {'model': model, 'datatype': datatype, 'k': k, 'filter': f,
              'selection': selection, 'prediction': prediction, 'ova': ova}
    return output
