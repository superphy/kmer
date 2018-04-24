def convert_filepath(filepath)
    name = filepath.replace('results/omnilog/yaml', '')
    name = name.split('/')
    model = name[0].replace('_', ' ').title()
    if name[1] == 'omnilog':
        datatype = 'Omnilog'
        k = None
        f = None
    else:
        kmer_info = name[1].split('_')
        datatype = 'Kmer'
        k = int(kmer_info[0].replace('mer', ''))
        f = kmer_info[1].replace('_', ' ').title()
    selection = name[2].replcae('_', ' ').title()
    prediction = name[3].replcae('_', ' ').title()
    ova = name[4].replcae('_', ' ').title()

    output = {'model': model, 'datatype': datatype, 'k': k, 'filter': f,
              'selection': selection, 'prediction': prediction, 'ova': ova}
    return output
