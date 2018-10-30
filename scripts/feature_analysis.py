import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, CDSView, GroupFilter
from bokeh.models import FactorRange, BooleanFilter, Legend
from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import row, widgetbox
from bokeh.palettes import Dark2
from bokeh.models.widgets import CheckboxButtonGroup, Button, RadioButtonGroup, PreText
from bokeh.client import push_session
from functools import partial
import random

kmer_length = 7
df = pd.read_csv('{}mer_features.csv'.format(kmer_length))

models = ['Random forest', 'Support vector machine']
datasets = ['Mixed', 'Reverse split', 'Split', 'UK', 'US']
filters = ['Complete', 'Complete filtered', 'Filtered']
sorts = {}
for m in models:
    for d in datasets:
        for f in filters:
            temp = df.loc[df['Model'] == m]
            temp = temp.loc[temp['Dataset'] == d]
            temp = temp.loc[temp['Filter'] == f]
            temp = temp.sort_values(by='Score', ascending=False)
            xrange = temp['Kmer'].values
            sorts['{} {} {}'.format(m, d, f)] = xrange

converters = {'Model': {0: 'Random forest', 1: 'Support vector machine'},
              'Dataset': {0: 'UK', 1: 'US', 2: 'Mixed', 3: 'Reverse split', 4: 'Split'},
              'Filter': {0: 'Complete', 1: 'Complete filtered', 2: 'Filtered'}}

source = ColumnDataSource(df)

def create_figure():
    active = {}
    for w in sort_widgets:
        active[w.name] = converters[w.name][w.active]
    key = '{} {} {}'.format(active['Model'], active['Dataset'], active['Filter'])
    xrange = sorts[key]

    alpha_dict = {'Support vector machine': 1.0, 'Random forest': 0.1}
    color_dict = {'Split': Dark2[5][0], 'Mixed': Dark2[5][1],
                  'Reverse split': Dark2[5][2], 'US': Dark2[5][3],
                  'UK': Dark2[5][4]}
    shape_dict = {'Complete': 'circle', 'Complete filtered': 'diamond',
                  'Filtered': 'square'}

    visible_models = [modelboxes.labels[x] for x in modelboxes.active]
    visible_datasets = [datasetboxes.labels[x] for x in datasetboxes.active]
    visible_filters = [filterboxes.labels[x] for x in filterboxes.active]
    height = 600 + (22*(len(visible_models) * len(visible_datasets) * len(visible_filters)))
    p = figure(x_range=FactorRange(factors=xrange),
               x_axis_label='Kmer', y_axis_label='Score', plot_width=900,
               plot_height=height, output_backend='webgl')
    p.xaxis.major_label_orientation = 3.1415/4
    legend = []
    for model in visible_models:
        for dataset in visible_datasets:
            for f in visible_filters:
                curr_filter = [GroupFilter(column_name='Model', group=model),
                               GroupFilter(column_name='Dataset', group=dataset),
                               GroupFilter(column_name='Filter', group=f),
                               bools = [True if k in xrange else False for k in source.data['Kmer']]
                               BooleanFilter(bools)]
                view = CDSView(source=source, filters=curr_filter)
                alpha = alpha_dict[model]
                color = color_dict[dataset]
                shape = shape_dict[f]
                size=10
                glyph = p.scatter(x='Kmer', y='Score', source=source, view=view,
                                  color=color, fill_alpha=alpha, size=size,
                                  marker=shape)
                legend.append(("{} {} {}".format(model, dataset, f), [glyph]))
    p.add_layout(Legend(items=legend), 'below')
    return p

modelboxes = CheckboxButtonGroup(labels=np.unique(df['Model'].values).tolist(),
                                 active=[0], name='Model')
datasetboxes = CheckboxButtonGroup(labels=np.unique(df['Dataset'].values).tolist(),
                                   active=[0], name='Dataset')
filterboxes = CheckboxButtonGroup(labels=np.unique(df['Filter'].values).tolist(),
                                  active=[0], name='Filter')
showall = Button(label='Show All')
hideall = Button(label='Hide All')
modelsort = RadioButtonGroup(labels=np.unique(df['Model'].values).tolist(),
                             active=0, name='Model')
datasetsort = RadioButtonGroup(labels=np.unique(df['Dataset'].values).tolist(),
                             active=0, name='Dataset')
filtersort = RadioButtonGroup(labels=np.unique(df['Filter'].values).tolist(),
                             active=0, name='Filter')

visibility_widgets = [modelboxes, datasetboxes, filterboxes]
sort_widgets = [modelsort, datasetsort, filtersort]

def handler(new):
    layout.children[0] = create_figure()

def showall_handler():
    for w in visibility_widgets:
        w.active = list(range(len(w.labels)))
    layout.children[0] = create_figure()

def hideall_handler():
    for w in visibility_widgets:
        w.active = []
    layout.children[0] = create_figure()

modelboxes.on_click(handler)
datasetboxes.on_click(handler)
filterboxes.on_click(handler)
showall.on_click(showall_handler)
hideall.on_click(hideall_handler)
modelsort.on_click(handler)
datasetsort.on_click(handler)
filtersort.on_click(handler)

legendtext = ('\n'*8 + 'Solid: SVM Score\nOutline: RF Score\n\n' +
              'Green: UK\nPink: US\nOrange: Mixed\n' +
              'Purple: Reverse Split\nBlue: Split\n\n' +
              'Circle: Complete\nDiamond: Filtered')
controls = widgetbox([PreText(text='Choose Visible Items')] +
                     visibility_widgets + [showall, hideall] +
                     [PreText(text='Choose What to Sort Kmers by')] +
                      sort_widgets +
                     [PreText(text=legendtext)])
layout = row(create_figure(), controls)

curdoc().add_root(layout)
