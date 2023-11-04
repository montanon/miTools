from itertools import chain, cycle
from IPython.display import display_html


def display_dataframes_side_by_side(dfs, titles=None):

    if titles is None:
        titles = cycle([''])

    html_str = ''

    for df, title in zip(dfs, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h2 style="text-align: center;">{title}</h2>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)