#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import re
import glob
from datetime import datetime
import nltk
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.rcParams['figure.figsize']= (17,5)
from collections import Counter
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
from tqdm import tqdm
from nltk.corpus import PlaintextCorpusReader

#nltk.download("punkt")
database = [ ]
corpus = PlaintextCorpusReader("./DATA", ".*\.txt")
files=corpus.fileids()
for f in files:
    number_of_sentences = len(corpus.sents(f))
    number_of_words = len(
        [word for sentence in corpus.sents(f) for word in sentence])
    number_of_fids = len(corpus.fileids())
    path = "./DATA/"
    newstr = "".join((path, f))
    with open(newstr, encoding= 'Latin') as speech_file:
        number = str(''.join(filter(str.isdigit, f)))
        f = f.replace(".txt", "")
        debater = ''.join(filter(lambda x: not x.isdigit(), f))
        speeches = {
            'filename' : f,
            'content' : speech_file.read(),
            'debater' : debater,
            'debate_number' : number,
            'number_of_sentences' : number_of_sentences,
            'number_of_words' : number_of_words,
            'average_sentence_length': number_of_words/number_of_sentences
        }   
    database.append(speeches)
df = pd.DataFrame(database)
df.head()
get_ipython().run_line_magic('store', 'df')


# In[34]:


df.head()


# In[35]:


#WEB APP
#from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_building_blocks as dbb
import plotly.express as px
import plotly.subplots as sp


#get_ipython().run_line_magic('store', '-r df')
#import quandl

class Graph(dbb.Block):

    def layout(self):
        return html.Div([
            dcc.Dropdown(
                id=self.register('dropdown'),
                options=self.data.options,
                value=self.data.value
            ),
            dcc.Graph(id=self.register('graph'))
        ], style={'width': '500'})

    def callbacks(self):
        @self.app.callback(
            self.output('graph', 'figure'),
            [self.input('dropdown', 'value')]
        )
        def update_charts(Debator):
            figure1_traces = []
            figure2_traces = []
            filtered_data = df[df["debator"] == Debator] #the graph/dataframe will be filterd by "Debator"
            scatter = px.scatter(
                    filtered_data,
                    x="average_sentence_length",
                    y="debator",
                    size="number_of_sentences",
                    color="number_of_sentences",
                    color_continuous_scale=px.colors.sequential.Plotly3,
                    title="Something funny test 2",
            )
            scatter.update_layout(
                width =500, 
                xaxis_tickangle=30,
                title=dict(x=0.5),
                xaxis_tickfont=dict(size=9),
                yaxis_tickfont=dict(size=9),
                margin=dict(l=500, r=20, t=50, b=20),
                paper_bgcolor="LightSteelblue",
            )
            for trace in range(len(scatter["data"])):
                figure1_traces.append(scatter["data"][trace])
            
            bar = px.bar(
                filtered_data,
                x=filtered_data.groupby("debator")["number_of_words"].agg(sum),
                y=filtered_data["debator"].unique(),
                color=filtered_data.groupby("debator")["number_of_words"].agg(sum),
                color_continuous_scale=px.colors.sequential.RdBu,
                text=filtered_data.groupby("debator")["number_of_words"].agg(sum),
                title="Something funny test 1",
                orientation="h",
            )
            bar.update_layout(
                title=dict(x=0.5), margin=dict(l=550, r=20, t=60, b=20), paper_bgcolor="#D6EAF8"
            )
            bar.update_traces(texttemplate="%{text:.2s}")
            for trace in range(len(bar["data"])):
                figure2_traces.append(bar["data"][trace])
            
            this_figure = sp.make_subplots(rows=2, cols=1) 
            for traces in figure1_traces:
                this_figure.append_trace(traces, row=1, col=1)
            for traces in figure2_traces:
                this_figure.append_trace(traces, row=2, col=1)
            this_figure.update_layout(
                width =800,
                height=600)
            return this_figure
        
        

#app = JupyterDash(__name__, external_stylesheets=[
 #   "https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap-grid.min.css"
#])

app = dash.Dash(__name__, external_stylesheets=[
    "https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap-grid.min.css"
])
server = app.server

options = [
            {'label': debator, 'value':debator}
            for debator in df.debater.unique()
          ]

data = {
    'options': options,
    'value': 'Biden'
}

n_graphs = 2
graphs = [Graph(app, data) for _ in range(n_graphs)]
print(graphs)

app.layout = html.Div(
    [html.Div(graph.layout, className='container')
    for graph in graphs],
    className='six.columns',
    style = {'display' : 'flex'}
)


for graph in graphs:
    graph.callbacks()

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run()




