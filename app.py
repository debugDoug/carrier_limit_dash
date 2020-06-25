# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 08:11:43 2020

@author: 1197058
"""

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import dash_table
import plotly.express as px

#hey

def generate_table(dataframe, max_rows=100):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

#%% Load in IFCL
df = pd.read_excel(r'M:\Workspace\Doug\Carrier Limit Tracking\Data\DailyWeekly\062220_Inforced Carrier Limit.xlsx')
# drop Centauri
df = df[df['Carrier Display Name'] != 'Centauri Specialty Insurance Company']
# drop Channel and Lloyds - DBD Slip (100% Channel) (as per Craig) and remove Harco (counted in iPartners report)
#df = df[df['Carrier Display Name'] != 'Argo Re']
#df = df[df['Carrier Display Name'] != 'Ariel']
df = df[df['Carrier Display Name'] != 'Channel-DBD']
df = df[df['Carrier Display Name'] != 'Lloyds - DBD Slip (100% Channel)']
df = df[df['Carrier Display Name'] != 'Harco National Insurance Company']
df = df[df['Carrier Display Name'] != 'Exclude']
# change S4242 Re to Syndicate 4242
df['Carrier Display Name'].replace("S4242 Re", "Syndicate 4242", inplace=True)
#change Exclude to QBE
#df['Carrier Display Name'].replace("Exclude","QBE", inplace=True)
# change Crum & Forster_PBU to Crum and Forster
df['Carrier Display Name'].replace("Crum & Forster_PBU","Crum and Forster", inplace=True)
# change RenRe_PBU to RenRe
df['Carrier Display Name'].replace("RenRe_PBU","RenRe", inplace=True)
# change NF&M and BHSIC to Berkshire Hathaway
df['Carrier Display Name'].replace("NF&M","Berkshire Hathaway", inplace=True)
df['Carrier Display Name'].replace("BHSIC","Berkshire Hathaway", inplace=True)
# change Ariel and Argo Re
df['Carrier Display Name'].replace("Ariel", "Other", inplace=True)
df['Carrier Display Name'].replace("Argo Re", "Other", inplace=True)

# create macrozone column
df['Microzone'].value_counts()
df['Microzone'].isna().sum()
df['Microzone'].fillna('UNKNOWN', inplace=True)

microMacro = pd.read_excel(r'M:\Workspace\Doug\Carrier Limit Tracking\Data\DailyWeekly\MicroToMacro.xlsx')
microMacro = dict(zip(microMacro.Microzone, microMacro.Macrozone))

df['Macrozone'] = df['Microzone'].map(microMacro)

df.Macrozone.value_counts()
check = df[df['Macrozone'].isna()]

# create segment column
def segment (row):
    if row['Carrier Display Name'] == 'Harco National Insurance Company':
        return 'HBU'
    if row['Carrier Display Name'] == 'Syndicate 2288_Harco Auth Participant':
        return 'HBU'
    elif row['Source System'] == 'Epicenter':
        return 'PBU'
    else:
        return 'MMBU'
    
df['Segment'] = df.apply(lambda row: segment(row), axis=1)
# print(df['Segment'].value_counts())

# change 2288 HBU name
df['Carrier Display Name'].replace("Syndicate 2288_Harco Auth Participant", "Syndicate 2288", inplace=True)

# create state col
df['State'] = df.Microzone.str[:2]
df['State'] = df['State'].astype(str)

# create peril col
wind_zones = ['AL','FL','GA','HI','LA','MA','MS','NC','NJ','NY','TX']
quake_zones = ['CA','OR','WA']


def peril (row):
    if row['Policy Number'][0] == 'A': # doesn't appear to exist anymore
        return 'App'
    if (row['Policy Number'][0] == 'E')  & (row['State'] in (wind_zones)) : # EQX polcies, was NAC now HU
        return 'HU'
    if (row['Policy Number'][0] == 'E') & (row['State'] in (quake_zones)) : # EQX polcies, was NAC now EQ
        return 'EQ'
    if row['Policy Number'][:2] == 'IQ': # was QBE Excess now HU
        return 'HU'
    if row['Policy Number'][:2] == 'IC': # was FL-Admitted now HU
        return 'HU'
    if row['Policy Number'][5] == '0':
        return 'EQ'
    if row['Policy Number'][5] == '6':
        return 'HU'
    if row['Policy Number'][5] == '8': # was CGL now HU
        return 'HU'
    if row['Policy Number'][5] == '9': # was AOP now HU
        return 'HU'

df['Peril'] = df.apply(lambda row: peril(row), axis=1)

df['Peril'].value_counts()


# create Month column
df['Month-Year'] = pd.to_datetime(df['Policy Effective Date']).dt.to_period('M')
df['Month-Year'] = df['Month-Year'].dt.strftime('%b-%Y')

# create New/Renewal column
def new_renew (row):
    if row['Policy Number'][-1] != '0':
        return 'Renewal'
    else:
        return 'New'

df['New/Renewal'] = df.apply(lambda row: new_renew(row), axis=1)
# print(df['New/Renewal'].value_counts())
        


### check for duplicates ###
df['Check_pol'] = df['Policy Number'].str[:16]
# print(df['Check_pol'].nunique())
# print(df['Policy Number'].nunique())

unique = pd.DataFrame(df.groupby('Check_pol')["Policy Effective Date"].nunique()).reset_index()

# merge grouped back to main df
df = pd.merge(df, unique, how='left', on="Check_pol")
# get df of duplicates (unique policy dates = 2)
duplicates = df[df['Policy Effective Date_y']==2]

# difference b/w expiring and renewing      True=2018
duplicates.groupby([(duplicates["Policy Effective Date_x"] >= '2018-01-01') & (duplicates["Policy Effective Date_x"] <= '2019-12-31')])["Carrier Limit"].sum()

# check values and drop duplicates from the df by finding rows with PED_y ==2 AND PED <= '2018-12-31'
duplicates.groupby(duplicates['Policy Effective Date_x'].dt.year)['Policy Effective Date_y'].value_counts()

df = df.drop(df[(df["Policy Effective Date_y"] == 2) & (df["Policy Effective Date_x"] <= '2019-12-31')].index)        
        
# get expiring/old pol number
def getExpNumber(row):
    if row['New/Renewal'] == 'Renewal':
        val = row['Policy Number'][-2:]
        val = int(val) - 1
        val = '0' + str(val)
        base_pol = row['Policy Number'][:-2]
        old_pol = base_pol + val
        return old_pol
    else:
        return 'None'

df['Old Policy Number'] = df.apply(getExpNumber, axis=1)

#%% generate some charts as tests
df2 = df.groupby(['Carrier Display Name','Segment'])[['Carrier Limit']].sum()
df2.reset_index(inplace=True)
df2 = df2[df2['Segment'] != 'HBU']
# graph
fig = px.bar(df2, x='Carrier Display Name', y='Carrier Limit', hover_data=['Segment'],color='Carrier Limit')

df3 = df[df['Carrier Display Name'] == 'Syndicate 4242']
df3 = df3.groupby('Policy Effective Date_x')[['Carrier Limit']].sum()
df3.reset_index(inplace=True)
fig2 = px.line(df3, x='Policy Effective Date_x',
               y='Carrier Limit')

df4 = df.groupby(['Carrier Display Name','Segment'])[['Carrier Limit']].sum()
df4.reset_index(inplace=True)
df4 = df4[df4['Segment'] != 'HBU']
df4['Carrier Limit'] = df4['Carrier Limit'].apply(lambda x : "{:,}".format(x))

df5 = df[df['Segment'] != 'HBU']
df5['Month-Year'] = pd.to_datetime(df5['Month-Year'])
summary_pivot = pd.pivot_table(data=df5, values='Carrier Limit', index=['Carrier Display Name','Segment'],
                        columns='Month-Year', aggfunc=np.sum, fill_value=0, margins=True)
summary_pivot.reset_index(inplace=True)

old_colNames = list(summary_pivot.columns[2:-1])
new_colNames = []
for i in summary_pivot.columns[2:-1]:
    i = i.strftime("%m-%Y")
    new_colNames.append(i)
    
col_rename_dict = {i:j for i,j in zip(old_colNames,new_colNames)}
summary_pivot.rename(columns=col_rename_dict, inplace=True)

summary_pivot.columns = summary_pivot.columns.astype(str)




#%% 
# CARRIER TAB
df_carrier = df
df_carrier['Carrier Limit'] = df_carrier['Carrier Limit'].astype(float)
df_carrier = df.pivot_table(values='Carrier Limit', index=['Month-Year','Carrier Display Name'],
                            aggfunc = sum, fill_value=0)
df_carrier.reset_index(inplace=True)
df_carrier['Month-Year'] = pd.to_datetime(df_carrier['Month-Year'])
df_carrier = df_carrier.sort_values(by='Month-Year')


fig_Carrier = px.line(df_carrier, x='Month-Year', y='Carrier Limit', color='Carrier Display Name')
fig_Carrier.update_xaxes(rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                            ])))

# get sum of limit for each carrier by segment and macrozone
df_agg = df.groupby(['Carrier Display Name','Segment','Macrozone','Peril'])[['Carrier Limit']].sum()
df_agg.reset_index(inplace=True)
df_agg['Carrier Limit'] = df_agg['Carrier Limit'].apply(lambda x : "{:,}".format(x))
# create watch zones column
watch_zones = ['CA Gtr Los Angeles','CA Gtr San Francisco','CA N Central Coast','CA N Coast','WA Washington',
               'OR Oregon','FL Tri County','FL Panhandle','FL Southwest','FL Inland','FL West',
               'FL East Coast','TX N Texas']
df_agg['Watch Zone'] = df_agg['Macrozone'].apply(lambda x: 'Yes' if x in watch_zones else 'No')

available_carriers = df_agg['Carrier Display Name'].unique()
available_segs = df_agg['Segment'].unique()
available_mz = df_agg['Macrozone'].unique()
available_wz = df_agg['Watch Zone'].unique()

#%%
# MACROZONE TAB
watch_zones = ['CA Gtr Los Angeles','CA Gtr San Francisco','CA N Central Coast','CA N Coast','WA Washington',
               'OR Oregon','FL Tri County','FL Panhandle','FL Southwest','FL Inland','FL West',
               'FL East Coast','TX N Texas']
df_macrozone = df[(df['Macrozone'].isin(watch_zones)) & (df['Segment'] != 'HBU')]
df_macrozone = df_macrozone.groupby(['Macrozone','Microzone','Carrier Display Name','Segment'])[['Carrier Limit']].sum()
df_macrozone.reset_index(inplace=True)
fig_WZ = px.bar(df_macrozone, x='Macrozone', y='Carrier Limit', color='Carrier Display Name', hover_data=['Segment','Microzone'])


#%%

# APP LAYOUT & STRUCTURE

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    dcc.Tabs([
        dcc.Tab(label='All ICAT', children=[
        
            html.H1(children='Carrier Limit Tracking!'),
        
            html.Div(children='''
                Dash: A web application framework for Python. 
            '''),
            
            html.H2(children='Graph of Commercial Inforce Limit'),
            dcc.Graph(
                id='example-graph',
                figure=fig
            ),
            html.H3(children='Graph of s4242 Limit'),
            dcc.Graph(figure=fig2),
            
            html.H4(children='Commercial Carriers Limit'),
            
            #generate_table(df4),
            
            dash_table.DataTable(
                id='practice_carrier_table',
                columns=[{"name": i, "id": i} for i in summary_pivot.columns],
                data = summary_pivot.to_dict('records'),
                )
            #generate_table(df_carrier)
        ]),

        ## CARRIER TAB
        dcc.Tab(label='Carriers', children=[
        
            html.H1(children='Carrier Limit'),
            # dash_table.DataTable(
            #     id='practice_carrier_table',
            #     columns=[{"name": i, "id": i} for i in df_carrier.columns],
            #     data = df_carrier.to_dict('records'),
            #     )
            
            #generate_table(df_carrier)
            html.H2(children='Carrier Limit by Policy Inception Date'),
            html.Div(children='''Shows When Limit was Bound   ''' ),
            
            # graph of all carriers and their limit by Pol Inception date
            dcc.Graph(figure=fig_Carrier),
            
            # dropdown for carrier name
            html.Div(children='''Select Carrier(s)'''),
            dcc.Dropdown(
                id='carrier_aggs_dd',
                options=[{'label':i, 'value':i} for i in available_carriers],
                value=[],
                multi=True
                ),
            # dropdown for segment
            html.Div(children='''Select Segment(s)'''),
            dcc.Dropdown(
                id='carrier_aggs_dd_segment',
                options=[{'label':i, 'value':i} for i in available_segs],
                value=[],
                multi=True
                ),
            # dropdown for Watch Zones
            html.Div(children='''Filter to Watch Zones or Rest'''),
            dcc.Dropdown(
                id='carrier_aggs_dd_wz',
                options=[{'label':i, 'value':i} for i in available_wz],
                value=[],
                multi=True
                ),
            # dropdown for macrozone
            html.Div(children='''Select Macrozone(s)'''),
            dcc.Dropdown(
                id='carrier_aggs_dd_mz',
                options=[{'label':i, 'value':i} for i in available_mz],
                value=[],#'CA Gtr Los Angeles','FL Tri County', 'TX N Texas'],
                multi=True
                ),            
            # dash_table.DataTable(
            #     id='carrier_aggs',
            #     columns=[{"name": i, "id": i} for i in df_agg.columns],
            #     data=df_agg.to_dict('records'),
            #     )
            html.Div(children=''' '''), 
            dash_table.DataTable(
                id='carrier_aggs',
                columns =  [{"name": i, "id": i,} for i in (df_agg.columns)])
        ]),
        
        dcc.Tab(label='Macrozones', children=[
            html.H1(children='Limit by Macrozone'),
            
            html.Div(children=''' Limit in Watch Zones (PML Drivers) '''),
            
            dcc.Graph(figure=fig_WZ)
            
        ])
    ])
])

                   
 
                     
@app.callback(Output('carrier_aggs', 'data'), [Input('carrier_aggs_dd', 'value'), 
                                               Input('carrier_aggs_dd_segment', 'value'),
                                               Input('carrier_aggs_dd_mz', 'value'),
                                               Input('carrier_aggs_dd_wz', 'value')])
def update_rows(selected_carriers, selected_segs, selected_mz, selected_wz):
    carriers = list(selected_carriers)
    segs = list(selected_segs)
    mz = list(selected_mz)
    wz = list(selected_wz)
    
    if len(selected_carriers) >= 1:
        dff_agg = df_agg[df_agg['Carrier Display Name'].isin(carriers)]
    else:
        dff_agg = df_agg
        
    if len(selected_segs) >= 1:
        dfF_agg = dff_agg[dff_agg['Segment'].isin(segs)]
    else:
        dfF_agg = dff_agg
    
    if len(selected_mz) >= 1:
        dFF_agg = dfF_agg[dfF_agg['Macrozone'].isin(mz)]
    else: 
        dFF_agg = dfF_agg
    
    if len(selected_wz) >= 1:
        dFFF_agg = dFF_agg[dFF_agg['Watch Zone'].isin(wz)]
    else:
        dFFF_agg = dFF_agg
        
    return dFFF_agg.to_dict('records')

                     
if __name__ == '__main__':
    app.run_server(debug=True)
    
# colnames='Syndicate 2288'
# test = df[df['Carrier Display Name'].isin(colnames)]  
# df4 = df[df['Segment'] != 'HBU']
# df4['Month-Year'] = pd.to_datetime(df4['Month-Year'])
# summary_pivot = pd.pivot_table(data=df4, values='Carrier Limit', index=['Carrier Display Name','Segment'],
#                         columns='Month-Year', aggfunc=np.sum, fill_value=0)

# print(summary_pivot.style.format('{0:,.2f}').hide_index())

# summary_pivot.info()
# df4.info()




