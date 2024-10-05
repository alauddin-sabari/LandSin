# from django import db
#./Documents/Patric/Deployment/AWS/heroku/price_radio_and_all._main.py

from time import process_time_ns
from dash import Dash, dcc, html, Input, Output
import os

import all_price_card









from atexit import register
import dash
import re
# from this import d
from turtle import position, title
from typing import Tuple
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dash_table
from dash.dependencies import  State
from dash.exceptions import PreventUpdate
import pickle
import numpy as np

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = Dash(__name__, external_stylesheets=external_stylesheets)

app =dash.Dash(__name__,
title="Patrick",
                 assets_folder ="static",
                 assets_url_path="static",
          external_stylesheets=[dbc.themes.SLATE])#[dbc.themes.BOOTSTRAP])
server = app.server


app.config.suppress_callback_exceptions=True         

app.title = 'Patrick Elder'

# df = pd.read_csv('CA_low_to_high_all.csv')
df = pd.read_csv('data/CA_low_to_high_all_new_geocoding_1_by_google_api.csv')
#######################-----------------------Scatter Map+other-------------------------###########################



percent_slider = dcc.Slider(           # Acre Size      
        id='percent-slider',  
        min=.2, max=1,value=0.25,
         step=0.01,
        # marks={0: '0', 2.5: '2.5'},
        #marks={0.125: '0.125 Acr', 0.25: '0.25 Acr',  0.50: '0.50 Acr', 1: '1Acr',  1.25:'1.25 Acr' , 2.5:'2.5 Acr', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        marks={ .20: '20%', .25: '25%',  .30:'30%' ,.35:'35%', .40:'40%', .50: '50%', .60: '60%',  .70:'70%' ,.80:'80%', .90:'90%', 1: '100%'},
        

        tooltip={"placement": "bottom", "always_visible": True}
        
    )



acre_range_between =[]

range_slider = dcc.RangeSlider(           # Acre Size      
        id='range-slider',  
        min=0, max=80, step=0.125,
        marks={ 1: '1',  1.25:'1.25 ' , 2.5:'2.5', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        
        # Need to control this value by radio button  [value+margin slider value]
        # value=[.125,80],
        value= [0,80], #acre_range_between,


        tooltip={"placement": "bottom", "always_visible": True}
        
    )


ml_radio_button = html.Div([
    
    html.Label('Select ML model'),
    html.Br(),
  
    
    dcc.RadioItems( id='ml-radio',
        options = [
                    {'label' : 'Linear', 'value':'lm'},   
                    {'label' : 'Pl-2D', 'value':'plm2d'},
                    {'label' : 'Pl-3D', 'value':'plm3d'},
                    {'label' : 'Pl-4D', 'value':'plm4d'}
                    
            
        ],
        value = 'lm'   # default selected value firstime
        
    ),
    html.Div(id='results-ml')
])


price_slider = dcc.RangeSlider(           # Acre Size      
        id='price-slider',  
        min=1000, max=500000, step=1000,
        # marks={0: '0', 2.5: '2.5'},
        
        marks={ 1000: '1k',  5000:'5k ' , 10000:'10k',20000:'20k', 50000:'50k', 100000:'100k', 250000:'250k'},
        
        # Need to control this value or acre range value for acre slider 
        value=[1000, 250000],
        tooltip={"placement": "bottom", "always_visible": True}
        
    )



#___________________________________Drop-Down_______________________________________
#https://ontheworldmap.com/usa/state/california/map-of-california-and-nevada.html states and cities

drop_down = html.Div([
                    dcc.Dropdown(id='data-set-chosen', multi=False, value='California',
                     options=[{'label':'California', 'value':'California'},
                              {'label':'Nevada', 'value':'California'},
                              {'label':'Arizona', 'value':'Arizona'}])
    ], className='row', style={'width':'50%'}),                                                 #dcc.Dropdown(['California', 'Nevada', 'SF'], 'NYC', id='data-set-chosen')
#___________________________________Box_Plot___________________________________

Box_plot = dbc.Container(
    
    dcc.Graph(id= "box-plot",config= {'displaylogo': False})
)
#___________________________________Reg_Plot______________________________________ 

reg_plot =dbc.Container(
    
    dcc.Graph(id= "reg-plots",config= {'displaylogo': False})
)

# ___________________________________Table_______________________________________

            # #______________________Polygon________________

filter_table = html.Div([
    # html.Button("Download Excel", id="btn_xlsx"),

    dbc.Button(id = 'btn_xlsx',
            children=[html.I(className="fa fa-download mr-1"), "Save  filtered data by slider"],
            color="info",
            className="mt-1"),
            #____________________________________________________________showing data table below_______________
 #_New__________   
    html.Div([
        html.Div(id='table-placeholder', children=[])
    ], className='row'),
# #______________________Polygon________________
dbc.Button(id = 'btn_xlsx-polygon',
            children=[html.I(className="fa fa-download mr-1"), "Save Polygon data"],
            color="info",
            className="mt-1"),
 html.Div([
        html.Div(id='table-placeholder-polygon', children=[])
    ], className='row'),
#     dash_table.DataTable( id ='polygon-table',

#                         columns = [{'name':i, 'id':i}  for i in df_table.columns],
#                         fixed_rows = {'headers': True, 'data': 1},
#                         data = df_table.to_dict('records'),
#     style_cell={'minWidth': 95, 'width': 95, 'maxWidth':195},
#     style_table={'height': '300px', 'overflowY': 'auto'},
#                     ),
            #___________________________________________________________ending table show_______

    
    dcc.Download(id="download-dataframe-xlsx"),
    dcc.Download(id="download-dataframe-xlsx-polygon"),

])


marginal_distribution_plot =dbc.Container(
    
    dcc.Graph(id= "marginal-distribution-plots",config= {'displaylogo': False})
)

 
#######################-----------------------End-------------------------###########################

def drawText(id, title_):
    return html.Div([
        dbc.Card(
             
            dbc.CardBody([
                html.H4(title_, className='card-title fw-bold' ),
                html.Div([
                    html.H2(id=id, className='text-warning'),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])

crd = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText('avg', 'Average Price')
                ], width=2),
                dbc.Col([
                    all_price_card.draw_radio_price_card('avg_4','Percentile Price')
                ], width=2),
                dbc.Col([
                    drawText('q1','Quartile 1 Price:')
                ], width=2),
                dbc.Col([
                    drawText('q2', 'Quartile 3 Price:')
                ], width=3),
                dbc.Col([
                    drawText('acre-between', 'Acre selected between:')
                ], width=3),
            ], align='center'), 
                 
        ]), color = 'dark'
    )
])

#1: '1',  1.25:'1.25' , 2.5:'2.5', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'}, # .125, .25, .5,  1   acres

price_point_crd = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText('a-125', '.125 acres : ')
                ]),
                dbc.Col([
                    drawText('a-25','.25 acres  :')
                ] )
            ]),
             dbc.Row([
                dbc.Col([
                    drawText('a-5','.5 acres :')
                ] ),
                dbc.Col([
                    drawText('a1', '1 acres :')
                ] )
            ]),
             dbc.Row([
               
                dbc.Col([
                    drawText('a1-25', '1.25 acres  :')
                ]),
                dbc.Col([drawText('a2-5', '2.5 acres  :')]),
            ], align='center'), 
      
        dbc.Row([
            
            dbc.Col([drawText('a5', '5 acres  :')]   ),
            dbc.Col([drawText('a10', '10 acres  :')] ),

        ]),
        dbc.Row([
            dbc.Col([drawText('a20', '20 acres  :')] ),
            dbc.Col([drawText('a40', '40 acres  :')] ),

        ]),
        dbc.Row([
            dbc.Col([drawText('a80', '80 acres  :')] ),
            dbc.Col([drawText('all', 'Avg on Polygon:')] ),

        ]),
        ]), color = 'dark'
    )
])
#######################-----------------------App Layout-------------------------###########################
price_mark_radio_button = all_price_card.price_mark_radio_button

 
radio = [     dbc.Row([
         dbc.Col([ all_price_card.card_radio], width=12),
         
      
        ]),

        all_price_card.map_and_box_plot_row[0],
] 

# import all_price_cards_box_percentage
app.layout = html.Div([
    html.H1(html.Mark("Patrick's Property listing App"), className= 'opacity-100 bg-secondary p-3 m-2 text-info text-center  fw-bold rounded'),
 
#  dbc.Row([single_price_pooint_slider ,
#                  html.Div(id='slider-output') ]),
    dbc.Row([
       dbc.Col( html.Label('Select Acre Size:', className='text-info fw-bold'), width=1),

        dbc.Col([all_price_card.price_mark_radio_button,  dcc.Checklist(id='show-all-check', options=[ 'Show all ', 'CA ', 'Nevada'])],className= 'text-info fw-bold',width=6),

       dbc.Col( html.Label('Acre margin:', className='text-warning fw-bold'), width=1),
        dbc.Col([all_price_card.acre_margin_slider],className= 'fw-bold',width=4),

        # dbc.Col([dcc.Checklist(['Show all'])],className= 'text-info fw-bold',width=1),

]),
        html.Div(radio),
        # percent_price_id(all_price_cards_box_percentage.cards_box_percentage( 'box_id', 'percent_price_id', 'map_plot_id')),
        html.Div(id='map-plot-row'),
 
        # dbc.Row([
        #  dbc.Col([ all_price_card.card_radio], width=12),
         
      
        # ]),

        # all_price_card.map_and_box_plot_row[0],


     

    crd,  #card all tile above

    dbc.Row([
        dbc.Col( [

                          # must double bracket
                dbc.Row(range_slider,className= 'text-warning fw-bold'),
  # should be acre
                
                dbc.Row(all_price_card.map_graph),  # should be 
                dbc.Row(price_slider),
                html.Br(),
                dbc.Row([ml_radio_button],className= 'text-info fw-bold')
    ], width= 8),

      
    
        # dbc.Col(None, width= 2),

        # dbc.Col(Box_plot, width=4)
        dbc.Col([dbc.Row(percent_slider),
        html.Br(),
        dbc.Row(drop_down),  # should be 
        html.Br(),
        dbc.Row(price_point_crd),
        


        ]
        , width=4)
        

    ]),
        dbc.Row(reg_plot),


    dbc.Row([

      
         # Clustering  bottom right side
        dbc.Col(marginal_distribution_plot, width=5),
        dbc.Col(Box_plot, width=2),

        dbc.Col( filter_table, width= 5)

        

    ]),

     # step- 1
    #__________________________Store Data in here______________________________

    dcc.Store( id = 'store-slider-data', data = [], storage_type = 'memory'),
    dcc.Store( id = 'store-polygon-data', data = [], storage_type = 'memory'),
    dcc.Store( id = 'store-dropdown-data', data = [], storage_type = 'memory'),


     
     


])



#______________________________________________________




def map_plt(df):
    # px.set_mapbox_access_token(open(".mapbox_token").read())
    fig = px.scatter_mapbox(
        # df,  lat='latitude', lon='longitude',     color="Price", size="Lot_Size",
        df,  lat='lat', lon='long',     color="Price", size="Lot_Size",

                #   color_continuous_scale=px.colors.cyclical.IceFire,    
        title = "US    Property Listing",

        # hover_data=1,
                        hover_name='Address',
#                         text='Address',
        
        zoom=5)#, mapbox_style='open-street-map')
    fig.update_layout(
        title = "US    Property Listing",
    autosize = True,
    # width = 1350,
    height = 750
)
    # fig.update_layout(
    #     mapbox_style="white-bg",
    #     mapbox_layers=[
    #         {
    #             "below": 'traces',
    #             "sourcetype": "raster",
    #             "sourceattribution": "United States Geological Survey",
    #             "source": [
    #                 "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
    #             ]
    #         }
    #     ])
    fig.update_layout(margin={"r":0,"t":25,"l":5,"b":0}),
    
    fig.update_layout(mapbox_style="open-street-map") 

    return fig
def data_filter_by__slider(df, acr_range, price_range):     # always need to call first this function for each callback to get filter data by slider range.
     
   
    low, high = acr_range # [acre_range_between[0]]
    # print('------------\n______________acre_range_between', [acre_range_between[0]])
    # print('------------\n______________acre_range ', acr_range )

    
    mask = (df['Lot_Size'] > low) & (df['Lot_Size'] < high)
    df1 = df.loc[mask]                                                   # never reassign   df = df[ something]  df1 or something else
    # if price_range:
    low_price, high_price = price_range
                                                                        # make sure to use df1 to right side
    mask = (df1['Price'] > low_price) & (df1['Price'] < high_price)
    # df2 = df1.l[mask] 
    df2 = df1.loc[mask] 

                                                
    return df2


@app.callback(
    Output("map-plot", "figure"), 
    Input("range-slider", "value"),
    Input("price-slider", "value")
    )

    

def update_map_chart(acr_range,price_range):
    global df
    df1 = data_filter_by__slider(df, acr_range, price_range) 
    m = map_plt(df1)
    return m

#__________________________________Each Card Price_______________________________________


def each_price_point(df, percent_range, radio_ml=None):
    import pickle 
    import pandas as pd
    import plotly.express as px

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures

    import plotly.graph_objects as go
    



    # df = df_no_outlier.copy()


    df  = df[df["Lot_Size"]<=80]


    X =  df[['Lot_Size']].values
    X_train, X_test, y_train, y_test = train_test_split(X, df.Price,  random_state=0)

    def format_coefs(coefs):
        equation_list = [f"{coef}x^{i}" for i, coef in enumerate(coefs)]
        equation = "$" +  " + ".join(equation_list) + "$"

        replace_map = {"x^0": "", "x^1": "x", '+ -': '- '}
        for old, new in replace_map.items():
            equation = equation.replace(old, new)

        return equation
    X = df[['Lot_Size']].values
    
    # X = df.total_bill.values.reshape(-1, 1)
    x_range = np.linspace(0, X.max(), 100).reshape(-1, 1)

    fig = px.scatter(df, x='Lot_Size', y='Price', opacity=0.65)
    fig.update_layout(
        title = "US    Property Listing",
    autosize = True,
    # width = 1350,
    height = 650
    )
    # ___________________________linear regression________________________________
    model_lr = LinearRegression()
    model_lr = model_lr.fit(X, df.Price)
    # ___________________________               ________________________________
    model_list = []
    for degree in [1, 2, 3, 4]:
        poly = PolynomialFeatures(degree)
        poly.fit(X)
        X_poly = poly.transform(X)
        x_range_poly = poly.transform(x_range)
        

        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, df.Price)
        model_list.append(model)
        # print('--\n X_range:', x_range_poly)
        y_poly = model.predict(x_range_poly)


        equation = format_coefs(model.coef_.round(2))
        fig.add_traces(go.Scatter(x=x_range.squeeze(), y=y_poly, name=equation))

    # print('df inside each price point', df)
    # print('--\n model list in here --------\n', model_list)
    if radio_ml == 'lm':
        degree = 1
    elif radio_ml == 'plm2d':    
        degree = 2
    elif radio_ml == 'plm3d':
        degree = 3   
    else:
        degree = 4
    
    import poly
    
    ypred = poly.poly_reg(df, degree=degree) 
    [a_125, a_25, a_50, a1, a1_25, a2_5, a5, a10, a20, a40, a80] = ypred
    a_125 = a_125*percent_range
    a_125 = f'${a_125 :.2f}'

    a_25 = a_25*percent_range
    a_25 = f'${a_25 :.2f}'

    a_50 = a_50*percent_range
    a_50 = f'${a_50 :.2f}'

    a1 = a1*percent_range
    a1 = f'${a1 :.2f}'

    a1_25 = a1_25*percent_range
    a1_25 = f'${a1_25 :.2f}'
    


    a2_5 = a2_5*percent_range
    a2_5 = f'${a2_5 :.2f}'

    a5 = a5*percent_range
    a5 = f'${a5 :.2f}'

    a10 = a10*percent_range
    a10 = f'${a10 :.2f}'

    a20 = a20*percent_range
    a20 = f'${a20 :.2f}'

    a40 = a40*percent_range
    a40 = f'${a40:.2f}'

    a80 = a80*percent_range
    a80 = f'${a80 :.2f}'
     

    ls_output = [a_125, a_25, a_50, a1, a1_25, a2_5, a5, a10, a20, a40, a80]
    ls_2d = [fig, ls_output]
    # print('--\n ls_2d-----------------------------------', ls_output)
    return ls_2d
@app.callback(
    Output('reg-plots', 'figure'),
    
    Output('a-125', 'children'),
    Output('a-25', 'children'),
    Output('a-5', 'children'),
    Output('a1', 'children'),
    Output('a1-25', 'children'),
    Output('a2-5', 'children'),
    Output('a5', 'children'),
    Output('a10', 'children'),
    Output('a20', 'children'),
    Output('a40', 'children'),
    Output('a80', 'children'),
    Output('all', 'children'),
 


                                      
 
    Input("percent-slider", "value"),
                                                                    
    Input(component_id="map-plot", component_property='selectedData') ,    # Extra input for drawing polygon.
    Input ('ml-radio', 'value')

)
 
def each_price_finder(percent_value, slct_data, radio_ml):
        global df
        # df = pd.read_csv('CA_low_to_high.csv') 


        if slct_data:                                                 
             number_of_points = len(slct_data['points'])
        if slct_data is  None or number_of_points==0 :                                               
           
            
            r = each_price_point(df,percent_value, radio_ml)
            # print('\n  ------ r inside each price point\n', r)
            polygon_price = df['Price'].mean()
            polygon_price = polygon_price*percent_value
            polygon_price = f'${polygon_price:.2f}'
            
            return r[0], r[1][0],r[1][1], r[1][2], r[1][3], r[1][4], r[1][5], r[1][6], r[1][7], r[1][8], r[1][9], r[1][10],polygon_price
        else:
            number_of_points = len(slct_data['points'])
            # print('|\n----------df inside each price finder ELSE ', df)

            ls = []

            ls_ad = []
            for i in  range(number_of_points):
                ls.append(slct_data['points'][i]['lon'])
                ls_ad.append(slct_data['points'][i]['hovertext'])


            df1= df[df['Price'].isin(ls)]
            df1= df[df['Address'].isin(ls_ad)]
            df1 = df1.dropna(subset=['Price'])
            
            r = each_price_point(df1,percent_value, radio_ml)
            # print('\n  ------ r inside each price point\n', r)
            polygon_price = df1['Price'].mean()
            polygon_price = polygon_price*percent_value
            polygon_price = f'${polygon_price:.2f}'

            return r[0],r[1][0],r[1][1], r[1][2], r[1][3], r[1][4], r[1][5], r[1][6], r[1][7], r[1][8], r[1][9], r[1][10] ,polygon_price














#___________________________________Table for download_______________________________________

#___________________________________Table for download_______________________________________
 
def data_filter_by__slider_and_storing(df, acr_range , price_range):     # always need to call first this function for each callback to get filter data by slider range.
     
   
    low, high = acr_range #  [acre_range_between[0]]
    
    mask = (df['Lot_Size'] > low) & (df['Lot_Size'] < high)
    df1 = df.loc[mask]                                                   # never reassign   df = df[ something]  df1 or something else
    # if price_range:
    low_price, high_price = price_range
                                                                        # make sure to use df1 to right side
    mask = (df1['Price'] > low_price) & (df1['Price'] < high_price)
    # df2 = df1.l[mask] 
    df2 = df1.loc[mask] 

                                                
    return df2
                #___________________________________storing table  by slider_______________________________________

@app.callback(
    Output('store-slider-data', 'data'),
    Input("range-slider", "value"),
    Input("price-slider", "value")
    # Input('data-set-chosen', 'value')
)
def store_the_data(acr_range,price_range):
    global df
    global dataset

  
    dataset = data_filter_by__slider(df, acr_range, price_range) 
                                                  

    return dataset.to_dict('records')


                #___________________________________storing filter table  by  Polygon_______________________________________

@app.callback(
    Output('store-polygon-data', 'data'),
    # Input("range-slider", "value"),
    # Input("price-slider", "value"),
    Input(component_id="map-plot", component_property='selectedData')     # Extra input for drawing polygon.

)
def store_the_data(slct_data):
    global df
    # global dataset
    dataset = df.copy()

   
    
    #df = dataset.copy()                                                # print('\n___________________inside store_the_data and dataset Shape\n',dataset.shape)
    if slct_data:                                                 
        number_of_points = len(slct_data['points'])

    if slct_data is  None or number_of_points==0 :
        
        
         
        dataset= df
        
        
    else:

        # print('---\n data read func\n####################',df)
       
        
        ls = []

        ls_ad = []
        for i in  range(number_of_points):
            ls.append(slct_data['points'][i]['lon'])
            ls_ad.append(slct_data['points'][i]['hovertext'])


        df1= df[df['Price'].isin(ls)]
        df1= df[df['Address'].isin(ls_ad)]
        df1 = df1.dropna(subset=['Price'])
        
       
        dataset = df1
    return dataset.to_dict('records')






















#___________________________________Showing slider filtered table______________________________________


@app.callback(
    Output('table-placeholder', 'children'),
    Input('store-slider-data', 'data'),
    # Input("btn_xlsx", "n_clicks"),

)
def create_graph1(data):
    dff = pd.DataFrame(data)
    # dff =df_d [['Address', 'Lot_Size', 'Price']]
    my_table = dash_table.DataTable( id='tbl',
        columns=[{"name": i, "id": i} for i in dff.columns],
        fixed_rows = {'headers': True, 'data': 1},
        data=dff.to_dict('records'),
        style_cell={'minWidth': 95, 'width': 95, 'maxWidth':195},
        style_table={'height': '300px', 'overflowY': 'auto'}
    )
    return my_table

                                    #________polygon_________

@app.callback(
    Output('table-placeholder-polygon', 'children'),
    Input('store-polygon-data', 'data'),
    # Input("btn_xlsx", "n_clicks"),

)
def create_graph1(data):
    dff = pd.DataFrame(data)
    # dff =df_d [['Address', 'Lot_Size', 'Price']]
    my_table = dash_table.DataTable( id='tbl-polygon',
        columns=[{"name": i, "id": i} for i in dff.columns],
        fixed_rows = {'headers': True, 'data': 1},
        data=dff.to_dict('records'),
        style_cell={'minWidth': 95, 'width': 95, 'maxWidth':195},
        style_table={'height': '300px', 'overflowY': 'auto'}
    )
    return my_table

#___________________________________Download-table_______________________________________
import io

@app.callback(
    Output("download-dataframe-xlsx", "data"),
    # Input('store-our-data', 'data'),
    
    Input("btn_xlsx", "n_clicks"),
    # Input('table-placeholder', 'data'),
    State ('tbl', 'data'),
    prevent_initial_call=True,
    
)
def func( n_clicks, table_data):
    
    # df = data
    
    df = pd.DataFrame.from_dict(table_data)
    # df = df[['Address', 'Lot_size', 'Price']]
    if not n_clicks:
        raise PreventUpdate
    download_buffer = io.StringIO()
    df.to_csv(download_buffer, index=True)
    download_buffer.seek(0)
    return dict(content=download_buffer.getvalue(), filename="patrick.csv")





@app.callback(
    Output("download-dataframe-xlsx-polygon", "data"),
    # Input('store-our-data', 'data'),
    
    Input("btn_xlsx-polygon", "n_clicks"),
    # Input('table-placeholder', 'data'),
    State ('tbl-polygon', 'data'),
    prevent_initial_call=True,
    
)
def func( n_clicks, table_data):
    
    # df = data
    
    df = pd.DataFrame.from_dict(table_data)
    # df = df[['Address', 'Lot_size', 'Price']]
    if not n_clicks:
        raise PreventUpdate
    download_buffer = io.StringIO()
    df.to_csv(download_buffer, index=True)
    download_buffer.seek(0)
    return dict(content=download_buffer.getvalue(), filename="patrick.csv")






#__________________________________marginal_distribution_plot_______________________________________
#__________________________________________________________________________
#


def marginal_distribution_plot(df):
    
    
    fig = px.density_heatmap(df, x="Lot_Size", y="Price", marginal_x="violin", marginal_y="violin")

    fig.update_layout(
    title  = {
                'text': "Marginal Distribution Plot ",
                'y':0.9,
                'x':0.5,
                'xanchor':'center',
                'yanchor': 'top'
            },
        xaxis = dict(title = "Lot Size"),
        yaxis = dict(title = " Lot count for each Price Range" ),
                    
                    
        autosize=True,
        # width=1200,
        height=725,)
    return fig

@app.callback(
    Output('marginal-distribution-plots', 'figure'),
    Input("range-slider", 'value'),
    Input("price-slider", "value")
    )
    
def reggs(acr_range, price_range):
    global df
    

    df1 = data_filter_by__slider(df, acr_range, price_range)
    len_df = len(df1)

    if len_df > 0:
        df1 =df1
    else:
        df1 = df
    b = marginal_distribution_plot(df1)
    return b


def top_bar_price(df1,percent_value=.25):
        # print('\n------------\n-------len df',len(df1))
        ln_df1 = len(df1)
        if ln_df1 ==0:# or number_of_points==0:
            return [0,0,0,0]
        else:
            avg = df1['Price'].mean()
            avg_= int(avg)
            avg_f= f'${avg_}'  #since later step it will be devide then get error

            # avg_by_4 =int(avg/4) 
            avg_by_4 = int( avg*percent_value)

            avg_by_4 = f'${avg_by_4}'
            Q1 = f'${int(df1.Price.quantile(0.25))}'
            
            # print(Q1)
            Q3 =f'${int(df1.Price.quantile(0.75))}' 
            ls = [avg_f, avg_by_4, Q1, Q3]
            return ls
@app.callback(
    Output('avg', 'children'),
    Output('avg_4', 'children'),
    Output('q1', 'children'),
    Output('q2', 'children'),
                                      
    Input("range-slider", "value"),
    Input("price-slider", "value"),
    Input("percent-slider", "value"),

                                                                    
    Input(component_id="map-plot", component_property='selectedData')     # Extra input for drawing polygon.
)
                            
def average_finder(acr_range,price_range, percent_value, slct_data):
    global df
    

    df1 = data_filter_by__slider(df, acr_range, price_range)
    # print('\n-------- percent_value ---------\n',percent_value)
    # df1 = pd.read_csv('drawing_polygon_without_outliers1.csv')    # we will use df when we don't wanna connect function with range slider.

    # print('\n----acre------df inside average callback functions\n', acr_range)
    if slct_data:                                                 
        number_of_points = len(slct_data['points'])

    if slct_data is  None or number_of_points==0 :
        
        
        r = top_bar_price(df1, percent_value)
        return r
        
        # return f'Average Price : ${int(avg)} \n  ||   targeted price (avg/4) : ${int(avg/4)}  [# Q1: ${Q1}   # Q3: ${Q3}]'

    else:

        # print('---\n data read func\n####################',df)
       
        
        ls = []

        ls_ad = []
        for i in  range(number_of_points):
            ls.append(slct_data['points'][i]['lon'])
            ls_ad.append(slct_data['points'][i]['hovertext'])


        df1= df[df['Price'].isin(ls)]
        df1= df[df['Address'].isin(ls_ad)]
        df1 = df1.dropna(subset=['Price'])
        
       
        r = top_bar_price(df1,percent_value)
        return r


#___________________________________Box-plot_______________________________________


def box_plt(df):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Box(

            y = df['Price'],

            name= 'Price',
            marker_color='Orange',
            boxmean=True,

        )
    )
    fig.update_layout(
        # title = "Property IQR",     this also valid
        title  = {
            'text': "Property's IQR",
            'y':0.9,
            'x':0.5,
            'xanchor':'center',
            'yanchor': 'top'
        },

        autosize=True,
        # width=1800,
        height=750,)
    fig.update_traces(orientation='v')
    return fig
@app.callback(
    Output('box-plot', 'figure'),
    Input("range-slider", 'value'),
    Input("price-slider", "value")

    # Input(component_id="scatter-plot", component_property='selectedData')
    
)
    
def set_display_children(acr_range, price_range):
    global df


    df1 = data_filter_by__slider(df, acr_range, price_range)
    
    b = box_plt(df1)
    return b



















#________________________________________________low high radio button_______________________________________________________

@app.callback(
    # Output("map-plot1", "figure"), 

    Output('acre-between', 'children'),
    Output('acre-between_radio', 'children'),


    Input('acre-mark-radio', 'value'),
    Input('acre-margin-slider', 'value'),

    # Input("range-slider1", "value"),

    Input("price-slider1", "value")



)


def update_acre_between(arce_radio_input, acre_margin,price_range):
    global acre_range_between
    
    # print(arce_radio_input, type(arce_radio_input))

    low = float(arce_radio_input+acre_margin)
    high = float(arce_radio_input-acre_margin)
    low_high = [ low , high ]
 
    acre_range_between.append(low_high)

    global df
    df1 = data_filter_by__slider(df, low_high, price_range) 
    m = map_plt(df1)
    ls = [   f'{arce_radio_input+acre_margin : .2f}-{arce_radio_input-acre_margin :.2f} acre.',f'{arce_radio_input+acre_margin : .2f}-{arce_radio_input-acre_margin :.2f} acre.']
    return ls 


# x = update_acre_between(1,1)
# print('--\nx',x)
#______________________________--Top bar price card and all------------

def acr_range_margin(arce_radio_input, acre_margin):
    low, high = 0,0

    global df
    global radio
    if acre_margin:
        low = float(arce_radio_input-acre_margin)
        high = float(arce_radio_input+acre_margin)
    if arce_radio_input ==200:
        low = 0
        high = 80
    acr_range = [ low , high ]
    return acr_range

@app.callback(
    # Output("adio-output", "children"), 

    Output("map-plot1", "figure"), 
    # Output("map-plot-row", "children"), 


# Output('avg_radio', 'children'),
#     Output('avg_4_radio', 'children'),
#     Output('q1_radio', 'children'),
#     Output('q2_radio', 'children'),

        Input('acre-mark-radio', 'value'),
    Input('acre-margin-slider', 'value'),

    Input("price-slider1", "value"),

     Input("percent-slider1", "value"),

                                                                    
    Input(component_id="map-plot", component_property='selectedData')     # Extr
    )

    

def update_map_chart(arce_radio_input, acre_margin,price_range,  percent_value, slct_data):

    global radio
    acr_range = acr_range_margin(arce_radio_input, acre_margin)
   
    df1 = data_filter_by__slider(df, acr_range, price_range) 

    m = all_price_card.map_plt1(df1)

    ls = [ radio,m]
 
    
    # ls = [m, r[0], r[1], r[2], r[3]]
    return m#, radio
    



@app.callback(
    Output('avg_radio', 'children'),
    Output('avg_4_radio', 'children'),
    Output('q1_radio', 'children'),
    Output('q2_radio', 'children'),

    
    Input('acre-mark-radio', 'value'),
    Input('acre-margin-slider', 'value'),
    Input("price-slider1", "value") ,     

    # Input("range-slider", "value"),
    # Input("price-slider", "value"),
    Input("percent-slider1", "value"),

                                                                    
    Input(component_id="map-plot1", component_property='selectedData')     # Extra input for drawing polygon.
)
                            
# def average_finder(acr_range,price_range, percent_value, slct_data):
def average_finder(arce_radio_input, acre_margin,price_range, percent_value, slct_data):

    acr_range = acr_range_margin(arce_radio_input, acre_margin)
    df1 = data_filter_by__slider(df, acr_range, price_range)
    
    if slct_data:                                                 
        number_of_points = len(slct_data['points'])

    if slct_data is  None or number_of_points==0 :
        
        
        r = top_bar_price(df1, percent_value)
        return r
        
        # return f'Average Price : ${int(avg)} \n  ||   targeted price (avg/4) : ${int(avg/4)}  [# Q1: ${Q1}   # Q3: ${Q3}]'

    else:

        # print('---\n data read func\n####################',df)
       
        
        ls = []

        ls_ad = []
        for i in  range(number_of_points):
            ls.append(slct_data['points'][i]['lon'])
            ls_ad.append(slct_data['points'][i]['hovertext'])


        df1= df[df['Price'].isin(ls)]
        df1= df[df['Address'].isin(ls_ad)]
        df1 = df1.dropna(subset=['Price'])
        
       
        r = top_bar_price(df1,percent_value)
        return r






@app.callback(
    Output('marginal-plot1', 'figure'),

    Input('acre-mark-radio', 'value'),
    Input('acre-margin-slider', 'value'),
    Input("price-slider1", "value") ,  
    )
    
def reggs(arce_radio_input, acre_margin, price_range):
    low, high = 0,0

    global df
    global radio
    if acre_margin:
        low = float(arce_radio_input-acre_margin)
        high = float(arce_radio_input+acre_margin)
    if arce_radio_input ==200:
        low = 0
        high = 80
    acr_range = [ low , high ]

    df1 = data_filter_by__slider(df, acr_range, price_range)

    len_df = len(df1)

    if len_df > 0:
        df1 =df1
    else:
        df1 = df
    b = all_price_card.marginal_distribution_plot1(df1)
    return b


 

if __name__ == '__main__':
    app.run_server(debug=True, port= 8082)
