from dash import dcc,html
from numpy import dtype
import pandas as pd
import dash_bootstrap_components as dbc
 
acres = pd.read_csv('data/acrs.csv')
labels = list(acres['acr_lable'])
values = list(acres['acr_size'])
options = []
options.append( {"label":"All","value":200}) # 200 means just okay and take ths for all card pricing linsting.

label = 'label'
value = 'value'
for i in range(len(acres)):
    options.append({"label":labels[i],"value":values[i]})

price_mark_radio_button =  dcc.RadioItems( id='acre-mark-radio',
        options = options,
       
        value =  .50   # default selected value firstime
        
    )
 
 




range_slider1 = dcc.RangeSlider(           # Acre Size      
        id='range-slider1',  
        min=0, max=80, step=0.125,
        marks={ 1: '1',  1.25:'1.25 ' , 2.5:'2.5', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        
        # Need to control this value by radio button  [value+margin slider value]
        # value=[.125,80],
        value= [0,80], #acre_range_between,


        tooltip={"placement": "bottom", "always_visible": True}
        
    )
percent_slider1 = dcc.Slider(           # Acre Size      
        id='percent-slider1',  
        min=.2, max=1,value=0.25,
         step=0.01,
        # marks={0: '0', 2.5: '2.5'},
        #marks={0.125: '0.125 Acr', 0.25: '0.25 Acr',  0.50: '0.50 Acr', 1: '1Acr',  1.25:'1.25 Acr' , 2.5:'2.5 Acr', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        marks={ .20: '20%', .25: '25%',  .30:'30%' ,.35:'35%', .40:'40%', .50: '50%', .60: '60%',  .70:'70%' ,.80:'80%', .90:'90%', 1: '100%'},
        

        tooltip={"placement": "bottom", "always_visible": True}
        
    )


price_slider1 = dcc.RangeSlider(           # Acre Size      
        id='price-slider1',  
        min=1000, max=500000, step=1000,
        # marks={0: '0', 2.5: '2.5'},
        
        marks={ 1000: '1k',  5000:'5k ' , 10000:'10k',20000:'20k', 50000:'50k', 100000:'100k', 250000:'250k'},
        
        # Need to control this value or acre range value for acre slider 
        value=[1000, 500000],
        tooltip={"placement": "bottom", "always_visible": True}
        
    )

Box_plot1 = dbc.Container(
    
    dcc.Graph(id= "marginal-plot1",config= {'displaylogo': False})
)










acre_margin_slider = dcc.Slider(           # Acre Size      
        id='acre-margin-slider',  
        min=0, max=5,value=0.01,
         step=0.01,
        # marks={0: '0', 2.5: '2.5'},
        #marks={0.125: '0.125 Acr', 0.25: '0.25 Acr',  0.50: '0.50 Acr', 1: '1Acr',  1.25:'1.25 Acr' , 2.5:'2.5 Acr', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        # marks={ .20: '20%', .25: '25%',  .30:'30%' ,.35:'35%', .40:'40%', .50: '50%', .60: '60%',  .70:'70%' ,.80:'80%', .90:'90%', 1: '100%'},
        

        tooltip={"placement": "bottom", "always_visible": True}
        
    )




#____________________________Acre Slider__________________________________



def acre_range_slider(min=0.11, max=0.2, step=0.01):

    dcc.RangeSlider(           # Acre Size      
        id='acre-range-slider',  
        min=min, max=max, step=step,
        # marks={0: '0', 2.5: '2.5'},
        #marks={0.125: '0.125 Acr', 0.25: '0.25 Acr',  0.50: '0.50 Acr', 1: '1Acr',  1.25:'1.25 Acr' , 2.5:'2.5 Acr', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        marks={ 1: '1',  1.25:'1.25 ' , 2.5:'2.5', 5:'5 Acr', 10:'10 Acr', 20:'20 Acr', 40:'40 Acr', 80: '80 acr'},
        
        #marks={i: '{}'.format(2 ** i) for i in range(7)},
        value=[0.12, 0.13],
        tooltip={"placement": "bottom", "always_visible": True}
        
    )

#___________________________________Map_______________________________________

map_graph = dbc.Container(
    
    dcc.Graph(id= "map-plot", config={#'position':center,#isplaylogo': False}) # config -> https://swdevnotes.com/python/2020/plotly_customise_toolbar/
                 "displaylogo": False,
                 'modeBarButtonsToRemove': [
              
                    'drawcircle',
                     'select2d',
                    #  'lasso2d',
                    'eraseshape'
                    
                     'autoScale2d',
                     'hoverClosestCartesian',
                     'hoverCompareCartesian'],

                     'modeBarButtonsToAdd':['toggleSpikelines',
                                           
                                       ]
                     })
)


map_graph1 = dbc.Container(
    
    dcc.Graph(id= "map-plot1", config={#'position':center,#isplaylogo': False}) # config -> https://swdevnotes.com/python/2020/plotly_customise_toolbar/
                 "displaylogo": False,
                 'modeBarButtonsToRemove': [
              
                    'drawcircle',
                     'select2d',
                    #  'lasso2d',
                    'eraseshape'
                    
                     'autoScale2d',
                     'hoverClosestCartesian',
                     'hoverCompareCartesian'],

                     'modeBarButtonsToAdd':['toggleSpikelines',
                                           
                                       ]
                     })
)



def price_range_slider(min=800, max=10000000, step=500):
    dcc.RangeSlider(           # Acre Size      
        id='price-range-slider',  
        min=min, max=max, step=step,
        # marks={0: '0', 1000000: '1000000'},
        marks={ 1000: '1k',  5000:'5k ' , 10000:'10k',20000:'20k', 50000:'50k', 100000:'100k', 250000:'250k'},
        

        value=[0, 1000000],
        tooltip={"placement": "bottom", "always_visible": True}
    )







def draw_radio_price_card(id, title_, className='text-info fw-bold'):
    return html.Div([
        dbc.Card(
             
            dbc.CardBody([
                html.H4(title_, className='text-info card-title fw-bold' ),
                html.Div([
                    html.H2(id=id, className=className),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])
 

card_radio =   dbc.Card(
        dbc.CardBody([
            dbc.Row([
                
              

                dbc.Col([
                    draw_radio_price_card('q1_radio','Quartile 1 Price:')
                ], width=2),

                dbc.Col([
                    draw_radio_price_card('q2_radio', 'Quartile 3 Price:')
                ], width=2),

                  dbc.Col([
                    draw_radio_price_card('avg_radio', 'Average Price')
                ], width=2),

                dbc.Col([
                    draw_radio_price_card('avg_4_radio','Percentile Price', className='text-success fw-bold'),
                ], width=3),
                dbc.Col([
                    draw_radio_price_card('acre-between_radio', 'Acre selected between:','text-success fw-bold')
                ], width=3),
            ], align='center'), 
          
                 
        ]), color = 'dark'
    )
 

map_and_box_plot_row = [dbc.Row([
        dbc.Col( [

                          # must double bracket
                # dbc.Row(range_slider1,className= 'text-warning fw-bold'),
  # should be acre
                
                dbc.Row(map_graph1),  # should be 
                dbc.Row(price_slider1),
                html.Br(),
                # dbc.Row([ml_radio_button],className= 'text-info fw-bold')
    ], width= 8),

      
    
        # dbc.Col(None, width= 2),

        dbc.Col([dbc.Row(percent_slider1),
        # html.Br(),

        # dbc.Row(drop_down),  # should be 
        dbc.Col(Box_plot1),

        html.Br(),
        # dbc.Row(price_point_crd),
        


        ]
        , width=4)
        

    ]),
    
    
]   
    

    
    
    
    
    
import plotly.express as px

def marginal_distribution_plot1(df):
    
    
    fig = px.density_heatmap(df, x="Lot_Size", y="Price", marginal_x="violin", marginal_y="box")

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



def data_filter_by__slider1(df, low, high, price_range):     # always need to call first this function for each callback to get filter data by slider range.
     
   
  
    
    mask = (df['Lot_Size'] > low) & (df['Lot_Size'] < high)
    df1 = df.loc[mask]                                                   # never reassign   df = df[ something]  df1 or something else
    # if price_range:
    low_price, high_price = price_range
                                                                        # make sure to use df1 to right side
    mask = (df1['Price'] > low_price) & (df1['Price'] < high_price)
    # df2 = df1.l[mask] 
    df2 = df1.loc[mask] 

                                                
    return df2

def map_plt1(df):
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