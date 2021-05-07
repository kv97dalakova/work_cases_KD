#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
from scipy.stats import zscore
import plotly.graph_objects as go


# In[2]:


ds = pd.read_csv("DataSample.csv", index_col = False)
ds


# In[3]:


poi = pd.read_csv("POIList.csv", index_col = False)
poi

#POI1 and POI2 are duplicates


# ### Task 1: Cleanup

# In[4]:


#exploring POI dataset: changing inconvenient column names

poi = poi.rename(columns = {' Latitude':'Latitude'}, inplace = False)
poi.info


# In[5]:


#removing the duplicats
poi.drop(poi.index[1], inplace = True)
poi


# In[6]:


#exploring DataSample dataset
pd.isnull(ds).sum()


# In[7]:


#exploring DataSample dataset vol 2
ds.info()


# In[8]:


#exploring DataSample dataset vol 3: renaming inconveniently named indexes
ds = ds.rename(columns = {' TimeSt':'timest'}, inplace = False)
ds


# In[9]:


#exploring DataSample dataset vol 4: finding duplicates 
ds.duplicated(subset = ['timest', 'Latitude', 'Longitude'], keep = 'first').sum()


# In[10]:


#exploring DataSample dataset vol 4: eliminating duplicates
ds_dd = ds.drop_duplicates(subset = ['timest', 'Latitude', 'Longitude'], keep = 'last')
ds_dd


# ### Task 2: Label

# In[11]:


#geo: transforming 'Latitude' and 'Longitude' to 'geometry'

#Option 1
#gdf_ds_dd = gpd.GeoDataFrame(ds_dd, geometry=gpd.points_from_xy(ds_dd['Longitude'], ds_dd['Latitude']))
#gdf_poi = gpd.GeoDataFrame(poi, geometry=gpd.points_from_xy(poi['Longitude'], poi['Latitude']))

#Option 2
def new_gdf(df, x="Longitude", y="Latitude"):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]))


# In[12]:


gdf_ds_dd = new_gdf(ds_dd)
gdf_ds_dd


# In[13]:


gdf_poi = new_gdf(poi)
gdf_poi


# In[14]:


# Assigning nearest values 

gdf_ds_dd['key'] = gdf_ds_dd.apply(lambda x: ((x['Latitude'] - gdf_poi['Latitude']).abs() + (x['Longitude'] - gdf_poi['Longitude']).abs()).idxmin(), axis=1)
gdf_ds_dd = pd.merge(gdf_ds_dd, gdf_poi[['POIID','geometry','Latitude','Longitude']], how='left', left_on='key', right_index=True).drop('key', axis=1)
gdf_ds_dd


# In[15]:


gdf_ds_dd.rename(columns={'_ID' : 'id', 'POIID' : 'poi_name'}, inplace=True)


# ### Task 3: Analysis

# In[16]:


# Measuring the distance between DataSample locations and POI locations using geopy

def distance(row):
    point1 = (row["Latitude_x"], row["Longitude_x"])
    point2 = (row["Latitude_y"], row["Longitude_y"])
    try: 
        return (geodesic(point1, point2).km) 
    except ValueError: return np.nan  


# In[17]:


gdf_ds_dd['km'] = gdf_ds_dd.apply(lambda row: distance(row), axis = 1 )


# In[18]:


#Looking at the outliers
sns.boxplot(x=gdf_ds_dd['km'])


# In[19]:


from scipy import stats

z = np.abs(stats.zscore(gdf_ds_dd['km']))
print(z)


# In[20]:


threshold = 3
print(np.where(z > 3))


# In[21]:


# Adding a z-score column to identify the outliers
gdf_ds_dd['z_score'] = zscore(gdf_ds_dd['km'])
gdf_ds_dd


# In[22]:


filt1 = gdf_ds_dd['z_score'] < 3
gdf_ds_dd[filt1].count()


# In[23]:


gdf_ds_dd_f = gdf_ds_dd[filt1]
gdf_ds_dd_f


# In[24]:


#Calculating of standard deviation, mean and radius
ds_grouped = gdf_ds_dd_f.groupby(['poi_name', 'Latitude_y', 'Longitude_y']).agg(
    avrg_km = ('km', "mean"),
    st_d = ('km', 'std'),
    radius = ('km', 'max'),
    quantity = ('poi_name', 'count'))
ds_grouped.reset_index(inplace = True)

#calculating density
ds_grouped["density"] = ds_grouped["radius"].apply(lambda row: (row**2) * np.pi).round(2)
#ds_grouped["radius_m"] = (ds_grouped["radius"] * 1000).round(1)
ds_grouped


# In[26]:


# Creating a map to see what do we have #1
m = folium.Map([45.521629, -73.566024], zoom_start = 4, tiles='CartoDb dark_matter')

#Mapping POI
for i in range(0,len(ds_grouped)):
    folium.Circle(
      location=[ds_grouped.iloc[i]['Latitude_y'], ds_grouped.iloc[i]['Longitude_y']],
      popup=ds_grouped.iloc[i]['poi_name'],
      radius=float(ds_grouped.iloc[i]['radius'])*1000,
      color='darkblue',
      fill=True,
      fill_color='darkblue'
   ).add_to(m)

#Mapping neighbours
for i in range(0,len(gdf_ds_dd_f)):
    folium.Circle(
      location=[gdf_ds_dd_f.iloc[i]['Latitude_x'], gdf_ds_dd_f.iloc[i]['Longitude_x']],
      popup=gdf_ds_dd_f.iloc[i]['poi_name'],
      radius=50,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)


# Show the map again
m


# #### Old visual: Mapping connections

# In[27]:


# Making linear connections & mapping them out
gdf_ds_dd_f['line'] = gdf_ds_dd_f.apply(lambda row: LineString([row['geometry_x'], row['geometry_y']]), axis=1)
lines_gdf = gdf_ds_dd_f[['id', 'poi_name', 'line']].set_geometry('line')

lines_gdf.set_crs("EPSG:4326")


# In[28]:


# Transforming to the acceptable coordinate system
lines_gdf.crs = 'epsg:4326'


# In[29]:


# Creating a map to see what do we have
m = folium.Map([45.521629, -73.566024], zoom_start = 4, tiles='CartoDb dark_matter')
poi_m = zip(ds_grouped.Latitude_y, ds_grouped.Longitude_y)
ds_m = zip(gdf_ds_dd_f.Latitude_x, gdf_ds_dd_f.Longitude_x)

for location in ds_m:
    folium.CircleMarker(location=location, 
        color='yellow', radius=2).add_to(m)
    
for location in poi_m:
    folium.CircleMarker(location=location, 
        color='blue', radius=50, fill = True).add_to(m)    

for location in poi_m:
    folium.Circle(location=location, 
        color='blue', radius=1, fill = True).add_to(m)

folium.GeoJson(data = lines_gdf).add_to(m)
folium.LayerControl().add_to(m)
m

#linear elements can be removed via layer selection on the map


# ### The End

# In[ ]:




