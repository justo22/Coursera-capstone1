#!/usr/bin/env python
# coding: utf-8

# In[13]:



from bs4 import BeautifulSoup
import requests
import pandas as pd
 
import numpy as np
import time
import folium
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors


get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values


# In[14]:


url='https://en.wikipedia.org/wiki/Category:Neighbourhoods_in_Hyderabad,_India'
country = 'India',
city = 'Hyderabad'
geolocator = Nominatim()
page = requests.get(url)
parsedHTML = BeautifulSoup(page.text, 'html.parser')


# In[15]:


citiesContainers = parsedHTML.find_all('div', class_='mw-category-group')
cities = [];
c = 1;
for citiesContainer in citiesContainers:
    for li in citiesContainer.find_all('li'):
        cities.append(li.text.strip().split(',')[0])
print(len(cities))


# **collected all cities list**

# In[16]:


cities_df = pd.DataFrame({'Neighbourhoods': cities})
cities_df.head(10)


# **collect all geo coordiantes for each neighnourhood**

# In[17]:


latitudes = []
longitudes = []
errorCities = []
for neighbourhood in cities:
    print(neighbourhood)
    address = "{},{},{}".format(neighbourhood, city, country)
    retries = 3
    while (retries > 0):
        try:
            location = geolocator.geocode(address)
            latitude = location.latitude
            longitude = location.longitude
            latitudes.append(latitude)
            longitudes.append(longitude)
            retries = 0
            if neighbourhood in errorCities:
                errorCities.remove(neighbourhood)
        except Exception as e:
            print("ERROR: ", address, e)
            if not neighbourhood in errorCities:
                errorCities.append(neighbourhood)
            retries = retries - 1
            time.sleep(1)


# In[18]:


cities_with_geo_coordinates = filter(lambda city: not city in errorCities, cities)
cities_with_geo_coordinates = [city for city in cities_with_geo_coordinates]


# In[20]:


cities_with_geo_coordinates[0:10]


# **prepare a dataframe with required features**

# In[21]:


cities_df = pd.DataFrame({'Neighbourhood': cities_with_geo_coordinates, 'Latitude': latitudes, 'Longitude': longitudes})


# In[22]:


cities_df.head(10)


# In[23]:


cities_df.tail(10)


# In[24]:


location = geolocator.geocode(city)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of {} are {}, {}.'.format(city, latitude, longitude))


# **View map of all neighbourhoods**

# In[26]:



hyderabad_map = folium.Map(location=[latitude, longitude], zoom_start=10)
for neighbourhood, latiude, longitude in zip(cities_df['Neighbourhood'], cities_df['Latitude'], cities_df['Longitude']):
    label = neighbourhood
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [latiude, longitude],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(hyderabad_map) 
hyderabad_map


# In[27]:


LIMIT = 100
radius = 500

CLIENT_ID = 'ZPT23A2KX3YWPVWSILJD5ABOSZIM241NTZU3PWVDWTMRWVFO'
CLIENT_SECRET = 'C25N1NKBSJUWOXIAGSHR1WB4135VYYH4TK3LEOB4S2JAX0OQ'
VERSION = '20180605'


# **Use folium to get venues of each neighbourhoods**

# In[28]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[29]:


hyderabad_venues = getNearbyVenues(names=cities_df['Neighbourhood'],
                                   latitudes=cities_df['Latitude'],
                                   longitudes=cities_df['Longitude'],
                                  )


# In[31]:



print(hyderabad_venues.shape)
hyderabad_venues.head()


# In[32]:



hyderabad_venues.groupby('Neighborhood').count()


# In[33]:



print('There are {} uniques categories.'.format(len(hyderabad_venues['Venue Category'].unique())))


# **One hot enconding for categorial values**

# In[34]:


# one hot encoding
hyderabad_onehot = pd.get_dummies(hyderabad_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
hyderabad_onehot['Neighborhood'] = hyderabad_venues['Neighborhood'] 
hyderabad_onehot.head()


# In[35]:



hyderabad_grouped = hyderabad_onehot.groupby('Neighborhood').mean().reset_index()
print(hyderabad_grouped.shape)
hyderabad_grouped.head()


# In[36]:


num_top_venues = 10

for hood in hyderabad_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = hyderabad_grouped[hyderabad_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[37]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# **Top 10 venues**

# In[38]:


indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{}'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = hyderabad_grouped['Neighborhood']

for ind in np.arange(hyderabad_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(hyderabad_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head(8)


# **Get 5 clusters**

# In[39]:


kclusters = 5

hyderabad_grouped_clustering = hyderabad_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(hyderabad_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[40]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

hyderabad_merged = cities_df

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
hyderabad_merged = hyderabad_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighbourhood')
hyderabad_merged.dropna(subset=['Cluster Labels'], inplace=True)
hyderabad_merged.head(8) # check the last columns!


# In[41]:


hyderabad_merged['Cluster Labels'] = hyderabad_merged['Cluster Labels'].fillna(0.0)


# In[42]:



hyderabad_merged['Cluster Labels'] = hyderabad_merged['Cluster Labels'].map(lambda x: int(x))


# In[43]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(hyderabad_merged['Latitude'], hyderabad_merged['Longitude'], hyderabad_merged['Neighbourhood'], hyderabad_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# **Cluster Analysis**

# **Cluster 1**

# In[44]:


hyderabad_merged.loc[hyderabad_merged['Cluster Labels'] == 0, hyderabad_merged.columns[[0, 1, 2] + list(range(4, hyderabad_merged.shape[1]))]]


# **Cluster 2**

# In[45]:



hyderabad_merged.loc[hyderabad_merged['Cluster Labels'] == 1, hyderabad_merged.columns[[0, 1, 2] + list(range(4, hyderabad_merged.shape[1]))]]


# **Cluster 3**

# In[46]:


hyderabad_merged.loc[hyderabad_merged['Cluster Labels'] == 2, hyderabad_merged.columns[[0, 1, 2] + list(range(4, hyderabad_merged.shape[1]))]]


# **Cluster 4**

# In[47]:



hyderabad_merged.loc[hyderabad_merged['Cluster Labels'] == 3, hyderabad_merged.columns[[0, 1, 2] + list(range(4, hyderabad_merged.shape[1]))]]


# **Cluster 5**

# In[48]:


hyderabad_merged.loc[hyderabad_merged['Cluster Labels'] == 4, hyderabad_merged.columns[[0, 1, 2] + list(range(4, hyderabad_merged.shape[1]))]]


# In[ ]:




