import pandas as pd
import folium

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans #research KMeans

import matplotlib.pyplot as plt

#Geospatial masking - altering coordinates of park data

parks_data = pd.read_csv('parks.csv')
species_data = pd.read_csv('species.csv')

sensitive_columns = ['Latitude', 'Longitude']

anonymized_parks_data = parks_data.copy()
anonymized_parks_data[sensitive_columns] = anonymized_parks_data[sensitive_columns].apply(lambda x: x + np.random.uniform(-5, 5))

#Normalization for AI assisted anonymization

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(anonymized_parks_data[sensitive_columns])

kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(normalized_data)

anonymized_parks_data['Cluster'] = clusters

#Add the 'Clusters_Labels' column to the dataset
parks_data['Cluster'] = anonymized_parks_data['Cluster']

cluster_to_region = {
    0 : 'Northern USA',
    1 : 'Southwest USA',
    2 : 'Eastern USA',
    3 : 'Alaska',
    4 : 'Hawaii'
}

parks_data['Region'] = parks_data['Cluster'].map(cluster_to_region)
parks_data.head()

plt.scatter(anonymized_parks_data['Longitude'], anonymized_parks_data['Latitude'], c=anonymized_parks_data['Cluster'], cmap='viridis')
plt.title('Parks Clusters Based on Anonymized Location')
plt.xlabel('Anonymized Longitude')
plt.ylabel('Anonymized Latitude')
plt.show()

m_true = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
m_anonymized = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

#Plotting true park coordinates in red
for index, row in parks_data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius = 5,
        color='red',
        fill = True,
        fill_color = 'red',
        fill_opacity = 0.6,
    ).add_to(m_true)

#Ploting anonymized park coordinates in red
for index, row in anonymized_parks_data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius = 5,
        color='red',
        fill = True,
        fill_color = 'red',
        fill_opacity = 0.6,
    ).add_to(m_anonymized)

#Saving maps
m_true.save('true_parks_map.html')
m_anonymized.save('anonymized_parks_map.html')

#Define cluster colors
cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']

# Loop through the data nad plot marks with cluster-specifc colors

for index, row in anonymized_parks_data.iterrows():
    cluster_color = cluster_colors[row['Cluster']]
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius = 5,
        color=cluster_color,
        fill=True,
        fill_color=cluster_color,
        fill_opacity=0.6,
    ).add_to(m)

#Display the map
m.save('clustered_parks_map.html')


                                      



