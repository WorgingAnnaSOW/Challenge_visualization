#########################################################################################
##----------------CAN'T CONTINUE THIS VISUALIZATION CUZ OSMX -----------------------
###---------------PACKAGE DOESN'T WORK----------------------------------------------
####------EVEN AFTER USING THE LINKS IN THE 'TEMPS_MEMOIRE' TP-----------------
#########################################################################################



# %%

import folium



# %%

map_osm.add_child(folium.RegularPolygonMarker(location=[43.610769, 3.876716],
                  fill_color='#132b5e', radius=5))
map_osm

# %%
import pandas as pd
# %%
df_compteur=pd.read_csv('compteurs.csv',encoding= 'unicode_escape')
df_compteur
# %%
#Tanneurs=[43.616209,3.874408]
#Berracasa=[43.609699,3.896940]
#Celleneuve=[43.614650,3.833600]
#Laverune=[43.590700,3.813240]
#Vieille_poste=[	43.615742,3.909632]
#Delmas1_2=[43.626698,3.895629]
#Gerhardt=[43.613884,3.868467]
#Lattes2=[43.579260,3.933270]
#Lattes1=[43.578830,3.933240]

# %%
map_osm1 = folium.Map(location=[43.616209,3.874408])
map_osm2 = folium.Map(location=[43.609699,3.896940])
map_osm3 = folium.Map(location=[43.614650,3.833600])
map_osm4 = folium.Map(location=[43.590700,3.813240])
map_osm5 = folium.Map(location=[43.615742,3.909632])
map_osm6 = folium.Map(location=[43.626698,3.895629])
map_osm7 = folium.Map(location=[43.613884,3.868467])
map_osm8 = folium.Map(location=[43.579260,3.933270])
map_osm9 = folium.Map(location=[43.578830,3.933240])


# %%
map_osm1.add_child(folium.RegularPolygonMarker(location=[43.610769, 3.876716],
                  fill_color='#132b5e', radius=5))
map_osm1

# %%
map_osm.add_child(folium.RegularPolygonMarker(location=[43.616209,3.874408],
                  fill_color='#132b5e', radius=5))
map_osm





###############################################################
##----------------CAN'T CONTINUE THE VISUALIZATION CUZ OSMX -----------------------
###---------------PACKAGE DOESN'T WORK------------------------------------
####------EVEN AFTER USING THE LINKS IN THE 'TEMPS_MEMOIRE' TP-----------------
##########################################################################
# %%

import osmnx as ox
ox.utils.config(use_cache=True)  # caching large download
ox.__version__

# %%

G = ox.graph_from_place('Montpellier, France', network_type='bike')

# %%
print(f"nb edges: {G.number_of_edges()}")
print(f"nb nodes: {G.number_of_nodes()}")


# %%

ox.plot_graph(G)






# # Visualize shorthest path between two points.

# %%

origin = ox.geocoder.geocode('******************, Montpellier, France')
destination = ox.geocoder.geocode('****************, Montpellier, France')

origin_node = ox.get_nearest_node(G, origin)
destination_node = ox.get_nearest_node(G, destination)

print(origin)
print(destination)
route = nx.shortest_path(G, origin_node, destination_node)


# %%

ox.plot_graph_route(G, route)


# %%

ox.plot_route_folium(G, route, route_width=2, route_color='#AA1111')

