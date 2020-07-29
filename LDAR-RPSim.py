# import basic package
import pandas as pd 
import numpy as np
import geopandas as gpd
import datetime 

# import plot package 
import seaborn as sns 
import matplotlib.pyplot as plt 

# import kmeans 
from sklearn.cluster import KMeans

# import GIS package 
from osgeo import gdal
import osmnx as ox
import networkx as nx
import pycrs 
import geopy 
from shapely.geometry import LineString, Point

#retrive the open street map 
def get_osm (y,x,buffer):
    # y and x are centroid coordinates, buffer is search range (meter)
    G = ox.graph_from_point([y,x],distance = buffer,distance_type='bbox', network_type='drive')
    return G 
	
#convert df to gdf fun 
def df_to_gdf (df,lat,lon):
    # lat and lon are string name f coordinates field in df 
    geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    df = df.drop([lon, lat], axis=1)
    crs = {'init':"epsg:4326"}
    df = gpd.GeoDataFrame(df,crs=crs,geometry = geometry)
    return df

# cluster analysis 
def spatial_clus(gdf,k): 
    # gdf is a geodataframe 
    # k is kmeans number
    X = [list([pt.y,pt.x]) for pt in gdf['geometry']]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    l = kmeans.labels_
    cnt = kmeans.cluster_centers_
    gdf['label'] = l
    
    cnt_la = [] 
    cnt_lo = [] 
    for i in cnt: 
        cnt_la.append(i[0])
        cnt_lo.append(i[1])
        
    c_df= pd.DataFrame()
    c_df['lat'] = cnt_la
    c_df['lon'] = cnt_lo
    c_df['label'] = c_df.index
        
    return gdf,c_df
	
	
# find the closed site 
def findclose(start,center):
    # both 2 inputs are geodataframe 
    dist=[]
    for cnt in center.geometry: 
        d = start.geometry.distance(cnt)
        if isinstance(d, float):
            dist.append(d)
        else:
            dist.append(d[0])
    index = dist.index(min(dist))        
    return index
	
def get_dist_and_time (graph,route):
    # find travel distance and travel time of route 
    # distacne: m  ,,,, time: s 
    fi = 0 
    bi = 1 
    index = 0 
    l = len(route)
    # route total distance
    route_t_dist = []
    # route total travel time  
    route_t_time = []
    while index< l-1:
        # get the edge between two nodes 
        speed_dict = graph.get_edge_data(route[fi],route[bi])[0]
        # Check whether there are maxspeed atrribute
        if 'maxspeed'in speed_dict.keys():
            s = speed_dict['maxspeed']
            if type(s) == list:
                speed = int((s[1]))
            else: 
                speed = int(s) 
        else: 
            highway = speed_dict['highway']
            if highway == 'primary':
                speed = 110
            elif highway == 'secondary':
                speed = 90 
            elif highway == 'unclassified':
                speed = 80
            else: 
                speed = 50

        # get the length between two nodes 
        distance = graph.get_edge_data(route[fi],route[bi])[0]['length']
        
        route_t_dist.append(distance)

        #calculatet the driving time (convert km/h to m/s)
        speed = speed / 3.6
        time = (distance/speed)
        route_t_time.append(time)

        fi += 1 
        bi += 1 
        index += 1 
    
    return sum(route_t_dist), sum(route_t_time)

def find_homebase(site,homebase_df): 
    # both site and homebase_df need to be geodataframe/series  
    
    site_pt = site.geometry 
    l = len(homebase_df)
    i = 0
    dist =[]
    while i<l: 
        hb = homebase_df.iloc[i]
        hb_pt = hb.geometry 
        d = site_pt.distance(hb_pt)
        dist.append(d)
        i += 1 
    
    index = dist.index(min(dist)) 

    final_hb = homebase_df.iloc[index]
    return final_hb
	
# load site list 
site = pd.read_excel('GP/GP_new_500_sites.xlsx')
site_df = df_to_gdf (site,'lat','lon')
# load hombe base data 
home_base_df = pd.read_excel('GP/Ori.xlsx')
home_base_df = df_to_gdf(home_base_df,'lat','lon')
homebase = home_base_df.iloc[1]
# download road network 
G = get_osm(homebase.geometry.y,homebase.geometry.x,110000)

# Get Edges and Nodes
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
# Get origin x and y coordinates in Open Street Map
orig_xy = (homebase.geometry.y,homebase.geometry.x)
home_node = ox.get_nearest_node(G, orig_xy, method='euclidean')

#Route Simulation
# site list = all sites 
site_list = site_df
# survey day 
survey_day = 0
# Daily surveyed sites number 
Daily_survey_site_list = []
# Daily_total_travel_Distacne_between_site_homebase
Daily_distance_between_site_homebase_list= []
# site to site 
Daily_distance_site_to_site_list =[]
# daily working hour
Daily_working_hour_list =[]

survey = True 
while survey: 
    # finish all sites 
    if site_list.empty:
        survey = False
        break
    
    else: 
        l1 = len(site_list)
        # survey clock  
        clock = datetime.datetime(2019,1,1,8,0,0)
        # Cluster analysis 
        if l1 <= 10: 
            cluster_df = site_list 
        else: 
            # kmean cluster number is one of control variable
            site_list, c_df= spatial_clus(site_list,5)
            center = df_to_gdf(c_df,'lat','lon')
            # find the closest cluster 
            clus = findclose(homebase,center)
            # extract sites inside that cluster 
            cluster_df = site_list[site_list.label==clus]
        # find nearest site inside that cluster 
        target_index = findclose(homebase,cluster_df)
        target = cluster_df.iloc[target_index]
        # convert site to geodataframe 
        target_site = gpd.GeoDataFrame([[target.geometry]], geometry='geometry', crs=edges.crs, columns=['geometry']) 
        pt = (target.geometry.y,target.geometry.x)
        # find node of site in graph 
        target_node = ox.get_nearest_node(G, pt, method='euclidean')
        Day = True 
        site_to_site = []
        while Day:
            Daily_distance_between_site_homebase = 0 
            ###### Time Control #####
            if clock.hour == 8: 
                # homebase to site 
                # create route
                orig_node = home_node
                route = nx.dijkstra_path(G=G, source=orig_node, target=target_node, weight='time')
                # Calculate Travel Distance and Travel Time 
                home_to_site_d,base_to_site_t = get_dist_and_time (G,route)
                
                # update time (Travel time and survey time)
                clock = clock + datetime.timedelta(seconds=base_to_site_t) 
                clock = clock + datetime.timedelta(hours = int(target.s_hour)) 
                
                # redefine the departure node  
                orig_node = target_node
                
                # update site list and cluster list 
                cluster_df = cluster_df[cluster_df.facility_ID != target.facility_ID ]
                site_list = site_list[site_list.facility_ID != target.facility_ID ]
                
                print (clock)
                
            elif clock.hour >= 17: 
                # time to go home (site to home) 
                # find the nearest homebase
                today_homebase = find_homebase(target,home_base_df)
                homebase = today_homebase
                today_home_coor = (today_homebase.geometry.y,today_homebase.geometry.x)
                town = today_homebase.town
                home_node = ox.get_nearest_node(G, today_home_coor, method='euclidean')
                # we only have one homebase - RMH
                print (town)
                # route analysis 
                route = nx.dijkstra_path(G=G, source=orig_node, target=home_node, weight='time')
                 # find distance & time  
                site_to_home_d,site_to_home_t = get_dist_and_time (G,route)
                # update only travel time  
                clock = clock + datetime.timedelta(seconds=site_to_home_t) 
                
                # time pool 
                dt = clock - datetime.datetime(2019,1,1,8,0,0)
                Daily_working_hour_list.append(dt.seconds)
                
                Day = False 
                
            else:
            
                ###  survey between sites ### 
                if cluster_df.empty:
                    
                    if site_list.empty:
                        # find the nearest site and keep working 
                        today_homebase = find_homebase(target,home_base_df)
                        homebase = today_homebase
                        today_home_coor = (today_homebase.geometry.y,today_homebase.geometry.x)
                        town = today_homebase.town
                        home_node = ox.get_nearest_node(G, today_home_coor, method='euclidean')
                        print (town)
                        # create route 
                        route = nx.dijkstra_path(G=G, source=orig_node, target=home_node, weight='time')

                        # calculate the travel distance and travel time 
                        site_to_home_d,site_to_home_time = get_dist_and_time (G,route)                            
                        clock = clock + datetime.timedelta(seconds= site_to_home_time) 
                        # time pool 
                        dt = clock - datetime.datetime(2019,1,1,8,0,0)
                        Daily_working_hour_list.append(dt.seconds)
                        Day = False 
                    else:
                        # redo the cluster analysis and update the target site list
                        ll = len(site_list)
                        if ll <= 10: 
                            cluster_df = site_list
                        else:
                            ## redo the cluster analysis and update the target site list 
                            site_list, c_df= spatial_clus(site_list,5)
                            clus = findclose(target_site,center)
                            cluster_df = site_list[site_list.label==clus]
                        
                else:
                    #### Real survey process #### 
                    # find next nereast site 
                    tar_index = findclose(target_site,cluster_df)
                    
                    # find that site from cluster_df and route graph 
                    target = cluster_df.iloc[tar_index]
                    target_site = gpd.GeoDataFrame([[target.geometry]], geometry='geometry', crs=edges.crs, columns=['geometry'])
                    pt = (target.geometry.y,target.geometry.x)
                    target_node = ox.get_nearest_node(G, pt, method='euclidean')
                    
                    print (orig_node,target_node) 
                    
                    if orig_node == target_node: 
                        # two sites have similar nodes in open street map (could be a issue for simulation)
                        # only add the survey time 
                        clock = clock + datetime.timedelta(hours = int(target.s_hour)) 
                        print (clock.hour)

                        #drop the visited site from both cluster site sitelist and overall sitelist 
                        cluster_df = cluster_df[cluster_df.facility_ID != target.facility_ID ]
                        site_list = site_list[site_list.facility_ID != target.facility_ID ]
                        
                        
                    else:    
                        # two sites have different nodes in open street map 
                        route = nx.dijkstra_path(G=G, source=orig_node, target=target_node, weight='time')

                        # calculate the travel distance and travel time 
                        site_to_site_d,site_to_site_t = get_dist_and_time (G,route)
                        site_to_site.append(site_to_site_d)

                        # update the clock 
                        clock = clock + datetime.timedelta(seconds=site_to_site_t) 
                        clock = clock + datetime.timedelta(hours = int(target.s_hour)) 

                        print (clock)

                        #drop the visited site from both cluster site sitelist and overall sitelist 
                        cluster_df = cluster_df[cluster_df.facility_ID != target.facility_ID ]
                        site_list = site_list[site_list.facility_ID != target.facility_ID ]
                    
                    # update departure node                                
                    orig_node = target_node 

        # distacne pool
        Daily_distance_site_to_site_list.append(sum(site_to_site))   
        Daily_distance_between_site_homebase = site_to_home_d + home_to_site_d
        Daily_distance_between_site_homebase_list.append(Daily_distance_between_site_homebase)
        # update survey day        
        survey_day = survey_day + 1 

        # Check how many site we visited today
        l2 = len(site_list)
        site_survey_per_day = l1 - l2
        Daily_survey_site_list.append(site_survey_per_day)
                                                   
        homebase = today_homebase
                                                   
        print (l2)
        print ("Day " + str(survey_day))
        
coop_dict = {'Daily_survey_site_list':Daily_survey_site_list,
       'Daily_distance_between_site_homebase_list (m)':Daily_distance_between_site_homebase_list,
       'Daily_distance_site_to_site_list (m)':Daily_distance_site_to_site_list,
       'Daily_working_hour_list (s)':Daily_working_hour_list}

fdf = pd.DataFrame(coop_dict)

print ("Output Table !!")

fdf.to_excel('Brooks/kmean=5_500_Brooks_2homebase_ColV3.xlsx')