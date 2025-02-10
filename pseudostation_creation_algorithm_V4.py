# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:03:10 2025

@author: gprib
"""

###############################################################################
# pseudostation_creation_algorithm_v4 
###############################################################################

# Algorithm for placing pseudostations in an area of interest
# The algortihm places the stations by combining height zones of a dgm and
# the normalised variance of a .nc meteo file. These two are combined to a score, called combined_score.
# The k-means clustering algorithm clusters the combined_score to the amount of pseudostations you want to place.
# In the clusters, the pixel with the highest value of the combined score is choosen, with the restriction that
# the pseudostations need to have a distance of 1 pixel to existing stations. If the pixel is to close 
# to an existing station, the next pixel will be choosen (iterative).
#####################################################################################################
# IMPORTANT NOTE:
# Make sure all grids and shapefiles are in the same projection. In this example the
# .nc file has a different projection. But due to the fact that .nc files have lon, lat coordinates, this is no
# issue. If you are working with .tif files, make sure everything is in the same projection.
# The output .nc files are not properly georeferenced, keep that in mind!

# Load libraries
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from rasterio.mask import mask
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os


def pseudostations_creation(existing_shapefile:str, dem_path:str, meteo_path:str, ug_shapefile:str, output_shapefile:str, 
                            plot_path:str, number_pseudostations:int, distance:int):
    '''
    

    Parameters
    ----------
    existing_shapefile : str
        Path to shapefile for existing stations.
    dem_path : str
        Path to DGM.
    meteo_path : str
        Path to meteo.nc file.
    ug_shapefile : str
        Path to shapefile for area of interest.
    output_shapefile : str
        Path to shapefile for the pseudostations.
    plot_path : str
        Path to the location for visualisation plot.
    number_pseudostations : int
        Number of clusters for placing the pseudostations. NUmber clusters = Number Pseudostations.
    distance : int
        Distance to existing pseudostations in pixel.

    Returns
    -------
    None.

    '''
    print('###################################')
    print('Running script: pseudostations_creation')
    
    # dynamic variables, adjust to your needs
    # weights for the height zones
    weight_height_zone_1 = 0.2
    weight_height_zone_2 = 0.3
    weight_height_zone_3 = 0.3
    weight_height_zone_4 = 0.2
        
    # weights for the variance and the height zones array
    meteo_variance_weight = 0.6
    height_zones_weight = 0.4
    
    try:
        # Load the shapefile with existing stations
        existing_stations_gdf = gpd.read_file(existing_shapefile)
        print(f"Shapefile loaded: {existing_shapefile}")
        
        # Load DEM
        print(f"Opening DEM file: {dem_path}")
        with rasterio.open(dem_path) as dem:
            dem_data = dem.read(1)  # Read the first band (elevation data)
            dem_transform = dem.transform
            dem_crs = dem.crs
            print(f"DEM loaded successfully with CRS: {dem_crs}")

        # Load the netCDF file with the variance
        print(f"Opening netCDF file: {meteo_path}")
        meteo_data = nc.Dataset(meteo_path)
        combined_variance_normalized = meteo_data["combined_variance_normalized"][:]

        # Load UG shapefile
        print(f"Loading UG shapefile: {ug_shapefile}")
        ug_gdf = gpd.read_file(ug_shapefile)
        
        # Create a mask from the UG shapefile without cropping the DEM
        print("Creating a mask from the UG shapefile without cropping the DEM.")
        with rasterio.open(dem_path) as src:
            # Generate a mask that retains the DEM's original dimensions
            out_image, _ = mask(src, ug_gdf.geometry, crop=False, invert=False)
            # Convert the masked areas (outside UG) to NaN, retaining the DEM dimensions
            masked_dem = np.where(out_image[0] == src.nodata, np.nan, out_image[0])
            print("Mask created successfully, retaining DEM dimensions.")

        # Apply mask to the reprojected combined variance
        combined_variance = np.where(np.isnan(masked_dem), np.nan, combined_variance_normalized)

        # Create height zones from the DEM
        print("Creating height zones from the DEM.")
        bins = [1000, 2000, 3000, 4000]
        digitized = np.digitize(np.nan_to_num(masked_dem, nan=-1), bins=bins)
        height_zones = digitized.astype(float)
        height_zones[np.isnan(masked_dem)] = np.nan
        
        # Create height weights array and assign the weighted zones
        height_weights = np.zeros_like(height_zones, dtype=np.float32)
        height_weights[height_zones == 1] = weight_height_zone_1  # 0–999m
        height_weights[height_zones == 2] = weight_height_zone_2  # 1000–1999m
        height_weights[height_zones == 3] = weight_height_zone_3  # 2000–2999m
        height_weights[height_zones == 4] = weight_height_zone_4  # 3000m and above

        # Combine factors into a combined score
        print("Combining factors to calculate combined score.")
        combined_score = np.zeros_like(combined_variance)
        valid_mask = ~np.isnan(combined_variance) & ~np.isnan(height_weights)
        combined_score[valid_mask] = (
            meteo_variance_weight * combined_variance[valid_mask] +
            height_zones_weight * height_weights[valid_mask]
        )
        combined_score[~valid_mask] = np.nan

        # Apply clustering
        print("Applying clustering to identify representative areas.")
        
        # Flatten the combined variance and height weights into a 2D array for clustering
        cluster_data = np.column_stack((combined_variance.ravel(), height_weights.ravel()))
        
        # Identify valid data points (non-NaN values) for clustering
        valid_indices = np.all(~np.isnan(cluster_data), axis=1)
        cluster_data = cluster_data[valid_indices]
        
        # Standardize the data to ensure equal weighting in clustering
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Define the number of clusters as the desired number of pseudostations
        n_clusters = number_pseudostations  # Set the number of clusters
        
        # Apply KMeans clustering algorithm to categorize the data into groups
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(cluster_data_scaled)
        
        # Initialize a cluster map with NaN values, matching the shape of the original data
        cluster_map = np.full_like(combined_variance, fill_value=np.nan)
        
        # Get the valid positions from the original mask (where data is available)
        valid_positions = np.argwhere(valid_mask)
        
        # Assign the cluster labels to the corresponding valid positions in the cluster map
        for i, pos in enumerate(valid_positions):
            cluster_map[pos[0], pos[1]] = clusters[i]

        # Mask the cluster map as well
        cluster_map = np.where(np.isnan(masked_dem), np.nan, cluster_map)

        # Select pseudostations
        print("Selecting pseudostations.")
        pseudostations = select_pseudostations_with_clustering(
            combined_score, cluster_map, n_clusters, min_distance=distance, max_stations = number_pseudostations, 
        existing_stations_gdf = existing_stations_gdf)
        print(f"Selected {len(pseudostations)} pseudostations.")

        # Write the pseudostations to a shapefile
        pseudostations_latlon = []
        heights = []
        longitudes = []
        latitudes = []
        combined_score_out = []
        cluster_score = []

        for (y, x) in pseudostations:
            lon, lat = rasterio.transform.xy(dem_transform, y, x)
            pseudostations_latlon.append((lat, lon))
            heights.append(masked_dem[y, x])
            longitudes.append(lon)
            latitudes.append(lat)
            combined_score_out.append(combined_score[y, x])
            cluster_score.append(cluster_map[y, x])

        geometry = [Point(lon, lat) for lat, lon in pseudostations_latlon]
        gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
        gdf['height'] = heights
        gdf['longitude'] = longitudes
        gdf['latitude'] = latitudes
        gdf['comb_score'] = combined_score_out  # Shortened name
        gdf['clus_score'] = cluster_score       # Shortened name
        
        # Save the GeoDataFrame to a shapefile
        gdf.to_file(output_shapefile)
        print(f"Shapefile saved: {output_shapefile}")
        
        # Visualization
        print("Creating visualization plot.")
        fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

        # Define a gray color for NaN values
        cmap_variance = plt.cm.viridis.copy()
        cmap_variance.set_bad(color='gray')
        cmap_height_zones = plt.cm.terrain.copy()
        cmap_height_zones.set_bad(color='gray')
        cmap_clusters = plt.cm.tab20b.copy()
        cmap_clusters.set_bad(color='gray')
        cmap_combined_score = plt.cm.inferno.copy()
        cmap_combined_score.set_bad(color='gray')

        # Define a light gray background for the axes
        background_color = 'lightgrey'

        # Combined Variance (Normalized)
        axs[0, 0].set_facecolor(background_color)  # Set background to light gray
        im1 = axs[0, 0].imshow(combined_variance_normalized, cmap=cmap_variance)
        axs[0, 0].scatter([s[1] for s in pseudostations], [s[0] for s in pseudostations], color='red', label='Pseudostations')
        axs[0, 0].set_title("Combined Variance (Normalized)")
        fig.colorbar(im1, ax=axs[0, 0], label='Variance')

        # Height Zones
        axs[0, 1].set_facecolor(background_color)  # Set background to light gray
        im2 = axs[0, 1].imshow(height_zones, cmap=cmap_height_zones)
        axs[0, 1].scatter([s[1] for s in pseudostations], [s[0] for s in pseudostations], color='red')
        axs[0, 1].set_title("Height Zones")
        fig.colorbar(im2, ax=axs[0, 1], label='Altitude Zone')

        # Clusters
        axs[1, 0].set_facecolor(background_color)  # Set background to light gray
        im3 = axs[1, 0].imshow(cluster_map, cmap=cmap_clusters)
        axs[1, 0].scatter([s[1] for s in pseudostations], [s[0] for s in pseudostations], color='red')
        axs[1, 0].set_title("Clusters")
        fig.colorbar(im3, ax=axs[1, 0], label='Cluster ID')

        # Combined Score
        axs[1, 1].set_facecolor(background_color)  # Set background to light gray
        im4 = axs[1, 1].imshow(combined_score, cmap=cmap_combined_score)
        axs[1, 1].scatter([s[1] for s in pseudostations], [s[0] for s in pseudostations], color='red')
        axs[1, 1].set_title("Combined Score")
        fig.colorbar(im4, ax=axs[1, 1], label='Score')

        # plt.tight_layout()
        plt.savefig(f"{plot_path}/pseudostations_{number_pseudostations}.png")
        plt.close()

    except Exception as e:
        print(f"Error during pseudostations creation: {e}")
    
    print("###################################")




def select_pseudostations_with_clustering(combined_score: np.ndarray, cluster_map: np.ndarray, n_clusters: int,
                                          min_distance: int, max_stations: int, existing_stations_gdf: gpd.GeoDataFrame):
    '''
    Selects pseudostations distributed evenly across clusters while ensuring a minimum distance.

    Parameters
    ----------
    combined_score : np.ndarray
        A 2D array representing the score for station selection.
    cluster_map : np.ndarray
        A 2D array representing cluster assignments for grid cells.
    n_clusters : int
        Number of clusters to distribute stations.
    min_distance : int
        Minimum allowed distance between selected pseudostations.
    max_stations : int
        Maximum number of pseudostations to be selected.
    existing_stations_gdf : gpd.GeoDataFrame
        GeoDataFrame containing existing station locations.

    Returns
    -------
    list
        A list of selected pseudostation coordinates (row, col).
    '''
    # Load existing stations from shapefile and extract coordinates
    existing_stations_coords = np.array([(geom.y, geom.x) for geom in existing_stations_gdf.geometry])
    
    # Initialise a empty list for the pseudostations
    pseudostations = []
    
    # create an occupation mask for distance checks
    occupied_mask = np.zeros_like(combined_score, dtype=bool)
    
    # Convert existing station locations to the occupied mask
    for station in existing_stations_coords:
        row, col = np.round(station).astype(int)
        if 0 <= row < combined_score.shape[0] and 0 <= col < combined_score.shape[1]:
            occupied_mask[row, col] = True
    
    # Select the best station per cluster
    for cluster_id in range(n_clusters):
        cluster_indices = np.argwhere(cluster_map == cluster_id)
        cluster_indices = sorted(
            cluster_indices, key=lambda idx: combined_score[idx[0], idx[1]], reverse=True
        )
        
        for idx in cluster_indices:
            row, col = idx
            new_station = (row, col)
    
            if not np.isnan(combined_score[row, col]) and check_minimum_distance_and_occupation(
                    new_station, pseudostations, occupied_mask, min_distance):
                pseudostations.append(new_station)
                occupied_mask[row, col] = True
                break
    
    # Sort remaining locations by score and fill additional stations
    sorted_indices = np.dstack(np.unravel_index(np.argsort(-combined_score.ravel()), combined_score.shape))[0]
    
    for idx in sorted_indices:
        if len(pseudostations) >= max_stations:
            break
        row, col = idx
        new_station = (row, col)
    
        if not np.isnan(combined_score[row, col]) and check_minimum_distance_and_occupation(
                new_station, pseudostations, occupied_mask, min_distance):
            pseudostations.append(new_station)
            occupied_mask[row, col] = True
    
    return pseudostations



def check_minimum_distance_and_occupation(new_station: tuple, existing_stations: list, occupied_mask: np.ndarray, min_distance: float) -> bool:
    '''
    Checks if a new station is at least `min_distance` away from existing stations and not in an occupied area.

    Parameters
    ----------
    new_station : tuple
        The (row, col) coordinates of the new station.
    existing_stations : list
        List of existing station coordinates.
    occupied_mask : np.ndarray
        A 2D boolean mask indicating occupied positions.
    min_distance : float
        Minimum required distance between stations.

    Returns
    -------
    bool
        True if the new station meets the distance and occupation criteria, False otherwise.
    '''
    # If no existing stations, return whether the location is unoccupied
    if len(existing_stations) == 0:
        return not occupied_mask[new_station[0], new_station[1]]
    
    # Use a KDTree to efficiently find the nearest existing station
    tree = KDTree(existing_stations)
    dist, _ = tree.query([new_station], k=1)
    
    # Return True if the distance is sufficient and the location is not occupied
    return dist[0] >= min_distance and not occupied_mask[new_station[0], new_station[1]]



# Example usage
path = os.getcwd()
if __name__ == "__main__":
    pseudostations_creation(
        existing_shapefile = fr"{path}/data/shapefiles/stations_LWD_HD.shp",
        dem_path = fr"{path}/data/dem/dem.tif",
        meteo_path = fr"{path}/data/meteo/output_variance.nc",
        ug_shapefile = fr"{path}/data/shapefiles/UG_EPSG_4326.shp",
        output_shapefile = fr"{path}/data/shapefiles/output_points.shp",
        plot_path = fr"{path}/plots/",
        number_pseudostations = 5,
        distance = 1  
    )
