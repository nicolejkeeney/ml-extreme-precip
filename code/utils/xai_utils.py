"""Functions for explainable AI methods

Author: 
    Nicole Keeney 

Version: 
    12-07-2023

"""

from sklearn.cluster import KMeans
import xarray as xr
import numpy as np 

def kmeans_clusters_xr(da, num_clusters=2, n_init=10, max_iter=1000, random_state=42):
    """
    Perform kmeans algorithm on xarray object to get cluster centers 
    Returns an xarray object such that the function can be applied using xr.apply or xr.map  

    Parameters 
    ----------
    da: xr.DataArray 
        DataArray with the dimensions "time","lat","lon"
    num_clusters: int, optional 
        Number of clusters (k)
    n_init: int, optional 
    max_iter: int, optional 
    random_state: int, optional 

    Returns 
    -------
    clusters_da: xr.DataArray 
        Cluster centers. Dimensions will be "cluster", "lat", "lon"

    References
    ---------- 
    - https://realpython.com/k-means-clustering-python/
    - https://github.com/wy2136/xlearn/blob/master/xlearn/cluster.py
    """
    # Perform n_init runs of the k-means algorithm on data with a maximum of max_iter iterations per run
    kmeans_fitted, da_stacked = kmeans_xr(
        da, 
        num_clusters=num_clusters, 
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
        )

    # Reformat data. Needs to go back into an xarray object
    clusters_da = xr.DataArray(
        kmeans_fitted.cluster_centers_, 
        coords={
            "cluster":np.arange(0,num_clusters),
            "xy":da_stacked.xy
            }
    )
    clusters_da = clusters_da.unstack("xy") # Unstack data to go back to lat,lon

    return clusters_da 

def kmeans_xr(da, num_clusters=2, n_init=10, max_iter=1000, random_state=42):
    """
    Perform kmeans algorithm on xarray object 

    Parameters 
    ----------
    da: xr.DataArray 
        DataArray with the dimensions "time","lat","lon"
    num_clusters: int, optional 
        Number of clusters (k)
    n_init: int, optional 
    max_iter: int, optional 
    random_state: int, optional 

    Returns 
    -------
    kmeans_fitted: sklearn.cluster._kmeans.KMeans
    da_stacked: xr.DataArray 
        Stacked input da with dimensions "time", "xy"

    References
    ---------- 
    - https://realpython.com/k-means-clustering-python/
    - https://github.com/wy2136/xlearn/blob/master/xlearn/cluster.py
    """
    # Instantiate sklearn kmeans class 
    kmeans = KMeans(
        init="k-means++",
        n_clusters=num_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    
    # Stack data across x & y dimension (i.e. flatten array)
    # kmeans can't take more than two dimensions 
    # We just want to give it time and the flattened xy dim 
    da_stacked = da.stack(dimensions={"xy":["lat","lon"]})

    # ORDER OF DIMENSIONS MATTERS! 
    # Time needs to go first 
    da_stacked = da_stacked.transpose("time","xy")

    # Perform n_init runs of the k-means algorithm on data with a maximum of max_iter iterations per run
    kmeans_fitted = kmeans.fit(da_stacked.values)

    return kmeans_fitted, da_stacked