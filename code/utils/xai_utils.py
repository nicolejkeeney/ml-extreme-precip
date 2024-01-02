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
    # Instantiate sklearn kmeans class 
    kmeans = KMeans(
            init="k-means++",
            n_clusters=num_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )

    # kmeans can't take more than two dimensions
    # Stack data across x & y dimension (i.e. flatten array)
    # Also stack the two variables on top of each other
    # This results in a repeating time index (i.e. [1981,1982,1983,1981,1982,1983])
    da_stacked = da.stack(
        {
            "var_and_time":["variable","time"], 
            "xy":["lat","lon"]
        }
        )

    # Perform n_init runs of the k-means algorithm on data with a maximum of max_iter iterations per run
    kmeans_fitted = kmeans.fit(da_stacked.values)

    # Get the cluster centers 
    clusters_np = kmeans_fitted.cluster_centers_

    # Reformat data. Needs to go back into an xarray object
    clusters_da = xr.DataArray(
            clusters_np, 
            coords={
                "cluster":np.arange(0,num_clusters),
                "xy":da_stacked.xy
                }
        )

    # Unstack spatial dimension
    clusters_da = clusters_da.unstack("xy")

    return clusters_da 