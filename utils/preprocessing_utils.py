# Helper functions for performing data preprocessing 

import xarray as xr 
import geopandas as gpd 
import rioxarray as rio

def calc_anomalies(data, var): 
    """
    Calculate daily standardized anomalies 

    Parameters 
    ----------
    data: xr.Dataset 
        Input data. Must contain variable var and time dimension 
    var: str 
        Name of variable to calculate anomalies for 

    Returns
    --------
    xr.Dataset

    Reference 
    ---------
    https://github.com/fdavenport/GRL2021/blob/main/notebooks/0a_read_reanalysis.ipynb
    
    """

    data['mean'] = data[var].groupby('time.dayofyear').mean(dim = 'time')
    data['sd'] = data[var].groupby('time.dayofyear').std(dim = 'time')
    data[var+'_anom'] = (data[var].groupby('time.dayofyear') - data['mean']).groupby('time.dayofyear')/data['sd']
    return data 

def get_state_geom(state="Colorado", shp_path="../data/cb_2018_us_state_5m/", crs="4326"): 
    """
    Get geometry for a specific state 
    Requires shapefile of US state boundaries 

    Parameters
    ----------
    state: str, optional
        Name of state. Default to "Colorado"
        Must be formatted to match state names in shapefile 
    shp_path: str, optional
        Path to shapefile. Default to developer's local path.
    crs: str or None, optional
        EPSG coordinate system. Default to "4326" 
        Set to None to keep shapefile's native CRS (EPSG: 4269)

    Returns 
    -------
    geopandas.geoseries.GeoSeries
        Geometry for desired state in desired projection
    
    """
    us_states = gpd.read_file(shp_path)
    geom = us_states[us_states["NAME"]==state].geometry
    if crs is not None: # Convert CRS
        geom = geom.to_crs(crs)
    return geom

def convert_lon_360_to_180(data): 
    """
    Convert longitude range from 0:360 to -180:180 

    Parameters
    ----------
    data: xr.DataArray or xr.Dataset
        Data with lon range from 0:360 
        Must contain coordinate "lon" 

    Returns
    -------
    xr.DataArray or xr.Dataset 
        Same datatype as input, with shifted longitude 

    References 
    ----------
    https://stackoverflow.com/questions/53345442/about-changing-longitude-array-from-0-360-to-180-to-180-with-python-xarray
    
    """
    lon_attrs = data.lon.attrs # Preserve attributes for longitude 
    data.coords["lon"] = (data.coords['lon'] + 180) % 360 - 180
    data.coords["lon"].attrs = lon_attrs # Reassign attributes to lon coord 
    data = data.sortby(data.lon)
    return data 

def clip_to_geom(data, geom, keep_spatial_ref=False): 
    """
    Clip data to input geometry

    Parameters 
    ----------
    data: xr.DataArray or xr.Dataset
        Data to clip 
    geom: geopandas.geoseries.GeoSeries
        Geometry to clip data to. 
        Must have CRS: EPSG 4326 
    keep_spatial_ref: boolean, optional 
        Keep spatial reference information as a coordinate in output data? 
        Default to False. 

    Returns 
    -------
    xr.DataArray or xr.Dataset
    
    """
    data = data.rio.write_crs("4326") # Assign CRS 
    rio_rename = {"lon":"x","lat":"y"} # Rioxarray requires spatial coordinates be named x,y
    data = data.rename(rio_rename)
    data_clipped = data.rio.clip(geom) # Clip to geometry 
    data_clipped = data_clipped.rename({v: k for k, v in rio_rename.items()}) # Go back to lat/lon coords 
    if keep_spatial_ref == False: 
        data_clipped = data_clipped.drop("spatial_ref")
    return data_clipped