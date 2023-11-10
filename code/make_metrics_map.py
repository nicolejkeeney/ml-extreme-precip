""" Module for making figure of metrics map  

Author:
    Nicole Keeney 
    
Version: 
    11-09-2023

"""

import os 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt 
from utils.misc_utils import (
    get_model_settings, 
    check_and_create_dir
)  

# ------------ SET GLOBAL VARIABLES ------------

GEOMS = ["Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

# Pattern for finding the data by model ID 
# Model ID dir should exist as MODEL_ID_PREFIX + GEOM + MODEL_ID_SUFFIX 
# i.e. If MODEL_ID_PREFIX = "nicole_", GEOM = "California", and MODEL_ID_SUFFIX = "_run2", the model_id is "nicole_California_run2"
MODEL_ID_PREFIX = "frances_" 
MODEL_ID_SUFFIX = ""
MODEL_IDS = [MODEL_ID_PREFIX+geom.replace(" ","_")+MODEL_ID_SUFFIX for geom in GEOMS]

# Directories. Needs to have a slash (/) after (i.e "dir/"")
DATA_DIR = "../data/input_data_preprocessed/" 
MODEL_OUTPUT_DIR = "../model_output/"
FIGURES_DIR = "../figures/metrics_maps/"

# Which dataset to make map for? Must be one of "training","validation","testing"
DATASET = "testing"


def make_metric_map_us(model_ids=MODEL_IDS, model_output_dir=MODEL_OUTPUT_DIR, figures_dir=FIGURES_DIR, savefig=True, dataset=DATASET): 
    """Plot precision and recall values for each state on a map of the US 
    
    Parameters
    ----------
    model_ids: list of str
        List of model ids to generate plots for 
        i.e. ["run2_colorado","run5_arizona","run2_virginia"]
    model_output_dir: str, optional
        GENERALIZED model output directory (should not include model_id).
        i.e. "data/model_output/", NOT "data/model_output/some_model_id_here/"
    figures_dir: str, optional
        GENERALIZED figures directory (should not include model_id).
        i.e. "data/model_output/", NOT "data/model_output/some_model_id_here/"
    savefig: boolean, optional 
        Save figure to directory figs_dir? Default to True
    dataset: str, optional
        Which dataset metrics to plot? Must be one of "training","validation","testing" 
        Default to "testing"

    Returns 
    ------- 
    matplotlib figure object 

    """

    check_and_create_dir(figures_dir)

    # Get metrics dataframe 
    def _retrieve_and_format_metrics_df(model_id, model_output_dir=MODEL_OUTPUT_DIR): 
        filepath = model_output_dir+"{0}/{0}_model_metrics.csv".format(model_id)
        metrics_df = pd.read_csv(filepath)
        metrics_df["GEOM"] = get_model_settings(model_id, print_to_console=False)["labels_geom"]
        return metrics_df 
    metrics_by_geom = pd.concat([_retrieve_and_format_metrics_df(model) for model in MODEL_IDS])
    metrics_by_geom = metrics_by_geom[metrics_by_geom["dataset"] == dataset]

    # Get US states geometry. Remove non-CONUS geometries. 
    shp_path = "../data/cb_2018_us_state_5m/"
    us_states = gpd.read_file(shp_path).to_crs("4326")[["NAME","geometry"]]
    not_CONUS = ["Alaska","Hawaii","Commonwealth of the Northern Mariana Islands", "Guam", "American Samoa", "Puerto Rico","United States Virgin Islands"]
    us_states = us_states[~us_states["NAME"].isin(not_CONUS)]
    us_states = us_states.rename(columns={"NAME":"GEOM"})
    us_states["GEOM"] = us_states["GEOM"].apply(lambda x: x.replace(" ","_")) # Replace spaces with hyphen

    # Merge to form one dataframe 
    geom_df = us_states.merge(metrics_by_geom, on="GEOM")

    plots = []
    for metric, cmap in zip(["Accuracy","Recall","Precision"],["YlOrRd","PuBuGn","YlGnBu"]):
        basemap = us_states.plot(color="white", edgecolor="black", linewidth=0.3)
        pl = _metrics_map(
            geom_df, 
            ax=basemap,
            metric=metric, 
            cmap=cmap
            )
        if savefig: # Save figure 
            plt.savefig("{0}{1}_map.png".format(figures_dir,metric.lower()), dpi=300, bbox_inches='tight')

    return plots

def _metrics_map(geom_df, ax=None, metric="Recall", cmap="Reds"): 
    """Plot map of a certain metric

    Parameters
    ----------
    geom_df: geopandas.geodataframe.GeoDataFrame
        Table with columns for each metrics, along with a geometry column 
    ax: matplotlib axis, optional 
        Basemap to use. Default to None 
    metric: str, optional 
        Metric to plot. Default to "Recall"ArithmeticError
    cmap: str, optional 
        Matplotlib colormap to use
    
    Returns 
    -------
    """
    vals = geom_df[metric].values 
    pl = geom_df.plot(
        ax=ax,
        column=metric, 
        legend=True, 
        cmap=cmap, 
        legend_kwds={"label":metric, "shrink":0.7},
        )
    pl.set_title("{0} by state".format(metric))
    return pl

if __name__ == "__main__":
    make_metric_map_us()