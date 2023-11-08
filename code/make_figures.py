""" Module for making figures 

Author:
    Nicole Keeney 
    
Version: 
    11-03-2023

"""

import os 
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils.misc_utils import (
    check_and_create_dir, 
    get_model_settings, 
    get_season_str
)
from utils.read_data_utils import get_input_data 


# ------------ SET GLOBAL VARIABLES ------------
 
# Model ID needs to match string key in model_settings.json
MODEL_ID = "frances_california"

# Directories. Needs to have a slash (/) after (i.e "dir/"")
DATA_DIR = "../data/input_data_preprocessed/" 
MODEL_OUTPUT_DIR = "../model_output/"+MODEL_ID+"/"  # Saved predictions should be here 
FIGURES_DIR = "../figures/"+MODEL_ID+"/" # Where to save figures to 

# Projection to use for basemap for map figures 
MAP_CRS = ccrs.AlbersEqualArea(central_longitude=-100, central_latitude=35, standard_parallels=(30, 60))


def main(model_id=MODEL_ID, data_dir=DATA_DIR, model_output_dir=MODEL_OUTPUT_DIR, figures_dir=FIGURES_DIR, map_crs=MAP_CRS):
    """Run all figure-making functions
    
    Parameters 
    ----------
    model_id: str, optional 
        String identifier for model
    data_dir: str, optional 
        Directory for preprocessed input data 
    model_output_dir: str, optional 
        Directory where model output (predictions) has been saved
    figures_dir: str, optional 
        Directory for saving figures
    map_crs: cartopy.crs, optional
        Caropy projection to use for basemap

    Returns 
    -------
    None 

    """

    ## -------- SETUP --------

    print("USER INPUTS\n------------")
    print("Model ID: {0}".format(model_id))
    print("Data directory: {0}".format(data_dir))
    print("Figure directory: {0}".format(figures_dir))
    print("Model output directory: {0}".format(model_output_dir))
    print("Map projection {0}".format(map_crs))

    # Get model settings 
    settings = get_model_settings(model_id, print_to_console=False)

    # Create figures dir if it does't exist 
    check_and_create_dir(figures_dir)

    # Confirm that predictions and training history data exists 
    predictions_filepath = model_output_dir+model_id+"_predictions.csv"
    history_filepath = model_output_dir+model_id+"_history.csv"
    for pathname, path in zip(["predictions","training history"],[predictions_filepath, history_filepath]):
        if not os.path.isfile(path): 
            raise ValueError("No file found for {0} data at path {1}. Figures cannot be generated.".format(pathname, path))

    ## -------- READ DATA  --------
    
    # Saved predictions data 
    predictions = pd.read_csv(predictions_filepath, index_col=False)

    # Preprocessed model input data 
    x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df = get_input_data(data_dir, settings)

    # Training history by epoch 
    training_history = pd.read_csv(model_output_dir + model_id + "_history.csv", index_col=False)
    
    ## -------- MAKE PLOTS --------
    
    ## (1) Mean precip (x) vs. predicted probability of EPCP (y)  
    make_precip_vs_prob_plot(
        predictions["precip_mean"], 
        predictions["prob_1"], 
        savefig=True, 
        figures_dir=figures_dir, 
        figname=model_id+"_mean_precip_vs_pred_epcp"
    )
    ## (2) BY SEASON: mean precip (x) vs. predicted probability of EPCP (y)  
    make_precip_vs_prob_plot_by_season(
        predictions, 
        savefig=True, 
        figures_dir=figures_dir, 
        figname=model_id+"_mean_precip_vs_pred_epcp_by_season"
        )

    ## (3) EPCP composite anomaly patterns 
    dates_epcp = predictions[predictions["predicted_class"]==1]["time"].values # Days when EPCP is predicted to occur
    x_all_ds = xr.concat([x_train_ds, x_val_ds, x_test_ds], dim="time") # Combine all the datasets into one xr.Dataset 
    x_all_epcp = x_all_ds.sel(time=dates_epcp) # Get just EPCP days
    x_epcp_mean = x_all_epcp.mean(dim="time").to_array() # Compute mean. Convert to xr.DataArray

    make_epcp_composite_plot(
        x_epcp_mean, 
        map_crs=map_crs, 
        savefig=True,
        figname=model_id+"_epcp_composite", 
        figures_dir=figures_dir
    )
 
    ## (4) Metrics by epoch from model training history 
    plot_metrics_by_epoch(
        training_history, 
        savefig=True,
        figname=model_id+"_training_metrics_by_epoch", 
        figures_dir=figures_dir) 

    return None 

def make_precip_vs_prob_plot(precip_mean, prob_epcp, savefig=True, figures_dir="", dpi=300, fontsize=10, figsize=(6,5), scattercolor="midnightblue", cmap="RdBu_r", in_plot_text=True, ax=None, colorbar=True, bins=25, figname="mean_precip_vs_pred_epcp", title_y=1.03, title="Monthly mean precipitation vs. predicted \nprobability of ECPC for each timestep"): 
    """Generate figure: Monthly mean precipitation vs. predicted probability of ECPC for each timestep
    
    Parameters 
    ----------
    precip_mean: np.array 
        Monthly mean precipitation timeseries 
    prop_epcp: np.array 
        Probability of EPCP timeseries 
    savefig: boolean, optional 
        Save figure as png? Default to True
    figures_dir: str, optional 
        Directory to save figure to. Default to current directory 
    figsize: tuple, optional 
        Size of figure 
    cmap: matplotlib.colormap, optional
        Colormap to use for 2D histogram
    scattercolor: str, optional 
        Color to use for scatters 
    dpi: int, optional 
        Figure DPI. Default to 300
    fontsize: float, optional 
        Size of font on plot. Default to 10 
    in_plot_text: str, optional 
        Add text inside plot showing direction of EPCP? Default to True 
    ax: matplotlib.axes._axes.Axes, optional 
        Plot figure on input axis? Default to False (axis will be created by function instead) 
        Useful for adding multiple of this plot to one figure (subplots)
    colorbar: boolean, optional 
        Add a colorbar? Default to True
    figname: str, optional 
        Name to give saved figure. Do not include extension. 
        Default to "mean_precip_vs_pred_epcp"
    title: str, optional 
        Title to give plot
    title_y: float, optional
        Title y heights  

    Returns
    --------
    ax: matplotlib.axes._axes.Axes
    
    """

    # Generate figure: Monthly mean precipitation vs. predicted probability of ECPC for each timestep
    if ax is None: 
        fig, ax = plt.subplots(figsize=figsize)
    x = precip_mean
    y = prob_epcp
    linecolor = "grey"
    pr_thr = precip_mean.quantile(0.95).item() # Compute 95% precipitation threshold 
    pr_max = precip_mean.max().item() # Max value on x-axis 

    # Make scatterplot 
    splot = ax.scatter(
        x, y,
        color=scattercolor
    )
    # Overlay histogram of number of values in each bin 
    hist2d = ax.hist2d(
        x, y, 
        bins=bins, cmin=10,
        norm=colors.LogNorm(), 
        cmap=cmap,
        zorder=5,
    )
    # Axis limits 
    ylim = (-0.02, 1.03)
    xlim = (0-pr_max*0.025, pr_max+pr_max*0.05)
    # Add vertical line for 95% precip threshold 
    ax.vlines(pr_thr, ylim[0], ylim[1], linestyle="dashed", color=linecolor, zorder=10, linewidth=3)

    # Add horizontal line for 0.5 probability 
    ax.hlines(0.5, 0, pr_max+3, linestyle="dotted", color=linecolor, zorder=10, linewidth=3)

    # Add plot decorators 
    if in_plot_text:
        ax.text(pr_max-pr_max*0.1, 0.5, s="non-EPCP          EPCP       ", va="center", size=fontsize, zorder=30, rotation="vertical")
        ax.text(pr_max-pr_max*0.05, 0.5, s="<---          --->", va="center",size=fontsize, zorder=30, rotation="vertical")
    ax.text(
        pr_thr, 
        1.07, 
        s="p95", 
        ha="center", 
        size=fontsize, 
        zorder=30
        )
    ax.set(
        ylabel="probability of EPCP", 
        xlabel="precipitation (mm)", 
        ylim=ylim, 
        xlim=xlim, 
        )
    if colorbar: 
        cbar = plt.colorbar(hist2d[3], label="number of days in bin")
    ax.set_title(x=0.45, y=title_y, label=title)

    # Save figure 
    if savefig: 
        plt.savefig("{0}{1}.png".format(figures_dir,figname), dpi=300, bbox_inches='tight')

    return ax

def make_precip_vs_prob_plot_by_season(predictions, savefig=True, figures_dir="", dpi=300, fontsize=10, figsize=(15,3.5), scattercolor="midnightblue", cmap="RdBu_r", in_plot_text=False, bins=25, figname="mean_precip_vs_pred_epcp_by_season"):
    """Generate figure: Monthly mean precipitation vs. predicted probability of ECPC for each timestep BY SEASON 
    Results in one figure with four subplots 

    Parameters 
    ----------
    predictions: pd.DataFrame 
        Table with columns "precip_mean", "prob_1", and "time" 
    savefig: boolean, optional 
        Save figure as png? Default to True
    figures_dir: str, optional 
        Directory to save figure to. Default to current directory 
    figsize: tuple, optional 
        Size of figure 
    cmap: matplotlib.colormap, optional
        Colormap to use for 2D histogram
    scattercolor: str, optional 
        Color to use for scatters 
    dpi: int, optional 
        Figure DPI. Default to 300
    fontsize: float, optional 
        Size of font on plot. Default to 10 
    in_plot_text: str, optional 
        Add text inside plot showing direction of EPCP? Default to True 
    figname: str, optional 
        Name to give saved figure. Do not include extension. 
        Default to "mean_precip_vs_pred_epcp"

    Returns 
    --------
    fig: matplotlib.figure.Figure

    """
    
    x = "precip_mean"
    y = "prob_1"

    # Add season as column to dataframe
    predictions["time"] = pd.to_datetime(predictions["time"].values)
    predictions["season"] = predictions["time"].dt.month.map(lambda x: get_season_str(x))

    # Get histogram for entire dataset 
    # This will be used to make the bin edges and colorbar (the figure will never be used )
    # Bins need to be the same for each season! 
    h, xedges, yedges, im = plt.hist2d(
        predictions[x],
        predictions[y],
        bins=bins
        )
    plt.close() # Close so it doesn't pop up automatically in jupyter notebook

    # Last plot needs to have a slightly higher width ratio to fit the colorbar but be the same size as the other 3 plots 
    fig, axes = plt.subplots(1, 4, figsize=figsize, width_ratios=[1,1,1,1.2])

    # Loop through each season and make a figure 
    seasons = ["DJF","MAM","JJA","SON"]
    for i in range(len(seasons)):
        predictions_one_season = predictions[predictions["season"] == seasons[i]]
        fig_one_season = make_precip_vs_prob_plot(
            predictions_one_season[x], 
            predictions_one_season[y], 
            ax=axes[i],
            bins=(xedges,yedges), 
            colorbar = True if i == len(seasons)-1 else False, # Only add colorbar to the last plot
            title=seasons[i], # Make the name of the season the title of the plot 
            title_y=1.1, 
            scattercolor=scattercolor, 
            cmap=cmap, 
            dpi=dpi,
            savefig=False, 
            in_plot_text=in_plot_text,
        )
        if i != 0: # Remove y-label for all but the first plot
            axes[i].axes.get_yaxis().set_visible(False)

    # Save figure 
    if savefig: 
        plt.savefig("{0}{1}.png".format(figures_dir,figname), dpi=300, bbox_inches='tight')
    return fig

def make_epcp_composite_plot(epcp_mean, map_crs=MAP_CRS, cmap="RdBu_r", levels=np.arange(-1, 1.2, .2), figures_dir="", savefig=True, figname="epcp_composite"): 
    """Make EPCP compositie patterns maps for each variable 

    Parameters
    ----------
    epcp_mean: xr.DataArray 
        Time-averaged data in the form of a 3D DataArray with dimensions: (variable, lat, lon)
    map_crs: cartopy.crs, optional
        Caropy projection to use for basemap 
    cmap: matplotlib.colormap, optional 
        Colormap to use for filled contours. 
        Default to Red-Blue, where blue is negative and red is positive
    levels: np.array, optional 
        Contour levels 
    figures_dir: str, optional 
        Directory to save figure to. Default to current directory 
    savefig: boolean, optional 
        Save figure as png? Default to True
    figname: str, optional 
        Name to give saved figure. Do not include extension. 

    Returns
    -------
    pl: xarray.plot.facetgrid.FacetGrid
        Figure with a separate plot for each variable 
    
    """

    # Plot settings 
    titles_per_var = {
        "hgt_detrended_anom":"500-hPa GPH anomaly", 
        "slp_anom": "Sea level pressure anomaly"
    } 
    fontsize = 10
    suptitle = "EPCP composite anomaly patterns" # Title for figure 
    data_crs = ccrs.PlateCarree()

    pl = epcp_mean.plot.contourf(
        x="lon", y="lat", col="variable", 
        transform=data_crs, 
        cmap=cmap, 
        levels=levels, 
        extend="both", 
        zorder=5,
        subplot_kws={'projection':map_crs}, 
        cbar_kwargs={"label":"anomaly (std dev)", 'orientation':'horizontal', 'shrink':0.3,}
        )

    # Add contour lines 
    # Useful references for overlaying countour lines in this manner: 
    # -- https://stackoverflow.com/questions/70672552/how-to-customize-plot-using-xarray-faceting-option
    # -- https://docs.xarray.dev/en/stable/generated/xarray.plot.FacetGrid.map_dataarray.html
    contour_lines = pl.map_dataarray(
        xr.plot.contour, 
        x="lon", y="lat", 
        transform=data_crs, 
        colors="black", 
        levels=levels, 
        zorder=5, 
        linewidths=0.6, 
        add_colorbar=False
    )

    # Beautify each plot. Add features and set titles
    for ax in pl.axs.flatten():
        # Add coastlines and state boundaries 
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'), zorder=10, linewidth=0.4)
        ax.add_feature(cfeature.STATES.with_scale('110m'), zorder=10, linewidth=0.4)

        # Get variable from axis title. Reassign title 
        var = ax.get_title().split("variable = ")[1] 
        ax.set_title(titles_per_var[var], fontsize=fontsize)

    pl.fig.suptitle(suptitle, y=1.06) # Set title for entire plot 
    
    # Save figure 
    if savefig: 
        plt.savefig("{0}{1}.png".format(figures_dir,figname), dpi=300, bbox_inches='tight')

    return pl 

def plot_metrics_by_epoch(training_history, metrics="all", figsize=(7, 6), train_color="blue", val_color="red", figures_dir="", savefig=True, figname="metrics_by_epoch"): 
    """ Plot lineplots of model metrics by epoch from model training history 

    Parameters
    -----------
    training_history: pd.DataFrame 
        Table containing model metrics for training and validation data by epoch 
    metrics: str or list of str, optional
        Metrics to plot. Must be valid column names in training_history table 
        Default to "all": all metrics in the table 
    figsize: tuple, optional 
        Size of figure 
    train_color: str, optional 
        Color for training line. Default to "blue" 
    val_color: str, optional 
        Color for validation line. Default to "red"
    figures_dir: str, optional 
        Directory to save figure to. Default to current directory 
    savefig: boolean, optional 
        Save figure as png? Default to True
    figname: str, optional 
        Name to give saved figure. Do not include extension. 

    Returns 
    -------
    pl: matplotlib.figure.Figure
    
    """
    # Deal with the metrics argument 
    if metrics == "all": # Default if no input: get the names of the metrics from the table columns 
        metrics = [metric for metric in training_history.columns.values if "val_" not in metric and metric != "epoch"]
    if type(metrics) != list:  # Convert to list so it can be looped through 
        metrics = [metrics] 

    suptitle = "Model Training Metrics by Epoch"
    fig = plt.figure(figsize=figsize)

    # Loop through each metric and add new axis with lineplot to figure 
    for i in range(len(metrics)): 
        metric = metrics[i]
        ax = fig.add_subplot(3,2,i+1)
        tr_pl = training_history.plot("epoch", metric, label="training", ax=ax, color=train_color) # Plot training line
        val_pl = training_history.plot("epoch","val_"+metric, label="validation", ax=ax, color=val_color) # Plot validation line
        ax.set_title(metric)
        ax.get_legend().remove() # Remove legend for each individual axis 

    # Add legend for figure 
    handles,labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    fig.tight_layout()
    fig.suptitle(suptitle, y=1.04)

    # Save figure 
    if savefig: 
        fig.savefig("{0}{1}.png".format(figures_dir,figname), dpi=300, bbox_inches='tight')

    return fig 


if __name__ == "__main__":
    main()