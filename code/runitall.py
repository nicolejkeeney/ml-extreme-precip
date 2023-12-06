
import make_figures
import train_model
import preprocess_chirps
import time 
from utils.misc_utils import add_to_settings_json
import utils.parameters as param 

# ------------ SET GLOBAL VARIABLES ------------

# Model ID needs to match string key in model_settings.json
MODEL_ID = "frances_Texas"

# Directories. Needs to have a slash (/) after (i.e "dir/"")
CHIRPS_BY_STATE_DATA_DIR = "../data/input_data_preprocessed/us_states/" 
GRIDDED_CHIRPS_DATA_DIR = "../data/input_data_preprocessed/chirps_5x5/"
OUTPUT_DIR = "../model_output/us_states/"+MODEL_ID+"/" 
FIGURES_DIR = "../figures/"+MODEL_ID+"/" 

# Save trained tensorflow model? 
SAVE_MODEL = False

# def run_for_grid(filepath=GRIDDED_CHIRPS_DATA_DIR, output_dir="../model_output/", figures_dir="../figures/", save_model=SAVE_MODEL)
#     """Preprocess input labels and run for all gridcells in input file """



def run_for_all_states(data_dir=CHIRPS_BY_STATE_DATA_DIR, output_dir="../model_output/us_states/", figures_dir="../figures/us_states/", save_model=SAVE_MODEL):
    """Preprocess input labels and run for all states in list"""
    
    start_time = time.time()
    features_geom = "CONUS" # Geometry of the features data 
    state_names = ["Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
    
    for state in state_names:
        print("Starting for {0}...".format(state))

        # Set directories 
        # (1) Preprocess CHIRPS data for that state 
        print("Preprocessing CHIRPS data...")
        preprocess_chirps.chirps_by_state(geom_name=state)
        print("CHIRPS data preprocessing complete!")

        # (2) Set hyperparameters
        model_id = "frances_"+state.replace(" ", "_") # Replace spaces with underscores in state name 
        settings = {
            "model_id": model_id,  
            "labels_geom": state.replace(" ", "_"),
            "features_geom": features_geom,
            "epochs": param.epochs,
            "batch_size": param.batch_size,
            "learning_rate": param.learning_rate,
            "activity_reg": param.activity_reg,
            "conv_filters": param.conv_filters,
            "dense_neurons": param.dense_neurons,
            "dense_layers": param.dense_layers,
            "random_seed": param.random_seed,
            "dropout_rate": param.dropout_rate
        }

        # (3) Add hyperparameters to settings dict 
        add_to_settings_json(
            m_id=model_id, 
            settings=settings, 
            settings_path="model_settings.json"
            )

        # (4) Set directories 
        output_dir_i = output_dir + model_id + "/"
        figures_dir_i = figures_dir + model_id + "/"
        
        # (5) Train model & make predictions 
        print("Training model and making predictions...")
        model, predictions = train_model.main(
            data_dir=data_dir, 
            output_dir=output_dir_i, 
            figures_dir=figures_dir_i, 
            model_id=model_id, 
            save_training_history=True, 
            save_model=save_model, 
            save_model_metrics=True, 
            save_predictions=True
        )
        print("Model trained and predictions made!")
        # (6) Make figures 
        print("Making figures...", end="")
        make_figures.main(
            model_id=model_id, 
            data_dir=data_dir, 
            model_output_dir=output_dir_i, 
            figures_dir=figures_dir_i
        )
        print(" complete!")

    print("Script complete!")
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print("Time elapsed: {0}".format(time_elapsed))


def main(data_dir=CHIRPS_BY_STATE_DATA_DIR, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR, model_id=MODEL_ID, save_model=SAVE_MODEL):

    # Train model 
    model, predictions = train_model.main(
        data_dir=data_dir, 
        output_dir=output_dir, 
        figures_dir=figures_dir, 
        model_id=model_id, 
        save_training_history=True, 
        save_model=save_model, 
        save_model_metrics=True, 
        save_predictions=True
    )

    # Make figures 
    make_figures.main(
        model_id=model_id, 
        data_dir=data_dir, 
        model_output_dir=output_dir, 
        figures_dir=figures_dir
    )

if __name__ == "__main__":
    main()