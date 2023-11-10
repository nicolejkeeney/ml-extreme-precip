
import make_figures
import train_model
import preprocess_chirps
import time 
from utils.misc_utils import add_to_settings_json

# ------------ SET GLOBAL VARIABLES ------------

# Model ID needs to match string key in model_settings.json
MODEL_ID = "frances_iowa"

# Directories. Needs to have a slash (/) after (i.e "dir/"")
DATA_DIR = "../data/input_data_preprocessed/" 
OUTPUT_DIR = "../model_output/"+MODEL_ID+"/" 
FIGURES_DIR = "../figures/"+MODEL_ID+"/" 

# Save trained tensorflow model? 
SAVE_MODEL = False


def run_for_all_states(data_dir=DATA_DIR, output_dir="../model_output/", figures_dir="../figures/", save_model=SAVE_MODEL):
    """Preprocess input labels and run for all states in list"""
    
    start_time = time.time()
    features_geom = "CONUS" # Geometry of the features data 
    state_names = ["Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
    
    for state in state_names:
        print("Starting for {0}...".format(state))

        # Set directories 
        # (1) Preprocess CHIRPS data for that state 
        raw_data_dir = "../data/"
        print("Preprocessing CHIRPS data...")
        preprocess_chirps.main(
            geom_name=state, 
            data_dir=raw_data_dir
            )
        print("CHIRPS data preprocessing complete!")

        # (2) Set hyperparameters
        model_id = "frances_"+state.replace(" ", "_") # Replace spaces with underscores in state name 
        settings = {
            "model_id": model_id,  
            "labels_geom": state.replace(" ", "_"),
            "features_geom": features_geom,
            "epochs": 500,
            "batch_size": 2048,
            "learning_rate": 0.004,
            "activity_reg": 0.001,
            "conv_filters": 16,
            "dense_neurons": 16,
            "dense_layers": 1,
            "random_seed": 333,
            "dropout_rate": 0.2
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


def main(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR, model_id=MODEL_ID, save_model=SAVE_MODEL):

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