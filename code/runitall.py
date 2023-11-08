
import make_figures
import train_model

# ------------ SET GLOBAL VARIABLES ------------

# Model ID needs to match string key in model_settings.json
MODEL_ID = "frances_iowa"

# Directories. Needs to have a slash (/) after (i.e "dir/"")
DATA_DIR = "../data/input_data_preprocessed/" 
OUTPUT_DIR = "../model_output/"+MODEL_ID+"/" 
FIGURES_DIR = "../figures/"+MODEL_ID+"/" 

# Save trained tensorflow model? 
SAVE_MODEL = False


def main():

    # Train model 
    model, predictions = train_model.main(
        data_dir=DATA_DIR, 
        output_dir=OUTPUT_DIR, 
        figures_dir=FIGURES_DIR, 
        model_id=MODEL_ID, 
        save_training_history=True, 
        save_model=SAVE_MODEL, 
        save_model_metrics=True, 
        save_predictions=True
    )

    # Make figures 
    make_figures.main(
        model_id=MODEL_ID, 
        data_dir=DATA_DIR, 
        model_output_dir=OUTPUT_DIR, 
        figures_dir=FIGURES_DIR
    )

if __name__ == "__main__":
    main()