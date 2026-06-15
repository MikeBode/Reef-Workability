"# Reef-Workability" 

Builds models for reef workability based on three datasets of historical reef visits.

uv used for dependency management.

Entrypoint at download_and_process_all.py, this will download all relevant and unsupplied data. Some visualizations will be written to PlotOutputs/ and report_visualizations.py.

Fully trained model will be output to best_model.pickle. If you wish to run this model in your own application, please make sure to import models/model_training.py for the requisite class.

This repository duplicates work done by https://github.com/BryceStansfield/Reef-Workability, which extends on https://github.com/Parsayarya/Reef-Workability.
