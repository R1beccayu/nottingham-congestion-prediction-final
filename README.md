# Nottingham Traffic Speed Forecasting and Congestion Prediction

This repository contains selected materials from a master's research project on short-term urban traffic speed forecasting in Nottingham, UK. The core task is to predict traffic speed at 15, 30, 45, and 60 minute forecasting horizons using a spatio-temporal graph neural network model. The predicted speeds are then used for downstream congestion analysis through the Speed Reduction Index (SRI).

The main focus of the project is the **speed forecasting model**. Congestion prediction and the interactive map are derived from the speed forecasting outputs and are included as interpretable applications of the model results.

## Project Overview

The project uses traffic speed data from Nottingham road sensor nodes to model short-term changes in urban traffic conditions. The forecasting pipeline includes:

- Auditing and cleaning raw 1-minute traffic speed observations.
- Aggregating traffic speeds to 5-minute intervals.
- Selecting stable sensor nodes for model input.
- Building an OpenStreetMap (OSM)-based road-network adjacency matrix.
- Adding weather, time-calendar, and node-level contextual features.
- Training deep learning models for multi-horizon traffic speed prediction.
- Comparing the proposed model with temporal, graph-based, and Transformer-based baselines.
- Converting predicted speeds into SRI values for binary and multi-class congestion evaluation.

## Technical Environment

The model experiments were implemented using:

- Python
- PyTorch
- CUDA
- GPU acceleration for neural network training and inference
- NumPy, pandas, and scikit-learn for data processing and evaluation
- OpenStreetMap and Leaflet for spatial visualisation

CUDA-enabled GPU acceleration was used to speed up the training and inference of the neural forecasting models.

## Speed Forecasting Task

At each prediction origin time, the model uses the recent traffic speed history to predict future speeds at four horizons:

| Horizon | Target time |
| --- | --- |
| 15 min | t + 15 min |
| 30 min | t + 30 min |
| 45 min | t + 45 min |
| 60 min | t + 60 min |

The predicted speed values are evaluated against the corresponding ground-truth speeds using standard forecasting metrics such as MAE, RMSE, and MAPE. Congestion indicators are calculated only after speed prediction, so the congestion analysis depends directly on the quality of the speed forecasting model.

## Proposed Speed Forecasting Model

The final selected speed forecasting configuration is referred to as **B5**. It is based on a shared STGCN-style spatio-temporal encoder and includes two key components:

- **Spatio-temporal enhancement**, which incorporates recent, daily, and weekly traffic patterns.
- **Time-calendar additive bias**, which accounts for recurring time-of-day and calendar-related traffic regularities.

The model was designed to improve forecasting accuracy and long-horizon stability while maintaining efficient inference for practical short-term traffic prediction.

## Speed Forecasting Baselines

The proposed model was evaluated against several baseline methods, including:

- Historical Average (HA)
- LSTM
- CNN-LSTM
- STGCN
- STSGCN
- iTransformer

These baselines cover simple historical-pattern methods, temporal neural models, graph-based spatio-temporal models, and Transformer-based time-series models. They were used to assess whether the proposed model improved speed prediction performance across different forecasting horizons and traffic periods.

CNN-LSTM is included in the code and was tested during the experiments. However, because its results were close to those of LSTM, it was not discussed separately in the final dissertation text in order to keep the baseline comparison concise.

## Ablation Study

The ablation study was used to examine the contribution of each model component. Two main ablation series were used.

### A-Series Ablation

The A-series tests the effect of spatio-temporal enhancement and weather gating without the node-level time-calendar baseline.

| Configuration | Spatio-temporal enhancement | Weather gate | Time-calendar bias | Description |
| --- | --- | --- | --- | --- |
| A0 | Off | Off | Off | Backbone configuration |
| A1 | On | Off | Off | Spatio-temporal enhancement only |
| A2 | Off | On | Off | Weather gate only |
| A3 | On | On | Off | Combined A-series model |
| A4 | On | On | Off | A3 with additional spatial attention |

### Note on A4 Spatial Attention

The spatial attention module was tested experimentally as configuration **A4**. It reweights neighbouring nodes on the fixed OSM graph. However, this component was not included in the main dissertation model description because of space limitations and because its performance gain was not large enough to justify adding another architectural section in the final write-up.

For transparency, A4 is still documented here as an experimental ablation configuration.

### B-Series Ablation

The B-series introduces the time-calendar additive bias and tests whether the other enhancement modules remain useful once explicit time-calendar information is included.

| Configuration | Spatio-temporal enhancement | Weather gate | Time-calendar bias | Description |
| --- | --- | --- | --- | --- |
| B4 | Off | Off | On | Time-calendar bias only |
| B5 | On | Off | On | Final selected model |
| B6 | Off | On | On | Weather gate with time-calendar bias |
| B7 | On | On | On | All three components enabled |

The final model was selected as **B5**, because it provided the most consistent balance of accuracy, robustness, and horizon stability across the evaluated forecasting horizons.

## Downstream Congestion Definition

After speed forecasting, congestion severity is calculated using the Speed Reduction Index (SRI), where larger SRI values represent more severe congestion. The study uses the following three-class congestion scheme:

| Class | SRI threshold |
| --- | --- |
| Non-congested | SRI <= 0.10 |
| Congested | 0.10 < SRI <= 0.40 |
| Severe congestion | SRI > 0.40 |

The same thresholds are applied to both ground-truth and predicted speeds.

## Interactive Spatial-Temporal Congestion Map

The repository includes an interactive map for inspecting congestion severity across Nottingham sensor nodes. The map compares ground-truth and B5-predicted congestion classes across the four forecasting horizons.

Open:

```text
interactive_sri_three_class_map_B5_full/index.html
```

The page can be opened locally in a web browser. The data files are stored in:

```text
interactive_sri_three_class_map_B5_full/data/
```

The `data/` folder must remain next to `index.html` for the interactive page to work.

An internet connection is required for the OpenStreetMap base map tiles to load.

## Map Features

The interactive map allows users to:

- Select a testing date.
- Select a valid prediction origin time.
- Compare ground-truth and predicted congestion maps.
- View 15, 30, 45, and 60 minute forecasting horizons.
- Click individual nodes to inspect node-level details.

Each node popup includes:

- Node ID
- Countline ID
- Traffic direction
- Speed
- SRI value
- Congestion class
- Target time
- Forecasting horizon

## Available Time Range

The interactive visualisation uses valid testing prediction origins from the model output. Because the model requires a historical input window and predicts up to 60 minutes ahead, boundary times at the beginning and end of each day are excluded.

Available origin times are therefore:

```text
01:00-23:00
```

at 5-minute intervals for the included testing dates.

## Repository Structure

```text
.
├── README.md
└── interactive_sri_three_class_map_B5_full/
    ├── index.html
    └── data/
        └── sri_map_YYYY-MM-DD.js
```

## Notes

- The interactive map is intended for visual interpretation of the congestion prediction results.
- The visualisation is based on valid testing samples only.
- The map shows node-level congestion states rather than continuous road-segment congestion.
- A4 is included as an experimental ablation setting, but it is not part of the main dissertation model description.
