```markdown
# Interactive SRI Congestion Severity Map

This folder contains the interactive spatial-temporal congestion severity visualisation for the Nottingham traffic speed forecasting study.

The visualisation compares ground-truth and B5-predicted congestion severity across four forecasting horizons: 15, 30, 45 and 60 minutes.

## How To Open

Open the following file in a web browser:

```text
interactive_sri_three_class_map_B5_full/index.html
```

The page can be opened locally. It uses local JavaScript data files stored in:

```text
interactive_sri_three_class_map_B5_full/data/
```

An internet connection is required for the OpenStreetMap base map tiles to load.

## What The Map Shows

The map allows users to select a prediction origin date and time. For each selected origin time, the page displays ground-truth and B5-predicted congestion severity maps for the 15, 30, 45 and 60 minute forecasting horizons.

Each traffic sensor node is represented by a coloured point:

| Colour | Class | SRI threshold |
| --- | --- | --- |
| Green | Non-congested | SRI <= 0.10 |
| Yellow | Congestion | 0.10 < SRI <= 0.40 |
| Red | Severe congestion | SRI > 0.40 |

Clicking a node shows its node ID, countline ID, direction, speed, SRI value, congestion class, target time and forecasting horizon.

## Evaluation Scope

The visualisation is based on valid testing prediction samples from the B5 model output. Because the forecasting samples use a fixed historical input window and a maximum 60-minute prediction horizon, boundary time steps at the beginning and end of each testing day are not included.

The available prediction origin times run from 01:00 to 23:00 for each testing day. They are therefore the valid within-day testing origins used by the model outputs, rather than every 5-minute interval in a complete 24-hour calendar day.

## Model

B5 is the final selected configuration of the proposed speed forecasting model. The congestion severity classes shown in this map are derived from predicted and ground-truth speed values using the Speed Reduction Index (SRI).

## Folder Structure

```text
interactive_sri_three_class_map_B5_full/
  index.html
  data/
    sri_map_YYYY-MM-DD.js
```

The `data/` folder must remain next to `index.html` for the interactive page to work.
```