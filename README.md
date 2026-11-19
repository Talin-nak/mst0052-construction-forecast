# Forecasting Regional Construction Activity in Norway

Course project for **MST0052 Predictive Modelling with Machine Learning, BI**.

## Setup

```bash
conda env create -f environment.yml
conda activate mst0052
```

## Run order

To reproduce the results in the report, skip step 1 and use the included
`panel_region_month.csv` (built 18 October 2026). Re-running step 1 downloads
the latest data from SSB and Norges Bank, which gives slightly different results.

1. `python build_dataset_b.py`: downloads the data and builds `panel_region_month.csv`
2. `python explore_data.py`: exploratory analysis and figures
3. `python model.py`: model training, cross-validation and test evaluation

## Data sources

- Statistics Norway (SSB), tables **08843**, **14543** and **01222**
- Norges Bank, policy rate series **IR/M.KPRA.SD**

## Licences

- SSB data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Norges Bank data: [NLOD 2.0](https://data.norge.no/nlod/en/2.0)
