# COVID-19 Open Research Dataset Analysis

This project performs basic analysis on the CORD-19 dataset using Python.

## Setup

Install required Python packages:

```
pip install pandas numpy matplotlib seaborn mlcroissant
```

Optionally, install wordcloud for additional visualizations:

```
pip install wordcloud
```

## Usage

Run the analysis script:

```
python assignment.py
```

The script fetches the CORD-19 dataset programmatically using the `mlcroissant` library and performs exploratory data analysis, generating plots and saving a cleaned sample CSV in the `cord19_outputs` directory.

## Notes

- The dataset is fetched from the Croissant JSON-LD endpoint for the CORD-19 Kaggle challenge.
- All plots and outputs are saved in the `cord19_outputs` folder.
- You can use the saved sample CSV for further analysis or in a Streamlit app.
