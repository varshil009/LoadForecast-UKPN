# Load Forecast for UKPN Power Networks Portal Data

This project focuses on load forecasting using data from the **UK Power Networks Open Data Portal**. The dataset is updated daily and includes historical electricity flow measurements from 2021 to 2024. With over 12 million data points, this dataset provides a comprehensive view of the grid's performance.

---

## Dataset Overview

### Data Source
The data is sourced from the **UK Power Networks Open Data Portal**. It covers electricity flow information from January 2021 to the present (2024).

### Key Features
The dataset includes the following columns:

- **Date**: Timestamps of data recording.
- **GSP (Grid Supply Point)**: Name of the grid supply point.
- **SGT (Super Grid Transformer)**: Identifier of the transformer.
- **Voltage Metrics**:
  - `Voltage_Min`: Minimum voltage recorded.
  - `Voltage_Max`: Maximum voltage recorded.
  - `Voltage_Avg`: Average voltage recorded.
- **Active Power Metrics**:
  - `ActivePower_Min`: Minimum active power recorded.
  - `ActivePower_Max`: Maximum active power recorded.
  - `ActivePower_Avg`: Average active power recorded.
- **Reactive Power Metrics**:
  - `ReActivePower_Min`: Minimum reactive power recorded.
  - `ReActivePower_Max`: Maximum reactive power recorded.
  - `ReActivePower_Avg`: Average reactive power recorded.
- **Current Metrics**:
  - `Current_Min`: Minimum current recorded.
  - `Current_Max`: Maximum current recorded.
  - `Current_Avg`: Average current recorded.

---

## Methodology

A methodology diagram will be added to represent the workflow of the project. This includes steps like data preprocessing, feature engineering, modeling, and evaluation.

**Diagram Placeholder**

![image](https://github.com/user-attachments/assets/e3db8e9b-7c24-4053-b9ae-c24b3eb0610f)


---

## Load Forecasting Approach

The methodology for load forecasting is as follows:

1. **Data Cleaning and Preprocessing**:
   - Handle missing values and outliers.
   - Normalize numerical features to standardize the range.

2. **Feature Engineering**:
   - Create lag features and rolling statistics to capture temporal patterns.
   - Encode categorical variables like `GSP` and `SGT`.

3. **Model Selection**:
   - Utilize advanced neural networks such as LSTMs or Transformers for capturing time-dependent patterns.
   - Incorporate exogenous variables like weather or market data to improve model accuracy.

4. **Training and Validation**:
   - Train models using historical data and evaluate their performance using metrics like Mean Absolute Error (MAE) and R².

5. **Forecasting**:
   - Predict future load profiles using the trained models.

---

## Data Update Frequency

The dataset is updated daily to ensure forecasts are based on the most recent data. Automated pipelines are implemented for continuous updates.

---

## Potential Applications

- Real-time load forecasting for efficient grid management.
- Detection of anomalies in power consumption patterns.
- Insights for energy trading and demand response programs.

---

## How to Use

1. Clone this repository.
2. Download the latest dataset from the UK Power Networks Open Data Portal. The dataset is found [here](https://ukpowernetworks.opendatasoft.com/explore/dataset/ukpn-super-grid-transformer/information/)
3. The models.ipynb contains 2 models, in which you can either load pre-trained TAT which is here in repository named TAT2_weekly or you can train both models. Please keep the TAT.py and nn_forecast.py in same folder with your current running file.
4. The LSTM have different code file and you can directly run the LSTMSmartGrid.ipynb
5. Also included the errors.py file where you can just import the error_df function and give results with test values to get table for results with model name. 

---

## Acknowledgments

This project utilizes data from the **UK Power Networks Open Data Portal**. We acknowledge their contribution to making this data openly available for research and development. [UKPN Data Portal link](https://ukpowernetworks.opendatasoft.com/

---
