# Cuxfilter Visualization Agent
---
A conversational cuxfilter visualization agent powered by NVIDIA GPUs.  
You can interact with it using natural language to run data exploration and interactive visualization tasks with minimal setup.

---

## 📊 Sample Dataset

For this project, you can use the classic datasets, [nyc_taxi_wide.parquet](https://s3.amazonaws.com/datashader-data/nyc_taxi_wide.parq
), [auto_accidents.arrow](https://data.rapids.ai/viz-data/auto_accidents.arrow.gz), or [146M_predicitions_v2.arrow](https://data.rapids.ai/viz-data/146M_predictions_v2.arrow.gz) 

Download the dataset and place it in the `data/` directory before running the agent.

---

## Package Installation

This project requires the following versions to ensure full compatibility:<br>
* cuxfilter: 25.10
* RAPIDS: 25.10
* Bokeh: <= 3.6.0

Please refer to the official [RAPIDS installation documentation](https://docs.rapids.ai/install/) for detailed instructions.

### Installation example: <br>
```bash
conda create -n rapids-25.10 -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.10 python=3.11 'cuda-version=13.0' \
    jupyterlab
```

## Running the Agent

The agent must be run in GPU mode:

```bash
conda activate rapids-25.10
export NVIDIA_API_KEY=""
streamlit run user_interface.py
```

### The agent supports queries such as:<br>
   create an interactive dashboard<br>
   analyze the data<br>
   dashboard history<br>
   ...

---

**Note:**  
- Ensure you have the appropriate dependencies installed.  
- It requires a supported NVIDIA GPU and the RAPIDS ecosystem installed.
