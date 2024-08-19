<div align="center">
<img src="data\readme_images\banner.webp" alt="banner.webp" style="border-radius: 15px; width: 100%; max-width: 2000; height: 10%;">
<br>
&nbsp;

# Re-ranking-optimization
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-purple.svg)](https://www.python.org/downloads/release/python-3110/)
[![Langchain](https://img.shields.io/badge/Langchain-0.2.11-blue)](https://python.langchain.com/v0.2/docs/introduction/)
[![Llama Index](https://img.shields.io/badge/Llama%20Index-0.10.59-orange)](https://docs.llamaindex.ai/en/stable/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34.0-red)](https://streamlit.io/)
![License](https://img.shields.io/badge/License-Hitachi-green.svg)

&nbsp;
</div>

## About


## Getting Started

1. Clone the repository and navigate to the directory
```bash
git clone https://github.com/satyanshu404/Re-ranking-optimization.git
cd Re-ranking-optimization

```
2. Install the requirements
```bash
pip install -r requirements.txt
```
3. Download the MS MARCO dataset, if needed
```bash
DATA_DIR=./data/MSMARCO
mkdir -p ${DATA_DIR}

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz -P ${DATA_DIR}
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz -P ${DATA_DIR}
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz -P ${DATA_DIR}
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz -P ${DATA_DIR}
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz -P ${DATA_DIR}
```