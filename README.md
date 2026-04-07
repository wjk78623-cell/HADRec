# LEADER: A Hierarchy-Aware Drug Recommendation Framework by Fusing Molecular Knowledge and Electronic Health Record

Electronic Health Records (EHR)-based medication recommendation models have demonstrated significant clinical value in promoting precision medicine and enabling personalized treatment. However, existing approaches face two major challenges: **insufficient integration of drug molecular knowledge** and **inadequate handling of hierarchical medication taxonomies**. 

To address these challenges, this study proposes the **LEADER (Large Language Model Enhanced Medication Recommendation)** framework, which achieves accurate and interpretable medication prediction through hierarchical prediction and cross-attention drug knowledge fusion. Specifically, this method first performs LoRA fine-tuning of LLaMA-7B on clinical narratives with hierarchical ATC labels. Subsequently, a cross-attention mechanism integrates molecular drug embeddings to enhance drug-specific representations. Finally, a 4-layer defense strategy addresses severe class imbalance in medication data.

This innovative solution not only greatly improves prediction accuracy but also provides interpretable hierarchical predictions and validated causal reasoning capability, offering a practical technical approach for clinical decision support systems.

## Framework

<img width="1980" height="1057" alt="fig1" src="https://github.com/user-attachments/assets/f8a7acaa-bfe1-47a2-a0f5-88310fd2d784" />


## Data Format

All data files are in **JSONL** format (one JSON object per line). All datasets used in this project are derived from the **MIMIC-III** clinical database.

We provide the complete data preprocessing pipeline. The processed datasets follow a hierarchical structure aligned with the ATC classification system.


### `train_atc_hierarchy2.json` / `val_atc_hierarchy2.json` / `test_atc_hierarchy2.json`

Each line is a training sample:

```json
{
  "input": "Currently, in the most recent visit, the patient presents with Diabetes mellitus type II, Hypertension, Hyperlipidemia and receives the following procedures: None. Based on these findings, the recommended medications are: ",
  "target": "insulins and analogues, beta blocking agents, ace inhibitors, plain, lipid modifying agents, plain",
  "subject_id": 12345,
  "atc_level_1": ["A", "C"],
  "atc_level_2": ["A1", "C0", "C1"],
  "atc_level_3": ["A10", "C07", "C09", "C10"],
  "atc_level_4": ["A10A", "C07A", "C09A", "C10A"]
}
```

### `drug_embeddings.pt`

PyTorch tensor file storing molecular drug embeddings:

```python
# Shape: (num_drugs, embedding_dim)
# Generated from drug SMILES strings using molecular fingerprints
drug_embeddings = torch.load('data/mimic3/l_data/drug_embeddings.pt')
```

---

### Step 1: Download MIMIC-III

Apply for access at [MIMIC-III Clinical Database](https://mimic.mit.edu/).

Download and place the following files in `data/mimic3/raw/`:
- `PRESCRIPTIONS.csv`
- `DIAGNOSES_ICD.csv`
- `PROCEDURES_ICD.csv`

### Step 2: Download Auxiliary Files

- **`drug-DDI.csv`**: Download from [GAMENet](https://github.com/sjy1203/GAMENet)
  - Contains drug-drug interaction pairs and side effects
  
- **`drug-atc.csv`**: Download from [GAMENet](https://github.com/sjy1203/GAMENet)
  - Maps CID (PubChem Compound ID) to ATC codes
  
- **`ndc2rxcui.txt`**: Download from [GAMENet](https://github.com/sjy1203/GAMENet)
  - Maps NDC (National Drug Code) to RXCUI
  - Named as `ndc2rxnorm_mapping.csv` in the repository

Place all files in `data/mimic3/auxiliary/`.

### Step 3: Preprocess Data

```bash
cd data/mimic3
python construction.py
```



### Run Train Model with LoRA

```bash
bash experiment/llm_cls.bash
```

---

###  Evaluate on Test Set

```bash
python final_evaluation_with_bootstrap.py
```



## Data Preparation

### File Structure

```text
data/
└── mimic3/
    ├── raw/                           # Raw MIMIC-III data
    │   ├── PRESCRIPTIONS.csv          # Medication prescriptions
    │   ├── DIAGNOSES_ICD.csv          # ICD-9 diagnoses
    │   └── PROCEDURES_ICD.csv         # ICD-9 procedures
    ├── auxiliary/                     # Auxiliary files
    │   ├── drug-DDI.csv               # Drug-drug interactions (from GAMENet)
    │   ├── drug-atc.csv               # CID to ATC mapping (from GAMENet)
    │   ├── ndc2rxcui.txt              # NDC to RXCUI mapping
    │   └── idx2SMILES.pkl             # Drug index to SMILES strings
    └── l_data/                        # Preprocessed data (generated)
        ├── train_atc_hierarchy2.json  # Training set
        ├── val_atc_hierarchy2.json    # Validation set
        ├── test_atc_hierarchy2.json   # Test set
        └── drug_embeddings.pt         # Molecular drug embeddings
```



### Project Structure

```text
├── data/
│   └── mimic3/
│       ├── raw/
│       ├── auxiliary/
│       └── l_data/
├── llm/
│   ├── llama.py                       # LLaMA with hierarchical head
│   ├── drug_knowledge_module.py       # Cross-attention module
│   ├── peft/                          # LoRA implementation
│   └── data_processor/
├── generators/
│   └── data.py                        # EHRTokenizer, vocabulary
├── experiment/
│   └── llm_cls.bash                   # Training script
├── saved/
│   └── lora-atc_hierarchy2/
│       └── checkpoint-5000/           # Best checkpoint
├── results/
├── main_llm_cls.py                    # Main training/inference
├── final_evaluation_with_bootstrap.py # Evaluation
└── README.md
```





















