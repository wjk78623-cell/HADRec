import os
import dill
import jsonlines
import pandas as pd
from utils import *  # 你的工具函数，比如 med_process、codeMapping2atc4 等

# ========== Step 0: 配置文件路径 ==========
base_dir = ""  # 你的数据根目录，或者留空表示当前目录

# 原始数据路径
med_file = os.path.join(base_dir, "H:/mimic/PRESCRIPTIONS.csv")
diag_file = os.path.join(base_dir, "H:/mimic/DIAGNOSES_ICD.csv")
procedure_file = os.path.join(base_dir, "H:/mimic/PROCEDURES_ICD.csv")
admission_file = os.path.join(base_dir, "H:/mimic/ADMISSIONS.csv")

# 辅助文件路径
RXCUI2atc4_file = os.path.join(base_dir, "./auxiliary/RXCUI2atc4.csv")
cid2atc6_file = os.path.join(base_dir, "./auxiliary/drug-atc.csv")
ndc2RXCUI_file = os.path.join(base_dir, "./auxiliary/ndc2RXCUI.txt")
ddi_file = os.path.join(base_dir, "./auxiliary/drug-DDI.csv")
drugbankinfo = os.path.join(base_dir, "./auxiliary/drugbank_drugs_info.csv")
atc2drug_file = os.path.join(base_dir, "./auxiliary/WHO ATC-DDD 2021-12-03.csv")

# 输出路径
file_path = os.path.join(base_dir, "./handled/")
os.makedirs(file_path, exist_ok=True)
os.makedirs(os.path.join(file_path, "full"), exist_ok=True)

med_structure_file = os.path.join(file_path, "atc32SMILES.pkl")
ddi_adjacency_file = os.path.join(file_path, "full/ddi_A_final.pkl")
ehr_adjacency_file = os.path.join(file_path, "full/ehr_adj_final.pkl")
ehr_sequence_file = os.path.join(file_path, "full/records_final.pkl")
vocabulary_file = os.path.join(file_path, "full/voc_final.pkl")
ddi_mask_H_file = os.path.join(file_path, "full/ddi_mask_H.pkl")
atc3toSMILES_file = os.path.join(file_path, "full/atc3toSMILES.pkl")

# ========== 词表类定义 ==========
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

def create_str_token_mapping(df, vocabulary_file):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    for _, row in df.iterrows():
        diag_voc.add_sentence(row["ICD9_CODE"])
        med_voc.add_sentence(row["ATC3"])
        pro_voc.add_sentence(row["PRO_CODE"])
    dill.dump({"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
              open(vocabulary_file, "wb"))
    return diag_voc, med_voc, pro_voc

# ========== Step 1: 药物数据处理 ==========
print("Processing medications...")
med_pd = med_process(med_file)
med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
med_pd = med_pd.merge(med_pd_lg2[["SUBJECT_ID"]], on="SUBJECT_ID", how="inner").reset_index(drop=True)
med_pd = codeMapping2atc4(med_pd, ndc2RXCUI_file, RXCUI2atc4_file)
med_pd = filter_300_most_med(med_pd)

# SMILES 映射
atc3toDrug_map = ATC3toDrug(med_pd)
druginfo = pd.read_csv(drugbankinfo)
atc3toSMILES_map = atc3toSMILES(atc3toDrug_map, druginfo)
dill.dump(atc3toSMILES_map, open(atc3toSMILES_file, "wb"))
med_pd = med_pd[med_pd.ATC3.isin(atc3toSMILES_map.keys())]
print("Medication processing complete.")

# ========== Step 2: 诊断和手术处理 ==========
print("Processing diagnoses and procedures...")
diag_pd = diag_process(diag_file)
pro_pd = procedure_process(procedure_file)
print("Diagnosis and procedure processing complete.")

# 合并数据
data = combine_process(med_pd, diag_pd, pro_pd)
print("Data combined.")

# ========== Step 3: 加入病人基本信息 ==========
admission = pd.read_csv(admission_file)
data = pd.merge(data, admission[["HADM_ID", "INSURANCE", "LANGUAGE", "RELIGION", "MARITAL_STATUS", "ETHNICITY", "DIAGNOSIS"]],
                how="left", on="HADM_ID")
data.fillna("unknown", inplace=True)

# 创建词表
diag_voc, med_voc, pro_voc = create_str_token_mapping(data, vocabulary_file)
print("Vocabulary created.")

# ========== Step 4: ATC / ICD码映射 ==========
RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
RXCUI2atc4["NDC"] = RXCUI2atc4["NDC"].map(lambda x: x.replace("-", "") if isinstance(x, str) else x)
with open(ndc2RXCUI_file, "r") as f:
    ndc2RXCUI = eval(f.read())
RXCUI2ndc = dict(zip(ndc2RXCUI.values(), ndc2RXCUI.keys()))
RXCUI2atc4["RXCUI"] = RXCUI2atc4["RXCUI"].astype(str)
RXCUI2atc4["NDC"] = RXCUI2atc4["RXCUI"].map(RXCUI2ndc)
RXCUI2atc4.dropna(inplace=True)
RXCUI2atc4.drop_duplicates(inplace=True)
RXCUI2atc4["ATC4"] = RXCUI2atc4["ATC4"].map(lambda x: x[:4] if isinstance(x, str) else x)

atc2drug_df = pd.read_csv(atc2drug_file)
atc2drug_df = atc2drug_df[atc2drug_df["atc_code"].map(len) == 4]
atc2drug_df.rename(columns={"atc_code": "ATC4"}, inplace=True)
atc2drug_df.drop(columns=["ddd", "uom", "adm_r", "note"], inplace=True)
atc2drug_df["atc_name"] = atc2drug_df["atc_name"].map(lambda x: x.lower())
atc2drug_dict = dict(zip(atc2drug_df["ATC4"], atc2drug_df["atc_name"]))

icd2diag_df = pd.read_csv(os.path.join(base_dir, "H:/mimic/D_ICD_DIAGNOSES.csv"))
icd2diag_dict = dict(zip(icd2diag_df["ICD9_CODE"].astype(str), icd2diag_df["SHORT_TITLE"]))
icd2proc_df = pd.read_csv(os.path.join(base_dir, "H:/mimic/D_ICD_PROCEDURES.csv"))
icd2proc_dict = dict(zip(icd2proc_df["ICD9_CODE"].astype(str), icd2proc_df["SHORT_TITLE"]))

def decode(code_list, decoder):
    return [decoder.get(code, "unknown") for code in code_list]

data["drug"] = data["ATC3"].map(lambda x: decode(x, atc2drug_dict))
data["diagnosis"] = data["ICD9_CODE"].map(lambda x: decode(x, icd2diag_dict))
data["procedure"] = data["PRO_CODE"].map(lambda x: decode(x, icd2proc_dict))

# ========== Step 5: 构造 Prompt ==========
main_template = "The patient has <VISIT_NUM> times ICU visits. \n <HISTORY> In this visit, he has diagnosis: <DIAGNOSIS>; procedures: <PROCEDURE>. Then, the patient should be prescribed: "
hist_template = "In <VISIT_NO> visit, the patient had diagnosis: <DIAGNOSIS>; procedures: <PROCEDURE>. The patient was prescribed drugs: <MEDICATION>. \n"

def concat_str(str_list):
    return ", ".join(str_list)

llm_data = []
for subject_id in data["SUBJECT_ID"].unique():
    item_df = data[data["SUBJECT_ID"] == subject_id]
    visit_num = item_df.shape[0] - 1
    patient_hist = []

    for visit_no, (_, row) in enumerate(item_df.iterrows()):
        drug_str, diag_str, proc_str = concat_str(row["drug"]), concat_str(row["diagnosis"]), concat_str(row["procedure"])
        patient_hist.append(hist_template.replace("<VISIT_NO>", str(visit_no+1))
                                          .replace("<DIAGNOSIS>", diag_str)
                                          .replace("<PROCEDURE>", proc_str)
                                          .replace("<MEDICATION>", drug_str))
    patient_hist.pop()  # 去除最后一次记录（作为预测目标）
    if len(patient_hist) > 3:
        patient_hist = patient_hist[-3:]
    hist_str = "".join(patient_hist)
    last_row = item_df.iloc[-1]
    input_str = main_template.replace("<VISIT_NUM>", str(visit_num)) \
                             .replace("<HISTORY>", hist_str) \
                             .replace("<DIAGNOSIS>", concat_str(last_row["diagnosis"])) \
                             .replace("<PROCEDURE>", concat_str(last_row["procedure"]))
    llm_data.append({
        "input": input_str,
        "target": concat_str(last_row["drug"]),
        "drug_code": [str(x) for x in last_row["ATC3"]],
        "subject_id": int(subject_id)
    })

# ========== Step 6: 保存数据 ==========
def save_data(filename, data):
    with jsonlines.open(os.path.join(file_path, filename), "w") as w:
        for meta_data in data:
            w.write(meta_data)

train_split = int(len(llm_data) * 0.8)
val_split = int(len(llm_data) * 0.1)
train = llm_data[:train_split]
val = llm_data[train_split:train_split + val_split]
test = llm_data[train_split + val_split:]

save_data("train_0105.json", train)
save_data("val_0105.json", val)
save_data("test_0105.json", test)

print("✅ 数据预处理完成")
