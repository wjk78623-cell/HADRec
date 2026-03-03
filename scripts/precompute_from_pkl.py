"""
🔬 从PKL文件预计算药物嵌入

从你提供的PKL文件格式直接读取并预计算：
    Index A01A: {'[F-].[Na+]', 'CC(=O)OC1=CC=CC=C1C(O)=O', ...}
    Index A02A: {'[OH-].[OH-].[Mg++]', '[MgH2]'}
    ...
"""

import os
import sys
import pickle
import torch
import argparse

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def load_pkl_and_precompute(
    pkl_file,
    output_file,
    model_name="DeepChem/ChemBERTa-77M-MLM"
):
    """
    从pkl文件加载ATC→SMILES并预计算嵌入
    
    Args:
        pkl_file: pkl文件路径
        output_file: 输出文件路径
        model_name: ChemBERTa模型名称
    """
    print("=" * 70)
    print("🔬 从PKL文件预计算药物嵌入")
    print("=" * 70)
    
    # 1. 加载pkl文件
    print(f"\n📂 加载 {pkl_file}...")
    with open(pkl_file, "rb") as f:
        atc_to_smiles_raw = pickle.load(f)
    
    # 2. 转换格式: set → list，并确保所有SMILES都是字符串
    atc_to_smiles = {}
    for atc_code, smiles_set in atc_to_smiles_raw.items():
        if isinstance(smiles_set, set):
            atc_to_smiles[atc_code] = [str(s) for s in smiles_set]
        elif isinstance(smiles_set, list):
            atc_to_smiles[atc_code] = [str(s) for s in smiles_set]
        else:
            atc_to_smiles[atc_code] = [str(smiles_set)]
    
    print(f"✅ 加载了 {len(atc_to_smiles)} 个ATC code")
    
    # 统计信息
    single_mol = sum(1 for v in atc_to_smiles.values() if len(v) == 1)
    multi_mol = len(atc_to_smiles) - single_mol
    print(f"  - 单分子: {single_mol}")
    print(f"  - 多分子: {multi_mol}")
    
    # 显示前3个示例
    print("\n📋 数据示例:")
    for i, (atc, smiles_list) in enumerate(list(atc_to_smiles.items())[:3]):
        print(f"  {atc}: {len(smiles_list)} 个分子")
        for s in smiles_list[:2]:  # 只显示前2个
            print(f"    - {s[:50]}{'...' if len(s) > 50 else ''}")
    
    # 3. 调用预计算函数
    print(f"\n🚀 开始编码（使用 {model_name}）...")
    
    from llm.drug_knowledge_module import load_drug_knowledge_from_pkl
    drug_embeddings, atc4_to_idx = load_drug_knowledge_from_pkl(
        pkl_file=pkl_file,
        output_file=output_file,
        model_name=model_name
    )
    
    print("\n" + "=" * 70)
    print("✅ 完成！")
    print(f"  - 输出文件: {output_file}")
    print(f"  - 药物数量: {len(atc4_to_idx)}")
    print(f"  - 嵌入维度: {drug_embeddings.shape[1]}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="从PKL文件预计算药物嵌入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/precompute_from_pkl.py \\
      --pkl_file data/mimic3/idx2SMILES.pkl \\
      --output_file data/handled/drug_embeddings.pt
        """
    )
    
    parser.add_argument(
        "--pkl_file",
        type=str,
        required=True,
        help="PKL文件路径（格式: {atc_code: {smiles1, smiles2, ...}}）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出文件路径（.pt格式）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="DeepChem/ChemBERTa-77M-MLM",
        help="ChemBERTa模型名称"
    )
    
    args = parser.parse_args()
    
    # 检查文件存在性
    if not os.path.exists(args.pkl_file):
        print(f"❌ 错误：文件不存在: {args.pkl_file}")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 执行预计算
    load_pkl_and_precompute(
        pkl_file=args.pkl_file,
        output_file=args.output_file,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()

