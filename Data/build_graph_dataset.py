import os
import pickle
import yaml
import io

# 从 graphs.py 中导入各个图的构造函数
from graphs import (
    build_ligand_atom_graph,
    build_protein_atom_graph,
    build_atom_interaction_graph,
    build_ligand_fragment_graph,
    build_protein_residue_graph,
    build_substructure_interaction_graph
)


def load_affinity_labels(index_file):
    """
    从 INDEX_data.2016 文件中解析亲和力标签。
    注意：文件从第七行开始存储复合物信息，
    每行的第一列为复合物名称，第四列为亲和力值（-logKd/Ki），假设以空白字符分隔。
    返回一个字典：{复合物名称: 亲和力(float)}
    """
    affinity_dict = {}
    with open(index_file, "r") as f:
        lines = f.readlines()
    # 从第七行开始读取数据（跳过前6行）
    for line in lines[6:]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        comp_id = parts[0]
        try:
            affinity = float(parts[3])
        except ValueError:
            affinity = None
        affinity_dict[comp_id] = affinity
    return affinity_dict


def build_sample_graphs(dataset, affinity_labels, d_atom, d_res, d_sub):
    """
    针对一个数据集（core-set 或 refined-set），构造每个复合物对应的样本数据字典。

    每个样本包含以下键：
      - 'ligand_atom_graph'
      - 'protein_atom_graph'
      - 'atom_interaction_graph'
      - 'ligand_fragment_graph'
      - 'protein_residue_graph'
      - 'substructure_interaction_graph'
      - 'label' (亲和力标签)

    参数：
      dataset: 复合物字典，键为复合物名称，值为包含 'pocket'、'ligand'、'protein' 三个键的内容
      affinity_labels: 亲和力标签字典，键为复合物名称
      d_atom: Atom Interaction Graph 的距离阈值
      d_res: Protein Residue Graph 的残基距离阈值
      d_sub: Substructure Interaction Graph 的子结构交互阈值

    返回：
      样本数据字典，键为复合物名称，值为上述样本字典。

    在处理过程中，会打印出每个复合物的处理状态（成功或失败）。
    """
    samples = {}
    total = len(dataset)
    count = 0
    for comp_id, data in dataset.items():
        count += 1
        print(f"[{count}/{total}] 正在处理复合物 {comp_id} ...", end=" ")
        try:
            # 使用 pocket 文件作为蛋白输入构建图
            ligand_atom_g = build_ligand_atom_graph(data['ligand'])
            protein_atom_g = build_protein_atom_graph(data['pocket'])
            atom_interaction_g = build_atom_interaction_graph(data['ligand'], data['pocket'], d_atom)
            ligand_fragment_g = build_ligand_fragment_graph(data['ligand'])
            protein_residue_g = build_protein_residue_graph(data['pocket'], d_res)
            substructure_interaction_g = build_substructure_interaction_graph(data['ligand'], data['pocket'], d_sub)
        except Exception as e:
            print(f"失败，错误信息: {e}")
            continue

        # 获取亲和力标签，如果不存在则设置为 None
        label = affinity_labels.get(comp_id, None)

        samples[comp_id] = {
            'ligand_atom_graph': ligand_atom_g,
            'protein_atom_graph': protein_atom_g,
            'atom_interaction_graph': atom_interaction_g,
            'ligand_fragment_graph': ligand_fragment_g,
            'protein_residue_graph': protein_residue_g,
            'substructure_interaction_graph': substructure_interaction_g,
            'label': label
        }
        print("成功。")
    return samples


def main():
    # 检查核心数据文件是否存在
    if not os.path.exists("core_set.pkl") or not os.path.exists("refined_set.pkl"):
        print("Error: core_set.pkl 和/或 refined_set.pkl 不存在，请先运行 loader.py。")
        return

    # 加载 core-set 和 refined-set 数据
    with open("core_set.pkl", "rb") as f:
        core_set = pickle.load(f)
    with open("refined_set.pkl", "rb") as f:
        refined_set = pickle.load(f)

    # 严格依赖 default.yaml 中的配置，不提供后备默认值
    with open("default.yaml", "r") as f:
        config = yaml.safe_load(f)
    # 配置文件中必须定义以下超参数，否则抛出 KeyError
    d_atom = config["d_atom"]
    d_res = config["d_res"]
    d_sub = config["d_sub"]

    # 加载亲和力标签，INDEX_data.2016 中从第七行开始，每行第一列为复合物名称，第四列为亲和力值
    affinity_labels = load_affinity_labels("INDEX_data.2016")

    print("开始构建图数据样本...")
    # 分别构造 core-set 与 refined-set 的图数据样本
    core_set_graphs = build_sample_graphs(core_set, affinity_labels, d_atom, d_res, d_sub)
    refined_set_graphs = build_sample_graphs(refined_set, affinity_labels, d_atom, d_res, d_sub)

    # 保存图数据样本到新的 pickle 文件
    with open("core_set_graphs.pkl", "wb") as f:
        pickle.dump(core_set_graphs, f)
    with open("refined_set_graphs.pkl", "wb") as f:
        pickle.dump(refined_set_graphs, f)

    print("成功生成并保存 core_set_graphs.pkl 和 refined_set_graphs.pkl。")


if __name__ == "__main__":
    main()
