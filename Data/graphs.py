import io
import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from Bio.PDB import PDBParser
from featurize import featurize_atom  # 从 featurize.py 导入原子特征提取函数
from featurize import featurize_substructure


def build_ligand_atom_graph(ligand_sdf_content):
    """
    根据传入的配体 SDF 字符串构建配体原子图（Ligand Atom Graph）。
    节点：配体的每个原子，其特征由 featurize_atom 函数提取；
    边：根据化学键连接（无向图）。

    参数:
      ligand_sdf_content: 字符串，包含配体的 SDF 文件内容。

    返回:
      一个 DGL 图对象，节点特征存储在字段 'h' 中。
    """
    # 利用 RDKit 从 SDF 字符串构建分子对象（不移除氢）
    mol = Chem.MolFromMolBlock(ligand_sdf_content, removeHs=False)
    if mol is None:
        raise ValueError("无法解析配体 SDF 内容！")

    # 计算 Gasteiger 部分电荷
    AllChem.ComputeGasteigerCharges(mol)

    # 提取所有原子的特征
    atom_features = []
    for atom in mol.GetAtoms():
        feat = featurize_atom(atom)  # 使用 featurize.py 中定义的函数
        atom_features.append(feat)
    atom_features = np.array(atom_features, dtype=np.float32)

    # 构造边列表，遍历所有化学键，添加双向边
    src_list = []
    dst_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        src_list.extend([i, j])
        dst_list.extend([j, i])

    # 使用 DGL 创建图
    num_atoms = mol.GetNumAtoms()
    g = dgl.graph((src_list, dst_list), num_nodes=num_atoms)

    # 将原子特征赋值到图节点数据 'h' 中（转换为 torch.tensor）
    g.ndata['h'] = torch.tensor(atom_features)

    return g


def build_protein_atom_graph(protein_pdb_content):
    """
    根据传入的蛋白质 PDB 字符串（例如 pocket.pdb），构建蛋白原子图（Protein Atom Graph）。

    节点：蛋白的每个原子，其特征由 featurize_atom 提取；
    边：依据化学键连接构建无向图（双向边）。

    参数:
      protein_pdb_content: 字符串，包含蛋白 pocket PDB 文件内容。

    返回:
      一个 DGL 图对象，节点特征存储在 'h' 字段中。
    """
    # 尝试利用 RDKit 从 PDB 字符串构建分子对象（不移除氢）
    mol = Chem.MolFromPDBBlock(protein_pdb_content, removeHs=False)
    if mol is None:
        raise ValueError("无法解析蛋白 PDB 内容！")

    # 计算 Gasteiger 部分电荷
    AllChem.ComputeGasteigerCharges(mol)

    # 提取所有原子的特征
    atom_features = []
    for atom in mol.GetAtoms():
        feat = featurize_atom(atom)
        atom_features.append(feat)
    atom_features = np.array(atom_features, dtype=np.float32)

    # 构造边列表，遍历所有化学键（假定 RDKit 能够解析出蛋白质中的化学键信息）
    src_list = []
    dst_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        src_list.extend([i, j])
        dst_list.extend([j, i])

    num_atoms = mol.GetNumAtoms()
    g = dgl.graph((src_list, dst_list), num_nodes=num_atoms)
    g.ndata['h'] = torch.tensor(atom_features)

    return g


def get_atom_positions_from_sdf(ligand_sdf_content):
    """
    从配体 SDF 内容中解析 RDKit 分子对象，并提取所有原子的 3D 坐标。
    返回：
      mol: RDKit 分子对象
      positions: numpy 数组，形状为 (num_atoms, 3)
    """
    mol = Chem.MolFromMolBlock(ligand_sdf_content, removeHs=False)
    if mol is None:
        raise ValueError("无法解析配体 SDF 内容！")
    # 计算构象（假定 SDF 中已包含 3D 信息）
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    positions = np.array([np.array(conf.GetAtomPosition(i)) for i in range(num_atoms)], dtype=np.float32)
    return mol, positions


def get_atom_positions_from_pdb(protein_pdb_content):
    """
    从蛋白 PDB 内容中解析 RDKit 分子对象，并提取所有原子的 3D 坐标。
    返回：
      mol: RDKit 分子对象
      positions: numpy 数组，形状为 (num_atoms, 3)
    """
    mol = Chem.MolFromPDBBlock(protein_pdb_content, removeHs=False)
    if mol is None:
        raise ValueError("无法解析蛋白 PDB 内容！")
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    positions = np.array([np.array(conf.GetAtomPosition(i)) for i in range(num_atoms)], dtype=np.float32)
    return mol, positions


def build_atom_interaction_graph(ligand_sdf_content, protein_pdb_content, d_atom):
    """
    构建原子交互图（Atom Interaction Graph）。

    参数:
      ligand_sdf_content: 字符串，配体 SDF 文件内容。
      protein_pdb_content: 字符串，蛋白 pocket PDB 文件内容。
      d_atom: 距离阈值，只有当配体原子与蛋白原子之间的欧氏距离小于该值时，
              才在它们之间添加边。

    返回:
      一个 DGL 图对象，节点为配体与蛋白原子组合而成（顺序：配体原子在前，蛋白原子在后），
      图中不包含节点特征，仅构建边结构。
    """
    # 解析配体，获取分子对象和原子位置
    ligand_mol, ligand_positions = get_atom_positions_from_sdf(ligand_sdf_content)
    # 解析蛋白，获取分子对象和原子位置
    protein_mol, protein_positions = get_atom_positions_from_pdb(protein_pdb_content)

    n_ligand = ligand_mol.GetNumAtoms()
    n_protein = protein_mol.GetNumAtoms()

    # 计算配体与蛋白之间的距离矩阵
    # ligand_positions shape: (n_ligand, 3), protein_positions shape: (n_protein, 3)
    # 利用广播计算距离矩阵，结果 shape: (n_ligand, n_protein)
    diff = ligand_positions[:, np.newaxis, :] - protein_positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    # 找出所有满足距离条件的原子对
    ligand_indices, protein_indices = np.where(dists <= d_atom)

    # 在图中，节点顺序为：前 n_ligand 个节点为配体原子，后 n_protein 个节点为蛋白原子
    # 构造边列表，添加双向边
    src_list = []
    dst_list = []
    for i, j in zip(ligand_indices, protein_indices):
        # ligand 原子 i 对应图中的节点 i
        # protein 原子 j 对应图中的节点 n_ligand + j
        src_list.append(i)
        dst_list.append(n_ligand + j)
        # 添加反向边
        src_list.append(n_ligand + j)
        dst_list.append(i)

    total_nodes = n_ligand + n_protein
    g = dgl.graph((src_list, dst_list), num_nodes=total_nodes)

    # 此图不需要添加节点特征信息
    return g


def build_ligand_fragment_graph(ligand_sdf_content):
    """
    根据传入的配体 SDF 文件内容构建配体碎片图（Ligand Fragment Graph）。

    步骤：
      1. 从 SDF 内容构建 RDKit 分子对象。
      2. 利用 BRICS 分解得到 fragment SMILES（集合形式）。
      3. 将每个 fragment SMILES 转换为 RDKit 分子对象，并用 featurize_substructure（node_type="ligand"）提取节点特征。
      4. 构造边列表：此处采用简单策略，假定所有不同的 fragment 之间均存在反应连接，构成完全图（添加双向边）。
      5. 使用 DGL 构建图对象，将节点特征保存于 'h' 字段中。

    返回：
      一个 DGL 图对象，其节点表示配体的碎片，节点特征为提取的子结构特征。
    """
    # 解析 SDF 内容获得分子对象
    mol = Chem.MolFromMolBlock(ligand_sdf_content, removeHs=False)
    if mol is None:
        raise ValueError("无法解析配体 SDF 内容！")

    # 使用 BRICS 分解，返回 fragment SMILES 的集合
    frag_smiles_set = BRICS.BRICSDecompose(mol)
    # 将集合转换为列表以便索引
    frag_smiles_list = list(frag_smiles_set)

    # 将每个 fragment SMILES 转换为 RDKit 分子对象，并计算节点特征
    fragment_mols = []
    node_features = []
    for smi in frag_smiles_list:
        frag_mol = Chem.MolFromSmiles(smi)
        if frag_mol is None:
            continue
        fragment_mols.append(frag_mol)
        feat = featurize_substructure(frag_mol, node_type="ligand")
        node_features.append(feat)

    if not fragment_mols:
        raise ValueError("BRICS 分解未产生有效的 fragment!")

    node_features = np.array(node_features, dtype=np.float32)
    num_fragments = len(fragment_mols)

    # 构造边列表：简单策略，假定所有不同 fragment 两两之间均存在反应连接（构成完全图）
    src_list = []
    dst_list = []
    for i in range(num_fragments):
        for j in range(num_fragments):
            if i != j:
                src_list.append(i)
                dst_list.append(j)

    g = dgl.graph((src_list, dst_list), num_nodes=num_fragments)
    g.ndata['h'] = torch.tensor(node_features)

    return g


def build_protein_residue_graph(protein_pdb_content, d_res):
    """
    根据传入的蛋白 PDB 字符串（例如 pocket.pdb 内容）构建蛋白残基图（Protein Residue Graph）。

    节点：每个残基（使用 Biopython 提取），节点特征由 featurize_substructure 提取（node_type="protein"）；
    边：如果两个残基的 Cα 原子之间的欧氏距离小于 d_res，则添加无向边（双向边）。

    参数:
      protein_pdb_content: 字符串，包含蛋白 PDB 文件内容。
      d_res: 残基距离阈值（float）。

    返回:
      一个 DGL 图对象，其节点特征存储在 'h' 字段中。
    """
    # 使用 Biopython 解析 PDB 内容（通过 StringIO 处理字符串）
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(protein_pdb_content))

    # 取第一个模型
    model = structure[0]

    residues = []
    ca_coords = []
    # 遍历所有链中的残基
    for chain in model:
        for residue in chain:
            # 过滤掉水分子及其他非标准残基（残基 id 第一个元素非" "时表示异构体等）
            if residue.id[0] != " ":
                continue
            # 如果存在 Cα 原子，则记录该残基及其 Cα 坐标
            if 'CA' in residue:
                ca_atom = residue['CA']
                coord = ca_atom.get_coord()
                residues.append(residue)
                ca_coords.append(coord)
    if not residues:
        raise ValueError("未找到任何残基或残基缺少 Cα 原子！")

    ca_coords = np.array(ca_coords, dtype=np.float32)
    num_residues = len(residues)

    # 构建节点特征：对每个残基调用 featurize_substructure（node_type="protein"）
    node_features = []
    for residue in residues:
        feat = featurize_substructure(residue, node_type="protein")
        node_features.append(feat)
    node_features = np.array(node_features, dtype=np.float32)

    # 构建边列表：对所有残基两两，若它们 Cα 坐标之间的距离 <= d_res，则添加双向边
    src_list = []
    dst_list = []
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist <= d_res:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)

    g = dgl.graph((src_list, dst_list), num_nodes=num_residues)
    g.ndata['h'] = torch.tensor(node_features)

    return g


def get_ligand_fragments_atom_coords(ligand_sdf_content):
    """
    从配体 SDF 内容中解析分子对象，
    利用 BRICS 分解（通过 FragmentOnBRICSBonds 和 GetMolFrags）获得配体碎片，
    对每个碎片：
      - 确保生成3D构象
      - 提取所有原子的 3D 坐标（从构象中）
      - 计算 canonical SMILES 用于排序
    返回:
      一个列表，每个元素为一个 numpy 数组，形状为 (num_atoms_in_fragment, 3)，
      列表顺序为 canonical SMILES 排序后的顺序，确保与 Ligand Fragment Graph 中顺序一致。
    """
    mol = Chem.MolFromMolBlock(ligand_sdf_content, removeHs=False)
    if mol is None:
        raise ValueError("无法解析配体 SDF 内容！")
    # 确保分子拥有3D构象
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    # 利用 BRICS 分解
    frag_mol = Chem.FragmentOnBRICSBonds(mol)
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)

    frag_info = []
    for frag in frags:
        # 确保每个碎片拥有构象
        if frag.GetNumConformers() == 0:
            AllChem.EmbedMolecule(frag)
            AllChem.UFFOptimizeMolecule(frag)
        conf = frag.GetConformer()
        num_atoms = frag.GetNumAtoms()
        coords = np.array([np.array(conf.GetAtomPosition(i)) for i in range(num_atoms)], dtype=np.float32)
        # 计算 canonical SMILES，用于排序确保一致性
        smi = Chem.MolToSmiles(frag, canonical=True)
        frag_info.append((smi, coords))

    # 按照 canonical SMILES 排序
    frag_info.sort(key=lambda x: x[0])
    frag_coords_list = [info[1] for info in frag_info]
    return frag_coords_list


def get_protein_residue_atom_coords(protein_pdb_content):
    """
    利用 Biopython 从蛋白 PDB 内容中解析结构，
    遍历所有标准残基（过滤掉水分子和非标准残基），
    对每个残基提取所有原子的坐标，
    返回一个列表，每个元素为对应残基所有原子坐标的 numpy 数组，
    顺序与 Protein Residue Graph 构建时保持一致。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(protein_pdb_content))
    model = structure[0]

    residue_coords_list = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            atom_coords = []
            for atom in residue.get_atoms():
                atom_coords.append(atom.get_coord())
            if atom_coords:
                coords = np.array(atom_coords, dtype=np.float32)
                residue_coords_list.append(coords)
    return residue_coords_list


def build_substructure_interaction_graph(ligand_sdf_content, protein_pdb_content, d_sub):
    """
    构建子结构交互图（Substructure Interaction Graph），用于捕获配体碎片与蛋白残基之间的非共价相互作用。

    节点：由配体碎片和蛋白残基组成（节点顺序：前 n_lig 个节点为配体碎片，后 n_prot 个节点为蛋白残基），
           此顺序与 Ligand Fragment Graph 及 Protein Residue Graph 保持一致。
    边：对于每对 (ligand fragment, protein residue)，计算两者之间所有原子对的欧氏距离，并取最小值；
         若该最小距离 <= d_sub，则添加边（添加双向边形成无向图）。

    参数:
      ligand_sdf_content: 字符串，配体 SDF 文件内容。
      protein_pdb_content: 字符串，蛋白 PDB 文件内容。
      d_sub: 子结构交互距离阈值（float）。

    返回:
      一个 DGL 图对象，不包含节点特征信息。
    """
    # 获取配体碎片原子坐标列表，顺序为 canonical SMILES 排序后的顺序
    ligand_frag_coords = get_ligand_fragments_atom_coords(ligand_sdf_content)
    # 获取蛋白残基原子坐标列表，顺序与 Protein Residue Graph 保持一致
    protein_res_coords = get_protein_residue_atom_coords(protein_pdb_content)

    n_lig = len(ligand_frag_coords)
    n_prot = len(protein_res_coords)

    src_list = []
    dst_list = []

    # 对于每对 (ligand fragment, protein residue) 计算所有原子对距离的最小值
    for i, frag_coords in enumerate(ligand_frag_coords):
        for j, res_coords in enumerate(protein_res_coords):
            # frag_coords: (n_frag, 3), res_coords: (n_res, 3)
            diff = frag_coords[:, np.newaxis, :] - res_coords[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            min_dist = np.min(dists)
            if min_dist <= d_sub:
                # ligand fragment i 对应节点 i, protein residue j 对应节点 n_lig + j
                src_list.append(i)
                dst_list.append(n_lig + j)
                src_list.append(n_lig + j)
                dst_list.append(i)

    total_nodes = n_lig + n_prot
    g = dgl.graph((src_list, dst_list), num_nodes=total_nodes)
    return g


