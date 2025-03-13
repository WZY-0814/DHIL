import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 channels.py 导入各类 Intra/Inter-type 编码模块
from channels import (
    LigandAtomChannel, ProteinAtomChannel, InterAtomChannel,
    LigandFragmentChannel, ProteinResidueChannel, InterSubstructureChannel
)
# 从 hil.py 导入 Atom-level 与 Substructure-level Interactive Learning 模块
from hil import (
    AtomLevelInteractiveLigand, AtomLevelInteractiveProtein,
    SubstructureLevelInteractiveLigand, SubstructureLevelInteractiveProtein
)


class DHILModel(nn.Module):
    """
    DHIL 模型：
      - 输入：六种图数据（配体原子图、蛋白原子图、原子交互图、配体碎片图、蛋白残基图、子结构交互图）
      - 经过 Atom-level 与 Substructure-level 编码、交互与信息融合，预测 binding affinity.
    """

    def __init__(self, config):
        super(DHILModel, self).__init__()
        # 从配置中严格获取各项超参数
        self.l_intra = config["l_intra"]  # Intra-type 编码迭代次数（原子级、子结构级均采用相同值）
        self.l_inter = config["l_inter"]  # Inter-type 编码迭代次数（原子级、子结构级均采用相同值）
        self.l_atom = config["l_atom"]  # Atom-level interactive learning 迭代次数
        self.l_sub = config["l_sub"]  # Substructure-level interactive learning 迭代次数
        self.d = config["embedding_dim"]  # 嵌入维度
        self.neg_slope = config["inter_negative_slope"]  # LeakyReLU 负斜率
        self.d_atom = config["d_atom"]  # 原子交互距离阈值
        self.d_sub = config["d_sub"]  # 子结构交互距离阈值
        # 假设物理化学描述符维度（子结构初始特征 x）为 config["sub_x_dim"]
        self.sub_x_dim = config["sub_x_dim"]

        # ---------------- Atom-level 模块 ----------------
        self.ligand_atom_intra_encoder = LigandAtomChannel(self.l_intra, self.d, self.d, self.neg_slope)
        self.protein_atom_intra_encoder = ProteinAtomChannel(self.l_intra, self.d, self.d, self.neg_slope)
        self.inter_atom_encoder = InterAtomChannel(self.l_inter, self.d_atom, self.d, self.neg_slope)
        self.ligand_atom_interactive = AtomLevelInteractiveLigand(self.l_atom, self.d)
        self.protein_atom_interactive = AtomLevelInteractiveProtein(self.l_atom, self.d)

        # ---------------- Substructure-level 模块 ----------------
        self.ligand_frag_intra_encoder = LigandFragmentChannel(self.l_intra, self.d + self.sub_x_dim,
                                                               self.d + self.sub_x_dim, self.neg_slope)
        self.protein_res_intra_encoder = ProteinResidueChannel(self.l_intra, self.d + self.sub_x_dim,
                                                               self.d + self.sub_x_dim, self.neg_slope)
        self.inter_sub_encoder = InterSubstructureChannel(self.l_inter, self.d_sub, self.d + self.sub_x_dim,
                                                          self.neg_slope)
        self.ligand_sub_interactive = SubstructureLevelInteractiveLigand(self.l_sub, self.d + self.sub_x_dim,
                                                                         self.neg_slope)
        self.protein_sub_interactive = SubstructureLevelInteractiveProtein(self.l_sub, self.d + self.sub_x_dim,
                                                                           self.neg_slope)

        # ---------------- 融合与亲和力预测模块 ----------------
        # 融合：对子结构交互后获得的 intra 表示进行全局平均池化，然后拼接 ligand 和 protein 部分
        # 假设融合后的维度为 2*(d + sub_x_dim)
        fusion_dim = 2 * (self.d + self.sub_x_dim)
        # GRU 层用于进一步融合序列信息（这里把整个复合物看作一个“序列”）
        self.gru = nn.GRU(input_size=fusion_dim, hidden_size=fusion_dim, batch_first=True)
        # 最后一个全连接层将 GRU 的输出映射到标量预测
        self.pred_fc = nn.Linear(fusion_dim, 1)

    def forward(self, sample):
        """
        sample: 字典，包含以下键：
          - "ligand_atom_graph": 配体原子图，节点包含 'h'、'coord'、'group'
          - "protein_atom_graph": 蛋白原子图，节点包含 'h'、'coord'、'group'
          - "atom_interaction_graph": 原子交互图
          - "ligand_fragment_graph": 配体碎片图，节点包含 'x'（物理化学描述符）、'atom_indices'、'coord'
          - "protein_residue_graph": 蛋白残基图，节点包含 'x'、'atom_indices'、'coord'
          - "substructure_interaction_graph": 子结构交互图（节点顺序：先 ligand 碎片，再 protein 残基）
        """
        # -------- Atom-level 编码与交互 --------
        # Intra-type 编码：对配体和蛋白原子图分别编码
        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        ligand_intra = self.ligand_atom_intra_encoder(ligand_atom_graph, ligand_atom_graph.ndata["h"])  # [N_lig, d]
        protein_intra = self.protein_atom_intra_encoder(protein_atom_graph,
                                                        protein_atom_graph.ndata["h"])  # [N_prot, d]

        # Inter-type 编码：利用原子坐标更新 atom_interaction_graph
        ligand_coords = ligand_atom_graph.ndata["coord"]  # [N_lig, 3]
        protein_coords = protein_atom_graph.ndata["coord"]  # [N_prot, 3]
        inter_lig, inter_prot = self.inter_atom_encoder(ligand_intra, protein_intra, ligand_coords, protein_coords)

        # Atom-level Interactive Learning：分别更新 ligand 和 protein 原子
        ligand_group = ligand_atom_graph.ndata["group"]  # [N_lig]
        protein_group = protein_atom_graph.ndata["group"]  # [N_prot]
        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(ligand_intra, inter_lig, ligand_group)
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(protein_intra, inter_prot, protein_group)
        # 保存 Atom-level 的最终 intra 表示，用于后续子结构初始表示更新
        # 假设更新后的 intra 表示为 H^{atom**}
        H_lig_atom_final = updated_intra_lig  # [N_lig, d]
        H_prot_atom_final = updated_intra_prot  # [N_prot, d]

        # -------- Substructure-level 编码与交互 --------
        # 先更新子结构初始表示：对于每个配体碎片节点，
        # H_{L,i}^{fra^(0)} = Concat( x_{L,i}, sum_{k in atom_indices} H_{L,k}^{atom**} )
        ligand_frag_graph = sample["ligand_fragment_graph"]
        protein_res_graph = sample["protein_residue_graph"]
        # 这里假设 ligand_frag_graph.ndata["x"] 为物理化学描述符，
        # ligand_frag_graph.ndata["atom_indices"] 为列表或 tensor 指示所属原子索引
        new_ligand_feats = []
        for i in range(ligand_frag_graph.num_nodes()):
            x_i = ligand_frag_graph.ndata["x"][i]  # [sub_x_dim]
            indices = ligand_frag_graph.ndata["atom_indices"][i]  # list or tensor
            if len(indices) == 0:
                atom_sum = torch.zeros(x_i.shape, device=x_i.device)
            else:
                atom_sum = H_lig_atom_final[indices].sum(dim=0)  # [d]
            new_feat = torch.cat([x_i, atom_sum], dim=0)  # [d + sub_x_dim]
            new_ligand_feats.append(new_feat)
        new_ligand_feats = torch.stack(new_ligand_feats, dim=0)
        ligand_frag_graph.ndata["h"] = new_ligand_feats

        new_protein_feats = []
        for j in range(protein_res_graph.num_nodes()):
            x_j = protein_res_graph.ndata["x"][j]
            indices = protein_res_graph.ndata["atom_indices"][j]
            if len(indices) == 0:
                atom_sum = torch.zeros(x_j.shape, device=x_j.device)
            else:
                atom_sum = H_prot_atom_final[indices].sum(dim=0)
            new_feat = torch.cat([x_j, atom_sum], dim=0)
            new_protein_feats.append(new_feat)
        new_protein_feats = torch.stack(new_protein_feats, dim=0)
        protein_res_graph.ndata["h"] = new_protein_feats

        # Intra-type 编码：对子结构图分别编码
        ligand_intra_sub = self.ligand_frag_intra_encoder(ligand_frag_graph, ligand_frag_graph.ndata["h"])
        protein_intra_sub = self.protein_res_intra_encoder(protein_res_graph, protein_res_graph.ndata["h"])

        # Inter-type 编码：利用子结构节点坐标更新子结构交互图
        inter_sub_graph = sample["substructure_interaction_graph"]
        # 假设 inter_sub_graph 节点顺序为：前部分为 ligand fragments，后部分为 protein residues
        inter_sub_input = torch.cat([ligand_intra_sub, protein_intra_sub], dim=0)
        inter_sub_graph.ndata["h"] = inter_sub_input
        ligand_sub_coords = ligand_frag_graph.ndata["coord"]
        protein_sub_coords = protein_res_graph.ndata["coord"]
        inter_lig_sub, inter_prot_sub = self.inter_sub_encoder(
            ligand_intra_sub, protein_intra_sub, ligand_sub_coords, protein_sub_coords
        )

        # Substructure-level Interactive Learning：分别更新 ligand 和 protein 子结构表示
        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(ligand_intra_sub,
                                                                                         inter_lig_sub)
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(protein_intra_sub,
                                                                                            inter_prot_sub)

        # -------- 融合与预测 --------
        # 对更新后的子结构 intra 表示进行全局池化（例如平均池化），对 ligand 和 protein 分别池化后拼接
        ligand_pool_intra = ligand_sub_updated_intra.mean(dim=0)  # [d + sub_x_dim]
        protein_pool_intra = protein_sub_updated_intra.mean(dim=0)  # [d + sub_x_dim]
        H_final = torch.cat([ligand_pool_intra, protein_pool_intra], dim=0)  # [2*(d+sub_x_dim)]

        ligand_pool_inter = ligand_sub_updated_inter.mean(dim=0)
        protein_pool_inter = protein_sub_updated_inter.mean(dim=0)
        Z_final = torch.cat([ligand_pool_inter, protein_pool_inter], dim=0)  # [2*(d+sub_x_dim)]

        # 最终融合：简单地将 H_final 和 Z_final 拼接作为最终复合物表示
        F_final = torch.cat([H_final, Z_final], dim=0)  # [4*(d+sub_x_dim)]
        # 将最终表示视为单个序列的输入，需增加 batch 和 sequence 维度
        # 这里假设 batch_size=1, sequence length=1
        F_final = F_final.unsqueeze(0).unsqueeze(0)  # [1, 1, 4*(d+sub_x_dim)]

        # 利用 GRU 进行进一步融合
        gru_out, _ = self.gru(F_final)  # 输出 shape [1, 1, fusion_dim]，fusion_dim = 4*(d+sub_x_dim)
        fusion_rep = gru_out.squeeze(0).squeeze(0)  # [fusion_dim]

        # 最后预测亲和力
        y_pred = self.pred_fc(fusion_rep)  # [1]

        return y_pred