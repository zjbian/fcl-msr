import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  
import os
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  

def compute_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    min_dim = min(X.shape[1], Y.shape[1])
    if X.shape[1] != min_dim:
        X = X[:, :min_dim]
    if Y.shape[1] != min_dim:
        Y = Y[:, :min_dim]
    cov_xy = (X.T @ Y) / (X.shape[0] - 1)
    cka_score = (cov_xy ** 2).sum() / np.sqrt(
        ((X.T @ X) ** 2).sum() * ((Y.T @ Y) ** 2).sum()
    )
    return round(cka_score, 3)

def unify_embedding_dim(emb_dict, target_dim=64):
    unified_emb = {}
    for modal, emb in emb_dict.items():
        if modal == 'method':
            unified_emb[modal] = emb
            continue
        if emb is None:
            continue
        current_dim = emb.shape[1]
        if current_dim > target_dim:
            pca = PCA(n_components=target_dim, random_state=42)
            emb_unified = pca.fit_transform(emb)
            unified_emb[modal] = emb_unified
            print(f"模态 {modal}：{current_dim}维 → PCA降维到 {target_dim}维（原始数据量：{emb.shape[0]}个样本）")
        else:
            unified_emb[modal] = emb
            print(f"模态 {modal}：{current_dim}维 → 保持不变（原始数据量：{emb.shape[0]}个样本）")
    return unified_emb

def plot_combined_visualization_umap(emb_dict, data_name, save_dir, target_dim=64):
    unified_emb = unify_embedding_dim(emb_dict, target_dim)
    method = unified_emb['method']
    modal_names = ['ID', 'Attr', 'Img', 'Text']
    fused_name = 'Fused'
    valid_modals = [m for m in modal_names if m in unified_emb]
    modal_colors = {
        "ID": "#1f77b4", "Attr": "#2ca02c", "Img": "#ff7f0e", 
        "Text": "#d62728", "Fused": "#9467bd"
    }

    MAX_SAMPLES_MODAL = 5000    # 每个单模态最大采样数
    MAX_SAMPLES_FUSED = 5000   # 融合特征最大采样数
    random_seed = 42            # 固定随机种子

    valid_data_exists = len(valid_modals) > 0 or (fused_name in unified_emb and unified_emb[fused_name].shape[0] >= 100)
    if not valid_data_exists:
        print(f"方法 {method}（UMAP版）无有效模态和融合特征，跳过绘图")
        return
    fig, (ax_modal, ax_fused) = plt.subplots(1, 2, figsize=(20, 9))
    
    # -------------------------- 子图1：单模态 UMAP --------------------------
    if len(valid_modals) > 0:
        all_embs = []
        all_labels = []
        total_sampled = 0
        for modal in valid_modals:
            embs = unified_emb[modal]
            n_total = embs.shape[0]
            n_sample = min(MAX_SAMPLES_MODAL, n_total)
            sample_idx = np.random.RandomState(random_seed).choice(n_total, n_sample, replace=False)
            sampled_embs = embs[sample_idx]
            all_embs.append(sampled_embs)
            all_labels.extend([modal] * n_sample)
            total_sampled += n_sample
            print(f"  模态 {modal}（UMAP版）：采样 {n_sample}/{n_total} 个样本")

        all_embs_stacked = np.vstack(all_embs)
        print(f"单模态采样后总数据量（UMAP版）：{total_sampled}个样本")
        umap_modal = UMAP(
            n_components=2,
            random_state=random_seed,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            n_jobs=-1
        )
        all_embs_2d = umap_modal.fit_transform(all_embs_stacked)

        start_idx = 0
        for modal in valid_modals:
            modal_sample_num = min(MAX_SAMPLES_MODAL, unified_emb[modal].shape[0])
            end_idx = start_idx + modal_sample_num
            ax_modal.scatter(
                all_embs_2d[start_idx:end_idx, 0],
                all_embs_2d[start_idx:end_idx, 1],
                c=modal_colors[modal],
                label=modal,
                s=25,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.3
            )
            start_idx = end_idx

        ax_modal.set_xlabel("UMAP Dimension 1", fontsize=20, fontweight='bold')
        ax_modal.set_ylabel("UMAP Dimension 2", fontsize=20, fontweight='bold')
        ax_modal.tick_params(axis='both', which='major', labelsize=16)
        ax_modal.legend(fontsize=18, loc="upper right", frameon=True, framealpha=0.8)
        ax_modal.grid(alpha=0.3, linestyle='--')
        ax_modal.set_aspect('equal')
    else:
        ax_modal.text(0.5, 0.5, "No Valid Single Modal", ha='center', va='center',
                     transform=ax_modal.transAxes, fontsize=18)
        ax_modal.set_xlabel("UMAP Dimension 1", fontsize=20, fontweight='bold')
        ax_modal.set_ylabel("UMAP Dimension 2", fontsize=20, fontweight='bold')
        ax_modal.tick_params(axis='both', which='major', labelsize=16)

    # -------------------------- 子图2：融合后 UMAP --------------------------
    if fused_name in unified_emb and unified_emb[fused_name].shape[0] >= 100:
        fused_embs = unified_emb[fused_name]
        n_total_fused = fused_embs.shape[0]
        n_sample_fused = min(MAX_SAMPLES_FUSED, n_total_fused)
        sample_idx_fused = np.random.RandomState(random_seed).choice(n_total_fused, n_sample_fused, replace=False)
        sampled_fused_embs = fused_embs[sample_idx_fused]
        print(f"融合特征（UMAP版）：采样 {n_sample_fused}/{n_total_fused} 个样本")

        umap_fused = UMAP(
            n_components=2,
            random_state=random_seed,
            n_neighbors=50,
            min_dist=0.2,
            metric='cosine',
            n_jobs=-1
        )
        fused_embs_2d = umap_fused.fit_transform(sampled_fused_embs)
        ax_fused.scatter(
            fused_embs_2d[:, 0],
            fused_embs_2d[:, 1],
            c=modal_colors[fused_name],
            label="Fused Representation",
            s=20,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.2
        )

        ax_fused.set_xlabel("UMAP Dimension 1", fontsize=20, fontweight='bold')
        ax_fused.set_ylabel("UMAP Dimension 2", fontsize=20, fontweight='bold')
        ax_fused.tick_params(axis='both', which='major', labelsize=16)
        ax_fused.legend(fontsize=18, loc="upper right", frameon=True, framealpha=0.8)
        ax_fused.grid(alpha=0.3, linestyle='--')
        ax_fused.set_aspect('equal')
    else:
        ax_fused.text(0.5, 0.5, "No Valid Fused Feature", ha='center', va='center',
                     transform=ax_fused.transAxes, fontsize=18)
        ax_fused.set_xlabel("UMAP Dimension 1", fontsize=20, fontweight='bold')
        ax_fused.set_ylabel("UMAP Dimension 2", fontsize=20, fontweight='bold')
        ax_fused.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(pad=3.0)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{method}_umap_sampled_{data_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"{method}（UMAP版）可视化图已保存到：{save_path}")

def plot_combined_visualization_tsne(emb_dict, data_name, save_dir, target_dim=64):
    unified_emb = unify_embedding_dim(emb_dict, target_dim)
    method = unified_emb['method']
    modal_names = ['ID', 'Attr', 'Img', 'Text']
    fused_name = 'Fused'
    valid_modals = [m for m in modal_names if m in unified_emb]
    modal_colors = {
        "ID": "#1f77b4", "Attr": "#2ca02c", "Img": "#ff7f0e", 
        "Text": "#d62728", "Fused": "#9467bd"
    }

    MAX_SAMPLES_MODAL = 3000    # t-SNE 单模态最大采样数（比 UMAP 少，加速计算）
    MAX_SAMPLES_FUSED = 3000   # t-SNE 融合特征最大采样数
    random_seed = 40           # 固定随机种子
    valid_data_exists = len(valid_modals) > 0 or (fused_name in unified_emb and unified_emb[fused_name].shape[0] >= 100)
    if not valid_data_exists:
        print(f"方法 {method}（t-SNE版）无有效模态和融合特征，跳过绘图")
        return
    fig, (ax_modal, ax_fused) = plt.subplots(1, 2, figsize=(20, 9))
    
    # -------------------------- 子图1：单模态 t-SNE --------------------------
    if len(valid_modals) > 0:
        all_embs = []
        all_labels = []
        total_sampled = 0
        for modal in valid_modals:
            embs = unified_emb[modal]
            n_total = embs.shape[0]
            n_sample = min(MAX_SAMPLES_MODAL, n_total)
            sample_idx = np.random.RandomState(random_seed).choice(n_total, n_sample, replace=False)
            sampled_embs = embs[sample_idx]
            all_embs.append(sampled_embs)
            all_labels.extend([modal] * n_sample)
            total_sampled += n_sample
            print(f"  模态 {modal}（t-SNE版）：采样 {n_sample}/{n_total} 个样本")

        all_embs_stacked = np.vstack(all_embs)
        print(f"单模态采样后总数据量（t-SNE版）：{total_sampled}个样本")
        tsne_modal = TSNE(
            n_components=2,
            random_state=random_seed,
            perplexity=30,  # 推荐值：5-50，样本少用小值，样本多用大值
            early_exaggeration=12,
            learning_rate=200,
            n_jobs=-1       
        )
        all_embs_2d = tsne_modal.fit_transform(all_embs_stacked)

        start_idx = 0
        for modal in valid_modals:
            modal_sample_num = min(MAX_SAMPLES_MODAL, unified_emb[modal].shape[0])
            end_idx = start_idx + modal_sample_num
            ax_modal.scatter(
                all_embs_2d[start_idx:end_idx, 0],
                all_embs_2d[start_idx:end_idx, 1],
                c=modal_colors[modal],
                label=modal,
                s=25,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.3
            )
            start_idx = end_idx

        ax_modal.set_xlabel("t-SNE Dimension 1", fontsize=20, fontweight='bold')
        ax_modal.set_ylabel("t-SNE Dimension 2", fontsize=20, fontweight='bold')
        ax_modal.tick_params(axis='both', which='major', labelsize=16)
        ax_modal.legend(fontsize=18, loc="upper right", frameon=True, framealpha=0.8)
        ax_modal.grid(alpha=0.3, linestyle='--')
        ax_modal.set_aspect('equal')
    else:
        ax_modal.text(0.5, 0.5, "No Valid Single Modal", ha='center', va='center',
                     transform=ax_modal.transAxes, fontsize=18)
        ax_modal.set_xlabel("t-SNE Dimension 1", fontsize=20, fontweight='bold')
        ax_modal.set_ylabel("t-SNE Dimension 2", fontsize=20, fontweight='bold')
        ax_modal.tick_params(axis='both', which='major', labelsize=16)

    # -------------------------- 子图2：融合后 t-SNE --------------------------
    if fused_name in unified_emb and unified_emb[fused_name].shape[0] >= 100:
        fused_embs = unified_emb[fused_name]
        n_total_fused = fused_embs.shape[0]
        n_sample_fused = min(MAX_SAMPLES_FUSED, n_total_fused)
        sample_idx_fused = np.random.RandomState(random_seed).choice(n_total_fused, n_sample_fused, replace=False)
        sampled_fused_embs = fused_embs[sample_idx_fused]
        print(f"融合特征（t-SNE版）：采样 {n_sample_fused}/{n_total_fused} 个样本")

        tsne_fused = TSNE(
            n_components=2,
            random_state=random_seed,
            perplexity=50,  # 融合特征样本多，perplexity 可适当增大
            early_exaggeration=12,
            learning_rate=200,
            n_jobs=-1
        )
        fused_embs_2d = tsne_fused.fit_transform(sampled_fused_embs)
        ax_fused.scatter(
            fused_embs_2d[:, 0],
            fused_embs_2d[:, 1],
            c=modal_colors[fused_name],
            label="Fused Representation",
            s=20,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.2
        )

        ax_fused.set_xlabel("t-SNE Dimension 1", fontsize=20, fontweight='bold')
        ax_fused.set_ylabel("t-SNE Dimension 2", fontsize=20, fontweight='bold')
        ax_fused.tick_params(axis='both', which='major', labelsize=16)
        ax_fused.legend(fontsize=18, loc="upper right", frameon=True, framealpha=0.8)
        ax_fused.grid(alpha=0.3, linestyle='--')
        ax_fused.set_aspect('equal')
    else:
        ax_fused.text(0.5, 0.5, "No Valid Fused Feature", ha='center', va='center',
                     transform=ax_fused.transAxes, fontsize=18)
        ax_fused.set_xlabel("t-SNE Dimension 1", fontsize=20, fontweight='bold')
        ax_fused.set_ylabel("t-SNE Dimension 2", fontsize=20, fontweight='bold')
        ax_fused.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout(pad=3.0)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{method}_tsne_sampled_{data_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"{method}（t-SNE版）可视化图已保存到：{save_path}")


def plot_all_combined_visualization(all_embeddings_list, data_name, save_dir, target_dim=64):
    os.makedirs(save_dir, exist_ok=True)
    for emb_dict in all_embeddings_list:
        method = emb_dict.get('method', 'unknown')
        
        # 1. 绘制 UMAP 版
        print(f"\n===== 开始绘制 {method} 可视化图（UMAP版）=====")
        plot_combined_visualization_umap(emb_dict, data_name, save_dir, target_dim)
        
        # 2. 绘制 t-SNE 版
        print(f"\n===== 开始绘制 {method} 可视化图（t-SNE版）=====")
        plot_combined_visualization_tsne(emb_dict, data_name, save_dir, target_dim)
    
    print(f"\n所有可视化图（UMAP + t-SNE）已保存到：{save_dir}")