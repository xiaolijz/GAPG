import argparse
from termcolor import cprint
import numpy as np
import random
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
from tqdm import tqdm
from models.push_networks import Push_model,Push_Model_Loss
import utils  
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from env.constants import WORKSPACE_LIMITS as workspace_limits
from pytorch3d.ops import sample_farthest_points

class PushDataset(Dataset):
    def __init__(self, global_npy_dir, pose_txt, label_txt, augment=True):
        self.global_npy_files = sorted(
            [os.path.join(global_npy_dir, f) for f in os.listdir(global_npy_dir) if f.endswith('.npy')],
            key=utils.natural_key
        )

        self.push_action = np.loadtxt(pose_txt).reshape(-1, 7).astype(np.float32)
        self.labels = np.loadtxt(label_txt).astype(np.int64)
        self.augment = augment
        self.fixed_point = torch.tensor([
            (workspace_limits[0][0] + workspace_limits[0][1]) / 2,
            (workspace_limits[1][0] + workspace_limits[1][1]) / 2
        ], dtype=torch.float32)

        assert len(self.global_npy_files) ==  len(self.push_action) == len(self.labels), "The length of data is not same!"

    def __len__(self):
        return len(self.global_npy_files)

    def __getitem__(self, idx):
        global_points_2onehot = torch.from_numpy(np.load(self.global_npy_files[idx])).float()
        pose = torch.from_numpy(self.push_action[idx]).float()

        global_points_2onehot_ee = utils.Transform_Push2Fixed_point_onehot(global_points_2onehot, self.fixed_point, pose)
        pose_points = self.fixed_point
        pose_points = torch.cat([pose_points,pose[2].unsqueeze(0)],dim=-1) # 3
        pose_points = pose_points.unsqueeze(0) # 1X3
        pose_points_3onehot = torch.cat([pose_points, torch.tensor([[0, 0, 1]], dtype=pose_points.dtype, device=pose_points.device).repeat(pose_points.size(0), 1)],dim=-1)

        if self.augment:
            global_points_2onehot_ee = self.augment_pointcloud_onehot(global_points_2onehot_ee)

        sence_points_3onehot = torch.cat([global_points_2onehot_ee, torch.zeros((global_points_2onehot_ee.shape[0], 1), dtype=global_points_2onehot_ee.dtype, device=global_points_2onehot_ee.device)],dim = 1)
        fuse_points_3onehot = torch.cat([sence_points_3onehot, pose_points_3onehot],dim = 0) # (N+1)X6
        normalize_fuse_points_3onehot,_,_ = utils.pc_normalize_grasp_onehot(fuse_points_3onehot)
        normalize_sence_points_3onehot,normalize_pose_points_3onehot = normalize_fuse_points_3onehot[:-1],normalize_fuse_points_3onehot[-1]

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(fuse_points_3onehot[:,:3].numpy())
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])

        normalize_sence_points_3onehot = normalize_sence_points_3onehot.T.to(dtype=torch.float32)  # [6, N]
        normalize_pose_points_3onehot = normalize_pose_points_3onehot.squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return normalize_sence_points_3onehot, normalize_pose_points_3onehot, label

    def augment_pointcloud_onehot(self, pc, prob=0.3):
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc)

        pc = pc.float()
        dev = pc.device

        if torch.rand(1).item() < prob:
            return pc
        
        xyz   = pc[:, :3]                 # [N,3]
        extra = pc[:, 3:]                 # [N, C-3]

        noise = torch.clamp(0.003 * torch.randn_like(xyz, device=dev), -0.001, 0.001)
        xyz   = xyz + noise

        if torch.rand(1).item() < 0.3:
            n = pc.shape[0]
            drop = int(n * 0.15 * torch.rand(1).item())
            if drop > 0:
                keep_idx = torch.randperm(n, device=dev)[: n - drop]
                xyz   = xyz.index_select(0, keep_idx)
                extra = extra.index_select(0, keep_idx)

        pc_aug = torch.cat([xyz, extra], dim=1)

        return pc_aug

# ---------- Arg Parser ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=55926)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, default='save/your_model_name.pth')
    parser.add_argument('--global_npy_dir', type=str, default='your_global_npy_dir')
    parser.add_argument('--pose_txt', type=str, default='your_pose_txt')
    parser.add_argument('--label_txt', type=str, default='your_label_txt')
    parser.add_argument('--width',type=int,default=128)
    parser.add_argument('--sence_FPS_count',type=int,default=1024)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_val', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--lr',type=float,default=0.0008)
    parser.add_argument('--lr_decay',type=float,default=0.95)
    parser.add_argument('--patience',type=int,default=1)
    parser.add_argument('--betas',type=float,default=(0.9, 0.999))
    parser.add_argument('--eps',type=float,default=1e-8)
    parser.add_argument('--weight_decay',type=float,default=1e-3)
    return parser.parse_args()

# ---------- Train + Validation ----------
def set_random_seed(seed=0, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if deterministic:
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    import numpy as np, random, torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)  

def collate_batch(batch):
    # batch: list of (scene_6N, seed_6, label)
    scenes, seeds, labels = zip(*batch)     # len = B
    B = len(scenes); C = scenes[0].shape[0] # C = 6
    lens = [s.shape[1] for s in scenes]    
    Nmax = max(lens)

    out = torch.zeros(B, C, Nmax, dtype=torch.float32)
    for i, s in enumerate(scenes):
        out[i, :, :s.shape[1]] = s

    seeds   = torch.stack(seeds, dim=0)             # [B,6]
    # labels  = torch.tensor(labels, dtype=torch.long) # [B]
    labels = torch.stack(labels, dim=0).long() 
    lengths = torch.tensor(lens, dtype=torch.int64)  # [B]
    return out, seeds, labels, lengths

def fps_fill_to_k(xyz, lengths, K_target, *, g=None):

    K_use = int(min(int(lengths.min().item()), int(K_target)))
    if K_use <= 0:
        raise RuntimeError("Found a sample with zero valid points. Please filter it out in Dataset.")

    _, fps_idx = sample_farthest_points(xyz, K=K_use, lengths=lengths)  # [B, K_use]

    if K_use < K_target:
        need = K_target - K_use
        if g is None:
            pad_sel = torch.randint(0, K_use, (xyz.shape[0], need), device=xyz.device)   # [B, need]
        else:
            pad_sel = torch.randint(0, K_use, (xyz.shape[0], need), device=xyz.device, generator=g)
        pad_idx = fps_idx.gather(1, pad_sel)                                            # [B, need]
        fps_idx = torch.cat([fps_idx, pad_idx], dim=1)                                  # [B, K_target]

    return fps_idx

def train():
    args = parse_args()
    set_random_seed(args.seed)
    g_train = torch.Generator().manual_seed(args.seed + 123)
    g_val   = torch.Generator().manual_seed(args.seed + 456)
    device = torch.device(args.device)

    train_dataset = PushDataset(args.global_npy_dir, args.pose_txt, args.label_txt, augment=True)
    val_dataset = PushDataset(args.global_npy_dir, args.pose_txt, args.label_txt, augment=False)

    total_size = len(train_dataset)
    val_ratio = args.val_ratio
    labels = train_dataset.labels
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels).reshape(-1)
    indices = np.arange(total_size)
    ys = labels.astype(np.int64)
    train_indices, val_indices = train_test_split(indices,test_size=val_ratio,random_state=args.seed,stratify=ys,shuffle=True)
    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    val_set = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size_train, num_workers=32,
        pin_memory=True,  shuffle=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g_train, persistent_workers=True,
        collate_fn=collate_batch, 
    )

    val_loader = DataLoader(
        val_set, batch_size=int(args.batch_size_val), num_workers=32,
        pin_memory=True, shuffle=False, drop_last=False,
        worker_init_fn=seed_worker, generator=g_val, persistent_workers=True,
        collate_fn=collate_batch, 
    )


    model = Push_model(additional_channel = 3).cuda()
    criterion = Push_Model_Loss().cuda()

    if args.weight_decay:
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=args.betas,
                eps=args.eps,
                weight_decay=args.weight_decay,
            )
    else:
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=args.betas,
                eps=args.eps )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay, patience=args.patience)
   
    writer = SummaryWriter(log_dir='push_runs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    hparam_summary = f"""
    # -------- Training Config Summary --------
    Best Model: {args.model_path}
    Label Version: {args.label_txt}
    Epoch Number: {args.epochs}
    Batch Size: train:{args.batch_size_train},val:{(args.batch_size_val)}
    optimizer: adam
    eps: {args.eps}
    betas: {args.betas}
    Learning Rate: {args.lr}
    Weight decay: {args.weight_decay}
    LR Scheduler:  ReduceLROnPlateau(factor={args.lr_decay}, patience={args.patience}, mode='min')
    Sample Count: scene:{args.sence_FPS_count}
    Split Ratio: {1 - args.val_ratio}/{args.val_ratio}
    Augment: train=True, val=False
    OutPut: 192
    """
    writer.add_text("hparams/training_config", hparam_summary)

    best_rate = 0.0
    K_target = args.sence_FPS_count
    with tqdm(total=args.epochs) as pbar:
        for epoch in range(args.epochs):
            # -------------------- Train --------------------
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for scene_pc, seed6, labels, lengths in train_loader:
                scene_pc = scene_pc.to(device, non_blocking=True)   # [B,6,Nmax]
                seed6    = seed6.to(device, non_blocking=True)      # [B,6]
                labels   = labels.to(device)
                lengths  = lengths.to(device)

                xyz = scene_pc[:, :3, :].transpose(1, 2).contiguous()  # [B,Nmax,3]

                fps_idx = fps_fill_to_k(xyz, lengths,K_target,g=g_train)  # fps_idx:[B,K]

                idx_exp  = fps_idx.unsqueeze(1).expand(-1, scene_pc.shape[1], -1)  # [B,6,K]
                fps_feat = torch.gather(scene_pc, 2, idx_exp)                      # [B,6,K]

                seed_feat = seed6.unsqueeze(-1)                                    # [B,6,1]
                fused     = torch.cat([fps_feat, seed_feat], dim=2)                # [B,6,K+1]

                select_idx = torch.full((fused.shape[0],), K_target, dtype=torch.long, device=device)

                optimizer.zero_grad(set_to_none=True)
                preds = model(fused)                                 # [B,K+1,1]
                loss  = criterion(preds, select_idx, labels.long())  

                loss.backward()                           
                # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()                

                probs = preds.squeeze(-1)                                    # [B,N]
                seed_prob = probs.gather(1, select_idx.view(-1,1)).squeeze(1) # [B]
                pred_bin = (seed_prob >= 0.5).long()                       
                correct += (pred_bin == labels).sum().item()
                total   += labels.size(0)

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
            writer.add_scalar("Score/train_epoch", train_acc, epoch)   

            # -------------------- Validation --------------------
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0         

            with torch.no_grad():
                for scene_pc, seed6, labels, lengths in val_loader:
                    scene_pc = scene_pc.to(device, non_blocking=True)
                    seed6    = seed6.to(device, non_blocking=True)
                    labels   = labels.to(device).long()
                    lengths  = lengths.to(device)

                    xyz = scene_pc[:, :3, :].transpose(1, 2).contiguous()
                    fps_idx = fps_fill_to_k(xyz, lengths, K_target,g=g_val)
                    idx_exp  = fps_idx.unsqueeze(1).expand(-1, scene_pc.shape[1], -1)
                    fps_feat = torch.gather(scene_pc, 2, idx_exp)
                    seed_feat = seed6.unsqueeze(-1)
                    fused     = torch.cat([fps_feat, seed_feat], dim=2)
                    select_idx = torch.full((fused.shape[0],), K_target, dtype=torch.long, device=device)

                    preds = model(fused)
                    loss  = criterion(preds, select_idx, labels.long())

                    val_loss += loss.item()

                    probs = preds.squeeze(-1)                                    # [B,N]
                    seed_prob = probs.gather(1, select_idx.view(-1,1)).squeeze(1) # [B]
                    pred_bin = (seed_prob >= 0.5).long()                       
                    correct += (pred_bin == labels).sum().item()
                    total   += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total

            scheduler.step(avg_val_loss)

            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
            writer.add_scalar("Score/val_epoch", val_acc, epoch)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            pbar.update(1)

            if val_acc > best_rate:
                best_rate = val_acc
                os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
                cprint(f"Best model with val acc: {best_rate:.4f} at epoch {epoch}",'blue')
            
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_rate,
                }, args.model_path)
                cprint(f"Saving at : {args.model_path}",'green')

if __name__ == "__main__":
    train()