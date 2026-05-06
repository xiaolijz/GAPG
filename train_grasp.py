import argparse
import numpy as np
import random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import open3d as o3d
import os
from tqdm import tqdm
from models.grasp_networks import Space_GraspFusion
import utils  
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split

class GraspPushDataset(Dataset):
    def __init__(self, global_npy_dir, obj_npy_dir, pose_txt, label_txt, augment=True):
        self.global_npy_files = sorted(
            [os.path.join(global_npy_dir, f) for f in os.listdir(global_npy_dir) if f.endswith('.npy')],
            key=utils.natural_key
        )
        self.obj_npy_files = sorted(
            [os.path.join(obj_npy_dir, f) for f in os.listdir(obj_npy_dir) if f.endswith('.npy')],
            key=utils.natural_key
        )
        self.poses = np.loadtxt(pose_txt).reshape(-1, 7)
        self.labels = np.loadtxt(label_txt).astype(np.int64)
        self.augment = augment
        self.gripper_count = 200
        self.sample_count = 345
        self.pose_points_fixed, _ = utils.grasp_pcd_bluenoise_like(n_target=self.gripper_count, oversample=2000, seed=55926)

        assert len(self.global_npy_files) == len(self.obj_npy_files) == len(self.poses) == len(self.labels), "数据长度不一致"

    def __len__(self):
        return len(self.global_npy_files)

    def __getitem__(self, idx):
        global point_num
        global_points = torch.from_numpy(np.load(self.global_npy_files[idx])).float()
        obj_points = torch.from_numpy(np.load(self.obj_npy_files[idx])).float()
        pose = torch.from_numpy(self.poses[idx]).float()
        global_points_ee = utils.TransformPCD2EndLink(global_points, pose)
        obj_points_ee = utils.TransformPCD2EndLink(obj_points, pose)
        pose_points = self.pose_points_fixed
        sence_points = utils.fuse_state_torch_v2(global_points_ee, obj_points_ee, pose_points)

        if self.augment:
            sence_points = self.augment_pointcloud(sence_points)

        sample_sence_points = utils.furthest_point_sampling_nocuda(sence_points,n_samples = (self.sample_count - self.gripper_count) )

        fuse_points = torch.cat([sample_sence_points,pose_points],dim=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fuse_points)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([pcd, frame])

        pose_points_label = torch.cat([pose_points,torch.zeros((pose_points.shape[0], 1), dtype=pose_points.dtype, device=pose_points.device)],dim=1)
        sence_points_label = torch.cat([sample_sence_points,torch.ones((sample_sence_points.shape[0], 1), dtype=sample_sence_points.dtype, device=sample_sence_points.device)],dim=1)
        fuse_points = torch.cat([sence_points_label,pose_points_label],dim = 0)
        uniform_fuse_points = utils.pc_normalize_grasp(fuse_points)
        uniform_fuse_points = uniform_fuse_points.T.to(dtype=torch.float32)  # [3, N]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return uniform_fuse_points, label

    def augment_pointcloud(self, pc, prob=0.3): 
        if torch.rand(1).item() < prob:
            return pc.float()
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc).float()

        pc = pc + torch.clamp(0.003 * torch.randn_like(pc), -0.001, 0.001)

        if torch.rand(1).item() < 0.3:
            n = pc.shape[0]
            drop = int(n * 0.08 * torch.rand(1).item())
            if drop > 0:
                keep_idx = torch.randperm(n)[:n - drop]
                pc = pc[keep_idx]

        if torch.rand(1).item() < 0.2:
            axis = torch.randint(0, 3, (1,)).item()
            thr = pc[:, axis].median()
            width = pc[:, axis].std() * 0.1
            mask = (pc[:, axis] < (thr - width)) | (pc[:, axis] > (thr + width))
            if mask.sum() > 16: pc = pc[mask]

        return pc

# ---------- Arg Parser ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=55926)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, default='env_data_collection/ground_true_dateset/best_model/best_model_v8.pt')
    parser.add_argument('--global_ply_dir', type=str, default='env_data_collection/ground_true_dateset/npy_global')
    parser.add_argument('--obj_ply_dir',type=str,default='env_data_collection/ground_true_dateset/npy_obj')
    parser.add_argument('--pose_txt', type=str, default='env_data_collection/ground_true_dateset/poses.txt')
    parser.add_argument('--label_txt', type=str, default='env_data_collection/ground_true_dateset/labels_update_v3.txt')
    parser.add_argument('--save_model_interval', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--width',type=int,default=128)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--lr_milestones', nargs='+', type=int, default=[30, 60, 90, 120, 150, 180, 210, 240])
    parser.add_argument('--lr_gamma', type=float, default=0.2)
    return parser.parse_args()

# ---------- Train + Validation ----------
def set_random_seed(seed=0, deterministic=False):
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

def train():
    args = parse_args()
    set_random_seed(args.seed)
    g_train = torch.Generator().manual_seed(args.seed + 123)
    g_val   = torch.Generator().manual_seed(args.seed + 456)
    device = torch.device(args.device)

    full_dataset = GraspPushDataset(
        args.global_ply_dir, args.obj_ply_dir, args.pose_txt, args.label_txt, augment=True
    )

    val_dataset = GraspPushDataset(
        args.global_ply_dir, args.obj_ply_dir, args.pose_txt, args.label_txt, augment=False
    )

    total_size = len(full_dataset)
    val_ratio = args.val_ratio

    labels = full_dataset.labels
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels).reshape(-1)

    indices = np.arange(total_size)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=args.seed,
        stratify=labels,
        shuffle=True
    )

    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    val_set   = torch.utils.data.Subset(val_dataset,   val_indices)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=32,
        pin_memory=True, shuffle=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g_train, persistent_workers=True
    )

    val_loader = DataLoader(
        val_set, batch_size=int(args.batch_size/4), num_workers=32,
        pin_memory=True, shuffle=False, drop_last=False,
        worker_init_fn=seed_worker, generator=g_val, persistent_workers=True
    )

    model = Space_GraspFusion(device='cuda').to(device)
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=3e-4) 
    scheduler = MultiStepLR(optimizer,args.lr_milestones,args.lr_gamma) 
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   
    writer = SummaryWriter(log_dir='runs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    hparam_summary = f"""
    # -------- Training Config Summary --------
    Best Model: {args.model_path}
    Epoch Number: {args.epochs}
    Batch Size: {args.batch_size}
    Learning Rate: {args.lr}
    Weight decay: 3e-4
    LR Scheduler: MultiStepLR(milestones={args.lr_milestones}, gamma={args.lr_gamma})
    FPS Sample Count: {full_dataset.sample_count}
    Split Ratio: {1 - args.val_ratio}/{args.val_ratio}
    Augment: train=True, val=False
    Experiment Start Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """
    writer.add_text("hparams/training_config", hparam_summary)

    # best_val_loss = float('inf')
    best_rate = 0.0
    with tqdm(total=args.epochs) as pbar:
        for epoch in range(args.epochs):
            # -------------------- Train --------------------
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (pointclouds, labels) in enumerate(train_loader):
                pointclouds = pointclouds.to(device)
                labels = labels.to(device)                     
                optimizer.zero_grad(set_to_none=True)
                preds = model(pointclouds)                    
                loss = criterion(preds, labels.long()) 
                # loss.backward()
                # optimizer.step()
                loss.backward()                           
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()                

                pred = preds.argmax(dim=1)                   
                correct += (pred == labels).long().cpu().sum().item()
                total   += labels.size(0)

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            scheduler.step()
            avg_train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
            writer.add_scalar("Loss/train_correct_rate_epoch", train_acc, epoch)   

            # -------------------- Validation --------------------
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for pointclouds, labels in val_loader:
                    pointclouds = pointclouds.to(device)
                    labels = labels.to(device)                   

                    preds = model(pointclouds)                  
                    loss = criterion(preds, labels.long())
                    val_loss += loss.item()

                    pred = preds.argmax(dim=1)                
                    correct += (pred == labels).long().cpu().sum().item()
                    total   += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
            writer.add_scalar("Loss/val_correct_rate_epoch", val_acc, epoch)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            pbar.update(1)
            if val_acc > best_rate:
                best_rate = val_acc
                os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_rate,
                }, args.model_path)
                print(f"Saved best model with val acc: {best_rate:.4f} at epoch {epoch}")
                
if __name__ == "__main__":
    train()