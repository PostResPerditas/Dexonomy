"""Fixed-object-state task surface regions for matching and contact labeling."""

import numpy as np
from scipy.spatial import cKDTree
from transforms3d.quaternions import quat2mat


class TaskRegions:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.names = data["task_names"].tolist()
            self.bodies = dict(zip(self.names, data["body_names"].tolist()))
            self.points = {name: data["points"][data["offsets"][i]:data["offsets"][i+1]].copy()
                           for i, name in enumerate(self.names)}
            self.scene_id = str(data["scene_id"])
            self.frame_body = str(data["frame_body"])
            self.frame_pose = data["frame_pose"].copy()
            self.object_name = str(data["object_name"])
            self.object_pose = data["object_pose"].copy()
            self.object_qpos = data["object_qpos"].copy()
        self.trees = {name: cKDTree(points) for name, points in self.points.items()}
        self.rotation = quat2mat(self.frame_pose[3:])

    def check_scene(self, scene):
        obj = scene["scene"][self.object_name]
        target = obj["part_info"][self.frame_body]
        if (scene["scene_id"] != self.scene_id
                or scene["task"]["part_name"] != self.frame_body
                or not np.allclose(obj["qpos"], self.object_qpos)
                or not np.allclose(obj["pose"], self.object_pose)
                or not np.allclose(target["pose"], self.frame_pose)
                or not np.allclose(target["scale"], 1)
                or not np.allclose(obj["scale"], 1)):
            raise ValueError("Task regions require their exported scene pose, joints and unit scale")

    def contact_labels(self, contact, mask, tolerance):
        points = (np.asarray(contact["pos"]).reshape(-1, 3) - self.frame_pose[:3]) @ self.rotation
        bodies = np.asarray(contact["bn2"])
        return [name for name in self.names
                if np.any(mask & (bodies == self.bodies[name])
                          & (self.trees[name].query(points)[0] <= tolerance))]

    def sampled_points(self, name, count):
        # Deterministic farthest-point subset; full clouds remain available for labels.
        points = self.points[name]
        if count >= len(points):
            return points
        indices = [0]
        distance = np.full(len(points), np.inf)
        for _ in range(count - 1):
            distance = np.minimum(distance, np.sum((points - points[indices[-1]]) ** 2, axis=1))
            indices.append(int(distance.argmax()))
        return points[indices]


class RegionLoss:
    def __init__(self, cfg, device):
        import torch
        self.cfg = cfg
        self.regions = TaskRegions(cfg.path)
        self.names = list(cfg.tasks)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Specify distinct task names when region.weight > 0")
        if cfg.n_points < 1 or cfg.tolerance < 0:
            raise ValueError("Invalid region point count or tolerance")
        self.points = [torch.as_tensor(self.regions.sampled_points(n, cfg.n_points), device=device)
                       for n in self.names]

    def distances(self, points):
        import torch
        shape = points.shape[:-2]
        flat = points.reshape(-1, 3)
        distances = []
        for target in self.points:
            # Direct differences avoid cancellation near coincident surface points.
            d = torch.cdist(flat[None], target[None], compute_mode="donot_use_mm_for_euclid_dist")[0]
            distances.append(d.amin(dim=-1).reshape(*shape, -1).amin(dim=-1))
        return torch.stack(distances, dim=-1)

    def loss(self, distances):
        return 1000 * (distances - self.cfg.tolerance).clamp_min(0).square().sum(dim=-1)
