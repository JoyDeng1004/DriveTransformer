#!/usr/bin/env python
import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from extract.petr_pe_extractor import PetrPEConfig, PetrPEExtractor  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    text = Path(path).read_text()
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_infos(path: str, info_root: Optional[str] = None) -> List[Dict[str, Any]]:
    raw = load_pickle(path)
    if isinstance(raw, dict) and {"routes_names", "divide_nums", "infos_dir_name"}.issubset(raw.keys()):
        root = Path(info_root) if info_root is not None else Path(path).resolve().parent
        route_dir = root / raw["infos_dir_name"]
        infos: List[Dict[str, Any]] = []
        for route_name in raw["routes_names"]:
            route_path = route_dir / f"{route_name}.pkl"
            route_infos = load_pickle(str(route_path))
            if not isinstance(route_infos, list):
                raise TypeError(f"Bench2Drive route file is not a frame list: {route_path}")
            infos.extend(route_infos)
        if raw.get("total_lenth") is not None and int(raw["total_lenth"]) != len(infos):
            print(f"Bench2Drive meta length diagnostic: total_lenth={raw['total_lenth']} loaded={len(infos)}")
        return infos
    if isinstance(raw, dict) and "infos" in raw:
        return list(raw["infos"])
    if isinstance(raw, dict) and "data_list" in raw:
        return list(raw["data_list"])
    if isinstance(raw, list):
        return raw
    raise TypeError(f"Unsupported info file structure in {path}. Pass a frame-list pickle or a dict with key 'infos'.")


def quat_to_rot(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[0] != 4:
        raise ValueError(f"Quaternion must have four values, got shape {q.shape}")
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_from_quat_translation(rotation: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_rot(rotation)
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def ensure_4x4_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return out


def image_aug_config(image_cfg: Dict[str, Any]) -> Dict[str, Any]:
    src_h, src_w = image_cfg.get("source_hw", [900, 1600])
    dst_h, dst_w = image_cfg.get("final_dim", [384, 1056])
    bot_pct_lim = image_cfg.get("bot_pct_lim", [0.0, 0.0])
    resize = max(float(dst_h) / float(src_h), float(dst_w) / float(src_w))
    new_w, new_h = int(src_w * resize), int(src_h * resize)
    crop_h = int((1 - float(np.mean(bot_pct_lim))) * new_h) - int(dst_h)
    crop_w = int(max(0, new_w - int(dst_w)) / 2)
    return {
        "resize": resize,
        "resize_dims": (new_w, new_h),
        "crop": (crop_w, crop_h, crop_w + int(dst_w), crop_h + int(dst_h)),
        "flip": False,
        "rotate": 0.0,
    }


def image_transform_matrix(aug: Dict[str, Any]) -> np.ndarray:
    resize = float(aug.get("resize", 1.0))
    crop = aug.get("crop", [0, 0, *aug.get("resize_dims", [0, 0])])
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] *= resize
    transform[:2, 2] -= np.asarray(crop[:2], dtype=np.float64)
    extend = np.eye(4, dtype=np.float64)
    extend[:3, :3] = transform
    return extend


def apply_image_geometry(
    lidar2img: np.ndarray,
    cam_intrinsic: np.ndarray,
    image_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    if not bool(image_cfg.get("apply_eval_resize", True)):
        return lidar2img, cam_intrinsic
    mat = image_transform_matrix(image_aug_config(image_cfg))
    out_lidar2img = np.asarray(lidar2img, dtype=np.float64).copy()
    out_intrinsic = np.asarray(cam_intrinsic, dtype=np.float64).copy()
    for i in range(out_lidar2img.shape[0]):
        out_lidar2img[i] = mat @ out_lidar2img[i]
        out_intrinsic[i, :3, :3] *= float(image_aug_config(image_cfg)["resize"])
    return out_lidar2img, out_intrinsic


def order_cameras(cams: Dict[str, Any]) -> List[str]:
    keys = [k for k in cams.keys() if "CAM" in k]
    return sorted(keys, key=lambda k: (0 if k == "CAM_FRONT" else 1, k))


def frame_to_model_input(info: Dict[str, Any], data_root: str) -> Dict[str, Any]:
    if all(k in info for k in ("lidar2img", "cam_intrinsic", "ego_pose")):
        img_filename = info.get("img_filename", [])
        return {
            "lidar2img": np.asarray(info["lidar2img"], dtype=np.float64),
            "cam_intrinsic": np.asarray(info["cam_intrinsic"], dtype=np.float64),
            "ego_pose": np.asarray(info["ego_pose"], dtype=np.float64),
            "img_filename": img_filename,
            "camera_names": info.get("camera_names", [f"CAM_{i}" for i in range(len(img_filename))]),
        }

    if "sensors" in info:
        sensors = info["sensors"]
        lidar2ego = np.asarray(sensors["LIDAR_TOP"].get("lidar2ego", np.eye(4)), dtype=np.float64)
        if "world2lidar" in sensors["LIDAR_TOP"]:
            ego_pose = np.linalg.inv(np.asarray(sensors["LIDAR_TOP"]["world2lidar"], dtype=np.float64))
        elif "ego_pose" in info:
            ego_pose = np.asarray(info["ego_pose"], dtype=np.float64)
        else:
            raise KeyError("Bench2Drive-style frame requires sensors.LIDAR_TOP.world2lidar or ego_pose")
        lidar2img, cam_intrinsic, img_filename, camera_names = [], [], [], []
        for cam_name in order_cameras(sensors):
            cam = sensors[cam_name]
            cam2ego = np.asarray(cam["cam2ego"], dtype=np.float64)
            intrinsic = ensure_4x4_intrinsic(cam["intrinsic"])
            lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
            lidar2img.append(intrinsic @ lidar2cam)
            cam_intrinsic.append(intrinsic)
            camera_names.append(cam_name)
            img_filename.append(os.path.join(data_root, cam.get("data_path", "")))
        return {
            "lidar2img": np.asarray(lidar2img, dtype=np.float64),
            "cam_intrinsic": np.asarray(cam_intrinsic, dtype=np.float64),
            "ego_pose": ego_pose,
            "img_filename": img_filename,
            "camera_names": camera_names,
        }

    if "cams" in info:
        lidar2ego = transform_from_quat_translation(info["lidar2ego_rotation"], info["lidar2ego_translation"])
        ego2global = transform_from_quat_translation(info["ego2global_rotation"], info["ego2global_translation"])
        ego_pose = ego2global @ lidar2ego
        lidar2img, cam_intrinsic, img_filename, camera_names = [], [], [], []
        for cam_name in order_cameras(info["cams"]):
            cam = info["cams"][cam_name]
            cam2lidar = np.eye(4, dtype=np.float64)
            cam2lidar[:3, :3] = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
            cam2lidar[:3, 3] = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
            intrinsic = ensure_4x4_intrinsic(cam["cam_intrinsic"])
            lidar2cam = np.linalg.inv(cam2lidar)
            lidar2img.append(intrinsic @ lidar2cam)
            cam_intrinsic.append(intrinsic)
            camera_names.append(cam_name)
            img_filename.append(os.path.join(data_root, cam.get("data_path", "")))
        return {
            "lidar2img": np.asarray(lidar2img, dtype=np.float64),
            "cam_intrinsic": np.asarray(cam_intrinsic, dtype=np.float64),
            "ego_pose": ego_pose,
            "img_filename": img_filename,
            "camera_names": camera_names,
        }
    raise KeyError("Frame metadata does not contain DriveTransformer, Bench2Drive, or nuScenes camera fields.")


def find_start_index(infos: List[Dict[str, Any]], scene_id: Optional[str], start_frame: int) -> int:
    if scene_id is None:
        return int(start_frame)
    candidates = []
    for i, info in enumerate(infos):
        keys = [info.get("scene_token"), info.get("folder"), info.get("scene_name"), info.get("log_token")]
        folder = info.get("folder")
        if folder is not None:
            keys.append(Path(str(folder)).name)
        if scene_id in keys:
            candidates.append(i)
    if not candidates:
        raise ValueError(f"scene_id={scene_id!r} was not found in the info file")
    return candidates[0] + int(start_frame)


def same_scene(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    for key in ("scene_token", "folder", "scene_name", "log_token"):
        if key in a or key in b:
            return a.get(key) == b.get(key)
    return True


def apply_se2_pose_delta(pose: np.ndarray, perturb: Dict[str, float]) -> np.ndarray:
    yaw = math.radians(float(perturb.get("dyaw_deg", 0.0)))
    c, s = math.cos(yaw), math.sin(yaw)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    delta[0, 3] = float(perturb.get("dx", 0.0))
    delta[1, 3] = float(perturb.get("dy", 0.0))
    return np.asarray(pose, dtype=np.float64) @ delta


def transform_points(mat: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    out = np.full_like(flat, np.nan, dtype=np.float64)
    hom = np.concatenate([flat[valid], np.ones((valid.sum(), 1), dtype=np.float64)], axis=1)
    out[valid] = (np.asarray(mat, dtype=np.float64) @ hom.T).T[:, :3]
    return out.reshape(pts.shape)


def select_static_points(cfg: Dict[str, Any], ref_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = np.asarray(cfg["sample"].get("static_anchors", []), dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] not in (2, 3) or anchors.shape[0] == 0:
        raise ValueError("mode_anchor requires sample.static_anchors as a non-empty list of [x, y] or [x, y, z]")
    if anchors.shape[1] == 2:
        anchors = np.concatenate([anchors, np.zeros((anchors.shape[0], 1), dtype=np.float64)], axis=1)
    world = transform_points(ref_pose, anchors)
    point_ids = np.asarray([f"anchor_{i}" for i in range(anchors.shape[0])], dtype=object)
    point_types = np.asarray(["static"] * anchors.shape[0], dtype=object)
    return world, point_ids, point_types


def select_dynamic_ids(ref_info: Dict[str, Any], max_agents: int) -> np.ndarray:
    ids = ref_info.get("gt_ids")
    boxes = ref_info.get("gt_boxes")
    if ids is None or boxes is None:
        raise KeyError("mode_dynamic requires gt_ids and gt_boxes in the frame metadata")
    ids = np.asarray(ids)
    return ids[:max_agents].astype(object)


def dynamic_points_for_frame(info: Dict[str, Any], selected_ids: np.ndarray, ego_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(info.get("gt_ids", []), dtype=object)
    boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64)
    local = np.full((selected_ids.shape[0], 3), np.nan, dtype=np.float64)
    if ids.shape[0] and boxes.ndim == 2 and boxes.shape[1] >= 3:
        for j, pid in enumerate(selected_ids):
            matches = np.where(ids == pid)[0]
            if matches.size:
                local[j] = boxes[matches[0], :3]
    world = transform_points(ego_pose, local)
    return local, world


def ray_points_world(
    cfg: Dict[str, Any],
    model_input: Dict[str, Any],
    ref_pose: np.ndarray,
    extractor: PetrPEExtractor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ray_cfg = cfg["sample"].get("ray", {})
    cam_idx = int(ray_cfg.get("camera_index", 0))
    uv = np.asarray(ray_cfg.get("pixel_uv", [0.0, 0.0]), dtype=np.float64)
    depths = np.asarray(ray_cfg.get("depths", extractor.coords_d.detach().cpu().numpy()), dtype=np.float64)
    lidar2img = np.asarray(model_input["lidar2img"][cam_idx], dtype=np.float64)
    img2lidar = np.linalg.inv(lidar2img)
    img_pts = np.stack([uv[0] * depths, uv[1] * depths, depths, np.ones_like(depths)], axis=1)
    local = (img2lidar @ img_pts.T).T[:, :3]
    world = transform_points(ref_pose, local)
    point_ids = np.asarray([f"ray_{i}" for i in range(depths.shape[0])], dtype=object)
    point_types = np.asarray(["static"] * depths.shape[0], dtype=object)
    return world, point_ids, point_types


def frame_meta(info: Dict[str, Any], model_input: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "sample_token": info.get("token", info.get("sample_idx", info.get("frame_idx", index))),
        "scene_id": info.get("scene_token", info.get("folder", info.get("scene_name", None))),
        "timestamp": info.get("timestamp", None),
        "front_image_path": model_input["img_filename"][0] if model_input.get("img_filename") else None,
        "camera_names": model_input.get("camera_names", []),
    }


def load_camera_images(model_inputs: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    if not bool(cfg["output"].get("store_camera_images", True)):
        return None
    try:
        from PIL import Image
    except ImportError:
        print("camera image diagnostic: PIL is not installed; camera image arrays were not stored")
        return None
    cam_idx = int(cfg["sample"].get("camera_index", 0))
    aug = image_aug_config(cfg["image"])
    images = []
    missing = []
    for t, mi in enumerate(model_inputs):
        paths = mi.get("img_filename", [])
        if cam_idx >= len(paths) or not paths[cam_idx] or not os.path.exists(paths[cam_idx]):
            missing.append((t, paths[cam_idx] if cam_idx < len(paths) else None))
            continue
        img = Image.open(paths[cam_idx]).convert("RGB")
        img = img.resize(tuple(aug["resize_dims"])).crop(tuple(aug["crop"]))
        images.append(np.asarray(img, dtype=np.uint8))
    if missing:
        print(f"camera image diagnostic: {len(missing)} images were not found; camera image arrays were not stored")
        for t, path in missing[:5]:
            print(f"  missing t={t}: {path}")
        return None
    if len(images) != len(model_inputs):
        print(f"camera image diagnostic: loaded={len(images)} target_count={len(model_inputs)}; camera image arrays were not stored")
        return None
    return np.stack(images, axis=0)


def summarize_array(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    print(
        f"{name}: shape={arr.shape}, finite={int(finite.sum())}/{arr.size}, "
        f"nan={int(np.isnan(arr).sum())}, inf={int(np.isinf(arr).sum())}"
    )


def output_name(cfg: Dict[str, Any], perturbed: bool) -> str:
    if not perturbed:
        return "samples_baseline.npz"
    p = cfg.get("perturb_ego_se2", {})
    return "samples_perturb_dx{:.3g}_dy{:.3g}_dyaw{:.3g}.npz".format(
        float(p.get("dx", 0.0)), float(p.get("dy", 0.0)), float(p.get("dyaw_deg", 0.0))
    )


def run(cfg: Dict[str, Any]) -> Path:
    infos = load_infos(cfg["dataset"]["ann_file"], cfg["dataset"].get("info_root"))
    start_idx = find_start_index(infos, cfg["dataset"].get("scene_id"), int(cfg["dataset"].get("start_frame", 0)))
    offsets = list(cfg["sample"].get("frame_offsets", [-2, -1, 0, 1, 2]))
    frame_indices = [start_idx + o for o in offsets]
    if min(frame_indices) < 0 or max(frame_indices) >= len(infos):
        raise IndexError(f"Frame window {frame_indices} is outside info file length {len(infos)}")
    ref_info = infos[start_idx]
    for idx in frame_indices:
        if not same_scene(ref_info, infos[idx]):
            raise ValueError(f"Frame index {idx} is not in the same scene as start index {start_idx}")

    image_hw = tuple(cfg["image"].get("final_dim", [384, 1056]))
    stride = int(cfg["image"].get("stride", 32))
    feature_hw = (int(math.ceil(image_hw[0] / stride)), int(math.ceil(image_hw[1] / stride)))
    extractor = PetrPEExtractor(PetrPEConfig(**cfg["pe"]))

    model_inputs = [frame_to_model_input(infos[idx], cfg["dataset"].get("data_root", "")) for idx in frame_indices]
    for mi in model_inputs:
        mi["lidar2img"], mi["cam_intrinsic"] = apply_image_geometry(mi["lidar2img"], mi["cam_intrinsic"], cfg["image"])
    base_ego_pose = np.stack([mi["ego_pose"] for mi in model_inputs]).astype(np.float64)
    ego_pose = base_ego_pose.copy()
    perturbed = bool(cfg.get("perturbed", False))
    perturb_config = cfg.get("perturb_ego_se2", {"dx": 0.0, "dy": 0.0, "dyaw_deg": 0.0})
    fixed_i = len(frame_indices) // 2
    if perturbed:
        ego_pose[fixed_i] = apply_se2_pose_delta(ego_pose[fixed_i], perturb_config)

    mode = cfg["sample"].get("mode", "mode_anchor")
    if mode == "mode_anchor":
        points_world_ref, point_ids, point_types = select_static_points(cfg, model_inputs[fixed_i]["ego_pose"])
        points_world = np.repeat(points_world_ref[None, :, :], len(frame_indices), axis=0)
        points_3d = np.stack([transform_points(np.linalg.inv(ego_pose[i]), points_world[i]) for i in range(len(frame_indices))])
    elif mode == "mode_dynamic":
        point_ids = select_dynamic_ids(infos[start_idx], int(cfg["sample"].get("max_dynamic_agents", 8)))
        point_types = np.asarray(["dynamic"] * point_ids.shape[0], dtype=object)
        local_world = [dynamic_points_for_frame(infos[idx], point_ids, base_ego_pose[i]) for i, idx in enumerate(frame_indices)]
        points_world = np.stack([lw[1] for lw in local_world])
        points_3d = np.stack([transform_points(np.linalg.inv(ego_pose[i]), points_world[i]) for i in range(len(frame_indices))])
    elif mode == "mode_ray":
        points_world_ref, point_ids, point_types = ray_points_world(cfg, model_inputs[fixed_i], model_inputs[fixed_i]["ego_pose"], extractor)
        points_world = np.repeat(points_world_ref[None, :, :], len(frame_indices), axis=0)
        points_3d = np.stack([transform_points(np.linalg.inv(ego_pose[i]), points_world[i]) for i in range(len(frame_indices))])
    else:
        raise ValueError(f"Unsupported sample.mode={mode!r}")

    t_count, n_count = points_3d.shape[:2]
    pe_vectors = np.full((t_count, n_count, int(cfg["pe"]["embed_dims"])), np.nan, dtype=np.float32)
    camera_uv = np.full((t_count, n_count, 2), np.nan, dtype=np.float32)
    valid_projection = np.zeros((t_count, n_count), dtype=bool)
    for i, mi in enumerate(model_inputs):
        enc = extractor.encode_points(
            points_3d[i],
            mi["lidar2img"],
            mi["cam_intrinsic"],
            image_hw=image_hw,
            camera_index=int(cfg["sample"].get("camera_index", 0)),
            feature_hw=feature_hw,
        )
        pe_vectors[i] = enc["pe"]
        camera_uv[i] = enc["uv"]
        valid_projection[i] = enc["valid"]

    meta = [frame_meta(infos[idx], model_inputs[i], idx) for i, idx in enumerate(frame_indices)]
    out_dir = Path(cfg["output"].get("data_dir", EXP_DIR / "outputs" / "data"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name(cfg, perturbed)
    payload = dict(
        points_3d=points_3d.astype(np.float32),
        points_world=points_world.astype(np.float32),
        pe_vectors=pe_vectors,
        point_ids=point_ids,
        point_types=point_types,
        camera_uv=camera_uv,
        ego_pose=ego_pose.astype(np.float32),
        perturbed=np.asarray(perturbed),
        perturb_config=np.asarray(perturb_config, dtype=object),
        frame_meta=np.asarray(meta, dtype=object),
        valid_projection=valid_projection,
        extractor_meta=np.asarray(extractor.metadata(), dtype=object),
        image_hw=np.asarray(image_hw, dtype=np.int64),
        feature_hw=np.asarray(feature_hw, dtype=np.int64),
        image_aug_config=np.asarray(image_aug_config(cfg["image"]), dtype=object),
    )
    camera_images = load_camera_images(model_inputs, cfg)
    if camera_images is not None:
        payload["camera_images"] = camera_images
    np.savez_compressed(out_path, **payload)
    print(f"wrote {out_path}")
    summarize_array("points_3d", points_3d)
    summarize_array("points_world", points_world)
    summarize_array("pe_vectors", pe_vectors)
    print(f"valid_projection: {int(valid_projection.sum())}/{valid_projection.size}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample 3D points and PETR-style PE vectors for experiment A.")
    parser.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    parser.add_argument("--override", action="append", default=[], help="JSON object merged into the config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for item in args.override:
        cfg = deep_update(cfg, json.loads(item))
    run(cfg)


if __name__ == "__main__":
    main()
