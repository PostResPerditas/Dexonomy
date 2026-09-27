"""Dexonomy baseline under the Intent final-state freeze protocol.

Run every stage, freezing its command at the first threshold crossing.
Success is the final task threshold conjunction; quality_success adds physics_valid.
Parallel/optimized version; both previous execution files are unchanged.

Adapted from Intent offline_evaluation/execute_final_state_freeze.py and the
local_precontact function in open_approach.py (2026-09-26 protocol).
Use --pre-mode local --physics-profile offline. Single buttons/trigger also
need --fix-other-object-joints; trigger needs --trigger-contact-linear.
Trigger direction uses explicit outward target_normal_world, otherwise the
first saved ho_c hand-to-trigger inward normal, without outcome-based search.
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from transforms3d.quaternions import axangle2quat, qmult, quat2mat

from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv, MuJoCo_OptCfg, MuJoCo_OptEnv
from dexonomy.util.file_util import load_scene_cfg
from dexonomy.util.np_util import np_interp_slide


TASKS = {
    "move_stick_forward": ("joystick_tilt_x", "stick_link", 1),
    "move_stick_backward": ("joystick_tilt_x", "stick_link", -1),
    "pull_trigger": ("trigger_00_joint", "trigger_00_link", 1),
    "press_button_center": ("button_00_joint", "button_00_link", 1),
    **{f"press_button_{i:02d}": (f"button_{i:02d}_joint", f"button_{i:02d}_link", 1)
       for i in range(1, 6)},
}


def hand_rows(value, size):
    rows = np.asarray(value, dtype=float).reshape(-1, size).copy()
    if not len(rows) or not np.isfinite(rows).all() or np.any(np.linalg.norm(rows[:, 3:7], axis=1) < 1e-8):
        raise ValueError("Invalid hand states; expected world xyz, wxyz, model-order joint qpos")
    rows[:, 3:7] /= np.linalg.norm(rows[:, 3:7], axis=1, keepdims=True)
    return rows


def local_precontact(grasp, reverse_start, model):
 data=mujoco.MjData(model)
 hand=[g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name.startswith('hand-') and (model.geom_contype[g] or model.geom_conaffinity[g])]
 obj=[g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name.startswith('obj-') and (model.geom_contype[g] or model.geom_conaffinity[g])]
 pairs=[(h,o) for h in hand for o in obj if (model.geom_contype[h]&model.geom_conaffinity[o]) or (model.geom_contype[o]&model.geom_conaffinity[h])]
 def gap(q):
  data.qpos[:]=model.qpos0;data.qpos[:19]=q;mujoco.mj_forward(model,data)
  return min(mujoco.mj_geomDistance(model,data,h,o,.02,None) for h,o in pairs)
 opened=grasp.copy()
 # Preserve thumb abduction; reduce each flexion chain by at most 0.03 rad at its driver.
 for chain in [[8,9,10],[11,12],[13,14],[15,16],[17,18]]:
  driver=grasp[chain[0]];scale=max(0.,driver-.03)/driver if driver>0 else 1.
  opened[chain]=grasp[chain]*scale
 baseline=gap(grasp)
 dirs=[];v=reverse_start[:3]-grasp[:3]
 if np.linalg.norm(v)>1e-9:dirs.append(v/np.linalg.norm(v))
 for i in range(48):
  z=1-2*(i+.5)/48;theta=i*np.pi*(3-np.sqrt(5));r=np.sqrt(1-z*z)
  dirs.append(np.array([r*np.cos(theta),r*np.sin(theta),z]))
 candidates=[]
 for distance in [.003,.005]:
  for direction in dirs:
   q=opened.copy();q[:3]+=distance*direction;candidates.append((gap(q),q))
 candidates.sort(key=lambda x:x[0],reverse=True)
 # Check the best endpoints along the entire sampled local approach, not only at reset.
 checked=[]
 for initial_gap,q in candidates[:8]:
  frames=np.array([q+(grasp-q)*t for t in np.linspace(0,1,11)])
  gaps=np.array([gap(f) for f in frames]);checked.append((gaps.min()>=min(0.,baseline)-1e-5,initial_gap,gaps.min(),frames,gaps))
 valid=[x for x in checked if x[0] and x[1]>0]
 best=max(valid,key=lambda x:x[1]) if valid else max(checked,key=lambda x:(x[2],x[1]))
 _,initial_gap,minimum,frames,gaps=best
 return frames[:-1],dict(direction=(frames[0,:3]-grasp[:3]).tolist(),initial_gap_mm=initial_gap*1000,path_min_gap_mm=minimum*1000,grasp_gap_mm=baseline*1000,clear_start_and_no_added_sampled_penetration=bool(valid),waypoint_gaps_mm=(gaps*1000).tolist())


def precontact_path(state, grasp, scene, args, model):
    if args.pre_mode == "local":
        reverse_start = hand_rows(state["pregrasp_qpos"], len(grasp))[0]
        pre, diagnostic = local_precontact(grasp, reverse_start, model)
        state["local_precontact_diagnostic"] = diagnostic
        return pre
    if args.pre_mode == "native":
        if "pregrasp_qpos" not in state:
            raise ValueError("No native pregrasp_qpos; choose generate, specified or retreat")
        return hand_rows(state["pregrasp_qpos"], len(grasp))
    if args.pre_mode == "specified":
        value = np.load(args.pre_state, allow_pickle=True)
        if value.shape == ():
            value = value.item()[args.pre_key]
        return hand_rows(value, len(grasp))
    if args.pre_mode == "retreat":
        start = grasp.copy()
        if args.retreat_direction is None:
            centers = np.asarray(state.get("ho_c", {}).get("pos", []))
            if centers.size == 0:
                raise ValueError("Specify --retreat-direction when no contact positions are available")
            direction = grasp[:3] - centers.mean(axis=0)
        else:
            direction = np.asarray(args.retreat_direction, dtype=float)
        if np.linalg.norm(direction) < 1e-8:
            raise ValueError("Retreat direction must be nonzero")
        start[:3] += args.pre_clearance * direction / np.linalg.norm(direction)
        return start[None]
    env = MuJoCo_OptEnv(HandCfg(args.hand, freejoint=True), scene,
                        sim_cfg=MuJoCo_OptCfg(), debug_view=False)
    env.reset_qpos(grasp)
    path = []
    # Same margin-expansion mechanism as native Dexonomy pregrasp generation.
    for i in range(20):
        env.set_obj_margin(args.pre_clearance * min((i + 1) / 10, 1))
        env.keep_hand_stable()
        env.step_sim(10)
        path.append(env.get_hand_qpos().copy())
    return hand_rows(np.asarray(path)[::-2], len(grasp))


def select_files(args):
    if args.data.is_file():
        return [args.data.resolve()]
    if args.selection == "all":
        paths = sorted(args.data.rglob("*.npy"))
    else:
        index = json.loads((args.data.parent / "contact_labels.json").read_text())
        paths = sorted(args.data / r["path"] for r in index["records"]
                       if args.task in r["region_labels"]
                       and (not args.then_task or args.then_task in r["region_labels"])
                       and (args.selection != "region-no-base" or not r["base_near_contact"]))
    if not paths:
        raise ValueError("No selected states")
    return [p.resolve() for p in (paths if args.limit == 0 else paths[:args.limit])]


def execute(env, state, source, args, scene):
    model, data = env._model, env._data
    grasp = hand_rows(state["grasp_qpos"], len(env.get_hand_qpos()))[0]
    pre = precontact_path(state, grasp, scene, args, model)
    env.reset_qpos(pre[0])
    joint, body, sign = TASKS[args.task]
    prefix = "obj-" + scene["task"]["obj_name"]
    jid = model.joint(prefix + joint).id
    bid = model.body(prefix + body).id
    adr = model.jnt_qposadr[jid]
    q0 = float(data.qpos[adr])
    limit = float(model.jnt_range[jid, 1 if sign > 0 else 0])
    travel = sign * (limit - q0)
    threshold = args.success_fraction * travel
    secondary = None
    if args.then_task:
        sjoint, sbody, ssign = TASKS[args.then_task]
        sjid = model.joint(prefix + sjoint).id
        sadr = model.jnt_qposadr[sjid]
        secondary = dict(task=args.then_task, joint_name=prefix+sjoint,
                         units="rad" if model.jnt_type[sjid] == mujoco.mjtJoint.mjJNT_HINGE else "m",
                         initial_joint_qpos=float(data.qpos[sadr]),
                         threshold=float(args.success_fraction * ssign *
                                         (model.jnt_range[sjid, 1 if ssign > 0 else 0] - data.qpos[sadr])))
        sbid = model.body(prefix+sbody).id
    if threshold <= 0:
        raise ValueError("Task has no available travel in the requested direction")
    hinge = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
    action_mode = args.action_mode
    if action_mode == "auto":
        action_mode = "tcp"  # Single tasks use the target joint axis; combined secondary stays finger-driven.
    base_id = model.body(prefix + "object_root").id
    is_hand = np.array([model.body(i).name.startswith("hand-") for i in range(model.nbody)])
    dt = model.opt.timestep * args.substeps
    records = {k: [] for k in ("time", "phase", "qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat",
                               "command", "progress", "target_contact", "base_contact", "min_contact_dist")}
    if secondary:
        records.update(secondary_progress=[], secondary_contact=[])
    contact_points, contact_offsets = [], [0]
    bad = False
    premature = False
    command = pre[0].copy()

    def record(phase):
        bodies_a = model.geom_bodyid[data.contact.geom1]
        bodies_b = model.geom_bodyid[data.contact.geom2]
        hand_a, hand_b = is_hand[bodies_a], is_hand[bodies_b]
        hand_object = hand_a != hand_b
        distances = data.contact.dist
        minimum = float(np.minimum(0.0, distances[hand_object].min(initial=0.0)))
        touching = hand_object & (distances <= 0)
        other = np.where(hand_a, bodies_b, bodies_a)[touching]
        target_contact = bool(np.any(other == bid))
        base_contact = bool(np.any(other == base_id))
        secondary_contact = bool(np.any(other == sbid)) if secondary else False
        points = data.contact.pos[touching].copy()
        values = dict(time=float(data.time), phase=phase, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                      ctrl=data.ctrl.copy(), mocap_pos=data.mocap_pos.copy(), mocap_quat=data.mocap_quat.copy(),
                      command=command.copy(), progress=sign * (float(data.qpos[adr])-q0),
                      target_contact=target_contact, base_contact=base_contact, min_contact_dist=minimum)
        if secondary:
            values.update(secondary_progress=ssign*(float(data.qpos[sadr])-secondary["initial_joint_qpos"]),
                          secondary_contact=secondary_contact)
        for key, value in values.items():
            records[key].append(value)
        contact_points.extend(points)
        contact_offsets.append(len(contact_points))

    def step(next_command, phase):
        nonlocal command, bad, premature
        command = next_command.copy()
        env.set_ctrl(command)
        before = data.time
        env.step_sim(args.substeps)
        mujoco.mj_forward(model, data)
        bad = (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
               or data.time <= before or data.warning.number.any())
        record(phase)
        if phase in ("approach", "settle") and (records["progress"][-1] >= threshold or
                (secondary and records["secondary_progress"][-1] >= secondary["threshold"])):
            premature = True
        return not bad

    def transition(start, end, seconds, phase):
        count = max(1, int(np.ceil(seconds / dt)))
        poses = np_interp_slide(start[:7], end[:7], count)
        for i, pose in enumerate(poses):
            q = start + (end - start) * ((i + 1) / count)
            q[:7] = pose
            if not step(q, phase):
                return False
        return True

    record("reset")
    waypoints = np.vstack([pre, grasp])
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        if not transition(start, end, args.approach_seconds / (len(waypoints)-1), "approach"):
            break
    contact_command = grasp.copy()
    if args.squeeze_blend:
        if "squeeze_qpos" not in state:
            raise ValueError("squeeze-blend requires squeeze_qpos; use zero for other methods")
        squeeze = hand_rows(state["squeeze_qpos"], len(grasp))[0]
        contact_command[7:] += args.squeeze_blend * (squeeze[7:] - grasp[7:])
    if not bad:
        transition(command, contact_command, args.settle_seconds, "settle")
    action_start = len(records["time"]) - 1
    contact_at_start = records["target_contact"][-1]
    pivot = data.xanchor[jid].copy()
    axis = data.xaxis[jid].copy()
    linear_contact = args.trigger_contact_linear and args.task == "pull_trigger" and not args.then_task
    if linear_contact:
        # Explicit outward normal, or the first saved hand-to-trigger inward normal.
        if "target_normal_world" in state:
            axis = -np.asarray(state["target_normal_world"], dtype=float).reshape(3)
            direction_source = "target_normal_world (outward)"
        else:
            contacts = state["ho_c"]
            indices = [i for i, name in enumerate(contacts["bn2"])
                       if str(name) == prefix + "trigger_00_link"]
            if not indices:
                raise ValueError("Single trigger needs target_normal_world or a saved trigger contact")
            axis = np.asarray(contacts["normal"][indices[0]], dtype=float)
            direction_source = f"ho_c.normal[{indices[0]}] (hand-to-object inward)"
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError("Invalid trigger contact normal")
        axis = axis / np.linalg.norm(axis)
    start_command = command.copy()
    closing_joints = []
    def closing_target(start_command):
        close_command = start_command.copy()
        hand_joints = [j for j in range(model.njnt)
                       if model.joint(j).name.startswith("hand-")
                       and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        # Tianyi closes toward the upper limits; mimic joints remain physics-driven.
        for actuator in range(model.nu):
            j = int(model.actuator_trnid[actuator, 0])
            if j not in hand_joints:
                continue
            index = 7 + hand_joints.index(j)
            upper = model.jnt_range[j, 1]
            if model.actuator_ctrllimited[actuator]:
                upper = min(upper, model.actuator_ctrlrange[actuator, 1])
            close_command[index] += args.close_fraction * max(0, upper - start_command[index])
            if model.joint(j).name not in closing_joints:
                closing_joints.append(model.joint(j).name)
        return close_command
    close_command = closing_target(start_command) if action_mode == "fingers" else start_command.copy()
    # Keep stage durations, but stop increasing the command once its task reaches
    # threshold. A later loss of progress is evaluated at the final state.
    primary_frozen = bool(records["progress"][-1] >= threshold)
    secondary_frozen = False
    if not bad:
        count = max(1, int(np.ceil(args.action_seconds / dt)))
        budget = np.deg2rad(args.max_angle_deg) if hinge and not linear_contact else args.max_translation
        for i in range(1, count + 1):
            amount = sign * budget * i / count
            q = start_command.copy()
            if action_mode == "fingers":
                q[7:] += (close_command[7:] - start_command[7:]) * i / count
            elif hinge and not linear_contact:
                rotation = axangle2quat(axis, amount)
                q[:3] = pivot + quat2mat(rotation) @ (start_command[:3] - pivot)
                q[3:7] = qmult(rotation, start_command[3:7])
            else:
                q[:3] += axis * amount
            if primary_frozen:
                q = command.copy()
            if not step(q, "action"):
                break
            primary_frozen = primary_frozen or records["progress"][-1] >= threshold
    if secondary:
        secondary["attempted"] = not bad
        secondary["action_start_frame"] = None
        if not bad:
            secondary["action_start_frame"] = len(records["time"])-1
            secondary["already_at_threshold"] = bool(records["secondary_progress"][-1] >= secondary["threshold"])
            secondary["target_contact_at_action_start"] = bool(records["secondary_contact"][-1])
            secondary_frozen = bool(records["secondary_progress"][-1] >= secondary["threshold"])
            start_command = command.copy()
            close_command = closing_target(start_command)
            count = max(1, int(np.ceil(args.action_seconds / dt)))
            for i in range(1, count+1):
                q = start_command.copy()
                q[7:] += (close_command[7:]-start_command[7:]) * i/count
                if secondary_frozen:
                    q = command.copy()
                if not step(q, "secondary_action"):
                    break
                secondary_frozen = secondary_frozen or records["secondary_progress"][-1] >= secondary["threshold"]
    # A fixed final observation period, with no threshold-triggered early exit.
    if not bad:
        for _ in range(max(1, int(np.ceil(args.hold_seconds / dt)))):
            if not step(command, "final_observation"):
                break
    success = bool(not bad and records["progress"][-1] >= threshold and
                   (not secondary or records["secondary_progress"][-1] >= secondary["threshold"]))
    reason = "simulation_unstable" if bad else "success" if success else "final_threshold_not_reached"
    if secondary:
        secondary.update(max_progress=float(max(records["secondary_progress"])),
                         final_progress=float(records["secondary_progress"][-1]),
                         contact_at_end=bool(records["secondary_contact"][-1]))
    summary = dict(source=str(source), task=args.task, success=success, reason=reason,
                   secondary=secondary,
                   action_mode=action_mode, closing_joints=closing_joints,
                   close_fraction=args.close_fraction,
                   joint_name=prefix+joint, units="rad" if hinge else "m", sign=sign,
                   initial_joint_qpos=q0, threshold=threshold, action_start_frame=action_start,
                   pretriggered=premature, target_contact_at_action_start=bool(contact_at_start),
                   contact_hold_required=args.require_contact_hold,
                   max_progress=float(max(records["progress"])), final_progress=float(records["progress"][-1]),
                   base_contact_any=bool(any(records["base_contact"])),
                   min_contact_dist=float(min(records["min_contact_dist"])),
                   physics="gravity disabled; passive object joints; no added springs or external forces",
                   pose_convention="world xyz,wxyz,model-order hand joints; object joints only moved by simulation")
    summary["evaluation_protocol"] = "intent_final_state_freeze_on_threshold_v1"
    summary["primary_command_frozen"] = bool(primary_frozen)
    summary["secondary_command_frozen"] = bool(secondary_frozen) if secondary else None
    summary["tcp_contact_direction_world"] = axis.tolist() if linear_contact else None
    summary["trigger_direction_source"] = direction_source if linear_contact else None
    summary["local_precontact_diagnostic"] = state.get("local_precontact_diagnostic")
    summary["other_object_joints_fixed"] = args.fix_other_object_joints
    summary["task_success"] = success
    summary["task_reason"] = reason
    arrays = {k: np.asarray(v) for k, v in records.items()}
    if args.physics_profile == "offline":
        # Offline mode records every physics step, including approach and settling.
        max_penetration = max(0.0, -summary["min_contact_dist"])
        overtravel = []
        for j in range(model.njnt):
            if not model.joint(j).name.startswith(prefix) or not model.jnt_limited[j]:
                continue
            values = arrays["qpos"][:, model.jnt_qposadr[j]]
            excess = float(max(0, np.max(model.jnt_range[j, 0]-values),
                               np.max(values-model.jnt_range[j, 1])))
            tolerance = args.max_slide_overtravel if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else args.max_hinge_overtravel
            if excess > tolerance:
                overtravel.append(dict(joint=model.joint(j).name, excess=excess, tolerance=tolerance))
        valid = not bad and max_penetration <= args.max_penetration and not overtravel
        summary.update(task_success=success, task_reason=reason, physics_valid=bool(valid),
                       max_penetration=max_penetration, overtravel=overtravel,
                       physics_profile=args.physics_profile)
        summary["quality_success"] = bool(success and valid)
    arrays.update(contact_points=np.asarray(contact_points).reshape(-1, 3),
                  contact_offsets=np.asarray(contact_offsets), precontact_qpos=pre,
                  metadata=np.array(json.dumps(summary)))
    return arrays, summary


def make_env(args, scene):
    env = MuJoCo_EvalEnv(HandCfg(args.hand, freejoint=True), scene,
                        sim_cfg=MuJoCo_EvalCfg(timestep=args.timestep, miu_coef=(0.6, 0.02)), debug_view=False)
    if args.physics_profile == "offline":
        model = env._model
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 200
        model.opt.ls_iterations = 50
        model.opt.tolerance = 1e-10
        model.opt.noslip_iterations = 10
        collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_solref[collision] = [0.004, 1]
        model.geom_solimp[collision] = [0.99, 0.999, 0.0001, 0.5, 2]
        model.jnt_solref[:] = [0.004, 1]
        model.jnt_solimp[:] = [0.99, 0.999, 0.0001, 0.5, 2]
    return env


def compact_trace(arrays, result, every):
    n = len(arrays["time"])
    changes = np.flatnonzero(arrays["phase"][1:] != arrays["phase"][:-1]) + 1
    boundaries = [0, n-1, result["action_start_frame"]]
    secondary = result.get("secondary")
    if secondary and secondary["action_start_frame"] is not None:
        boundaries.append(secondary["action_start_frame"])
    keep = np.unique(np.concatenate([np.arange(0, n, every), changes, changes-1, boundaries])).astype(int)
    offsets = arrays["contact_offsets"]
    chunks = [arrays["contact_points"][offsets[i]:offsets[i+1]] for i in keep]
    saved = {k: v[keep] for k, v in arrays.items()
             if k not in ("contact_points", "contact_offsets", "precontact_qpos", "metadata")}
    saved["contact_points"] = np.concatenate(chunks, axis=0)
    saved["contact_offsets"] = np.concatenate([[0], np.cumsum([len(c) for c in chunks])])
    saved["precontact_qpos"] = arrays["precontact_qpos"]
    saved["physics_frame"] = keep
    result["physics_action_start_frame"] = result["action_start_frame"]
    result["action_start_frame"] = int(np.searchsorted(keep, result["action_start_frame"]))
    if secondary and secondary["action_start_frame"] is not None:
        secondary["physics_action_start_frame"] = secondary["action_start_frame"]
        secondary["action_start_frame"] = int(np.searchsorted(keep, secondary["action_start_frame"]))
    result.update(physics_frame_count=n, saved_frame_count=len(keep), record_every=every)
    saved["metadata"] = np.array(json.dumps(result))
    return saved


def init_worker(args, scene, scene_path):
    global _env, _args, _scene, _scene_path
    _args, _scene, _scene_path = args, scene, scene_path
    _env = make_env(args, scene)


def run_candidate(job):
    i, path = job
    state = np.load(path, allow_pickle=True).item()
    if str(_args.scene or state["scene_path"]) != _scene_path:
        raise ValueError("One execution batch must use one scene")
    start = time.perf_counter()
    arrays, result = execute(_env, state, path, _args, _scene)
    result["execution_seconds"] = time.perf_counter() - start
    arrays = compact_trace(arrays, result, _args.record_every)
    trajectory = f"{i:04d}_{path.stem}.npz"
    np.savez_compressed(_args.output / trajectory, **arrays)
    result["trajectory"] = trajectory
    return i, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger-contact-linear", action="store_true")
    parser.add_argument("--fix-other-object-joints", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--record-every", type=int, default=1,
                        help="Save every Nth physics frame plus boundaries; all physics steps still checked")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--then-task", choices=[t for t in TASKS if not t.startswith("move_stick_")],
                        help="After the lever stage, hold TCP and close fingers for this task")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--hand", default="assets/hand/tianyi_right/right.xml")
    parser.add_argument("--selection", choices=["region-no-base", "region", "all"], default="region-no-base")
    parser.add_argument("--limit", type=int, default=10, help="First N sorted states; zero runs all")
    parser.add_argument("--pre-mode", choices=["native", "generate", "specified", "retreat", "local"], default="native")
    parser.add_argument("--pre-state", type=Path)
    parser.add_argument("--pre-key", default="grasp_qpos")
    parser.add_argument("--pre-clearance", type=float, default=0.045)
    parser.add_argument("--retreat-direction", type=float, nargs=3)
    parser.add_argument("--approach-seconds", type=float, default=1.5)
    parser.add_argument("--settle-seconds", type=float, default=0.3)
    parser.add_argument("--action-seconds", type=float, default=2.0)
    parser.add_argument("--action-mode", choices=["auto", "fingers", "tcp"], default="auto",
                        help="Auto: primary/single task uses TCP; combined secondary uses finger closure")
    parser.add_argument("--close-fraction", type=float, default=1.0,
                        help="Fraction of remaining travel toward Tianyi upper joint limits")
    parser.add_argument("--max-angle-deg", type=float, default=25)
    parser.add_argument("--max-translation", type=float, default=0.012)
    parser.add_argument("--success-fraction", type=float, default=0.5)
    parser.add_argument("--hold-seconds", type=float, default=0.2)
    parser.add_argument("--hold-timeout", type=float, default=0.5)
    parser.add_argument("--require-contact-hold", action="store_true",
                        help="Also require uninterrupted physical target contact during the success hold")
    parser.add_argument("--squeeze-blend", type=float, default=0.0)
    parser.add_argument("--timestep", type=float, default=0.004)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--physics-profile", choices=["native", "offline"], default="native",
                        help="Offline fixes timestep=0.0005 and substeps=1, with stiff contacts and quality checks")
    parser.add_argument("--max-penetration", type=float, default=0.002)
    parser.add_argument("--max-slide-overtravel", type=float, default=0.0002)
    parser.add_argument("--max-hinge-overtravel", type=float, default=0.02)
    args = parser.parse_args()
    if args.workers < 1 or args.record_every < 1:
        parser.error("workers and record-every must be positive")
    if args.trigger_contact_linear and (args.task != "pull_trigger" or args.then_task or args.action_mode == "fingers"):
        parser.error("Contact-linear direction is for a single TCP trigger task")
    if args.require_contact_hold:
        parser.error("Final-state protocol does not require contact hold; omit --require-contact-hold")
    if args.physics_profile == "offline":
        args.timestep, args.substeps = 0.0005, 1
    if min(args.max_penetration, args.max_slide_overtravel, args.max_hinge_overtravel) <= 0:
        parser.error("Physics quality tolerances must be positive")
    if args.then_task and (not args.task.startswith("move_stick_") or args.action_mode == "fingers"):
        parser.error("Sequential execution requires a TCP lever task first")
    if args.pre_mode == "specified" and args.pre_state is None:
        parser.error("specified mode needs --pre-state")
    if not 0 < args.success_fraction <= 1 or not 0 <= args.squeeze_blend <= 1 or args.limit < 0:
        parser.error("Invalid success fraction, squeeze blend or limit")
    if not 0 < args.close_fraction <= 1:
        parser.error("close-fraction must be in (0, 1]")
    if min(args.pre_clearance, args.approach_seconds, args.settle_seconds, args.action_seconds,
           args.max_angle_deg, args.max_translation, args.hold_seconds, args.timestep, args.substeps) <= 0:
        parser.error("Motion budgets and durations must be positive")
    if args.hold_timeout < args.hold_seconds:
        parser.error("hold-timeout must be at least hold-seconds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory for a new trial")
    paths = select_files(args)
    first = np.load(paths[0], allow_pickle=True).item()
    scene_path = str(args.scene or first["scene_path"])
    scene = load_scene_cfg(scene_path)
    if args.fix_other_object_joints:
        if args.then_task or args.task.startswith("move_stick_"):
            parser.error("Target-only joint mode is for single button/trigger tasks")
        target_joint = TASKS[args.task][0]
        obj = scene["scene"][scene["task"]["obj_name"]]
        if not np.allclose(obj["qpos"], 0):
            raise ValueError("Target-only model currently requires neutral object joints")
        original_path = Path(obj["xml_path"]).resolve()
        xml = ET.parse(original_path)
        compiler = xml.getroot().find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        for mesh in xml.getroot().iter("mesh"):
            if mesh.get("file"):mesh.set("file", str((original_path.parent / meshdir / mesh.get("file")).resolve()))
        if compiler is not None:compiler.set("meshdir", "")
        for body in xml.getroot().iter("body"):
            for j in list(body.findall("joint")):
                if j.get("name") != target_joint:body.remove(j)
        args.output.mkdir(parents=True, exist_ok=True)
        object_xml = args.output.resolve()/"target_only_object.xml"
        xml.write(object_xml)
        obj["xml_path"] = str(object_xml)
        obj["qpos"] = np.zeros(1)
    env = make_env(args, scene)
    args.output.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(env._model, str(args.output / "model.mjb"))
    results = [None] * len(paths)
    start = time.perf_counter()
    def collect(i, result, finished):
        results[i] = result
        print(f"{finished}/{len(paths)} {paths[i].name}: {result['reason']}; "
              f"final={result['final_progress']:.6f} {result['units']}", flush=True)
    if args.workers == 1:
        global _env, _args, _scene, _scene_path
        _env, _args, _scene, _scene_path = env, args, scene, scene_path
        for finished, job in enumerate(enumerate(paths), 1):
            i, result = run_candidate(job)
            collect(i, result, finished)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker, initargs=(args, scene, scene_path)) as pool:
            pending = [pool.submit(run_candidate, job) for job in enumerate(paths)]
            for finished, future in enumerate(as_completed(pending), 1):
                i, result = future.result()
                collect(i, result, finished)
    elapsed = time.perf_counter() - start
    summary = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   scene_path=scene_path, model="model.mjb", total=len(results),
                   batch_seconds=elapsed, samples_per_second=len(results)/elapsed,
                   success=sum(r["success"] for r in results),
                   task_success=sum(r["task_success"] for r in results),
                   physics_valid=sum(r.get("physics_valid", False) for r in results),
                   quality_success=sum(r.get("quality_success", False) for r in results),
                   reasons=dict(Counter(r["reason"] for r in results)), results=results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("total", "success", "reasons")}), flush=True)


if __name__ == "__main__":
    main()
