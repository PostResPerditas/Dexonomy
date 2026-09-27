"""Low-impedance single/sequential task execution with interchangeable pre-contact inputs."""

import argparse
from collections import Counter
import json
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


def precontact_path(state, grasp, scene, args):
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
    pre = precontact_path(state, grasp, scene, args)
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
        action_mode = "tcp" if args.task.startswith("move_stick_") else "fingers"
    base_id = model.body(prefix + "object_root").id
    hand_ids = {i for i in range(model.nbody) if model.body(i).name.startswith("hand-")}
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
        target_contact, base_contact, minimum = False, False, 0.0
        secondary_contact = False
        points = []
        for c in data.contact:
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if a in hand_ids and b not in hand_ids:
                other = b
            elif b in hand_ids and a not in hand_ids:
                other = a
            else:
                continue
            minimum = min(minimum, float(c.dist))
            if c.dist <= 0:
                points.append(c.pos.copy())
                target_contact |= other == bid
                if secondary:
                    secondary_contact |= other == sbid
                base_contact |= other == base_id
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
               or data.time <= before or any(w.number for w in data.warning))
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
    reached = False
    success = False
    reason = "simulation_unstable" if bad else "pretriggered" if premature else "motion_budget_exhausted"
    if not bad and not premature:
        count = max(1, int(np.ceil(args.action_seconds / dt)))
        budget = np.deg2rad(args.max_angle_deg) if hinge else args.max_translation
        for i in range(1, count + 1):
            amount = sign * budget * i / count
            q = start_command.copy()
            if action_mode == "fingers":
                q[7:] += (close_command[7:] - start_command[7:]) * i / count
            elif hinge:
                rotation = axangle2quat(axis, amount)
                q[:3] = pivot + quat2mat(rotation) @ (start_command[:3] - pivot)
                q[3:7] = qmult(rotation, start_command[3:7])
            else:
                q[:3] += axis * amount
            if not step(q, "action"):
                reason = "simulation_unstable"
                break
            if records["progress"][-1] >= threshold:
                reached = True
                reason = "hold_failed"
                break
        if reached and not bad:
            streak = 0.0
            for _ in range(max(1, int(np.ceil(args.hold_timeout / dt)))):
                if not step(command, "hold"):
                    reason = "simulation_unstable"
                    break
                condition = (records["progress"][-1] >= threshold
                             and (not args.require_contact_hold or records["target_contact"][-1]))
                streak = streak + dt if condition else 0.0
                if streak + 1e-9 >= args.hold_seconds:
                    success, reason = True, "success"
                    break
    if secondary:
        secondary["attempted"] = bool(success)
        secondary["action_start_frame"] = None
        if success:
            secondary["action_start_frame"] = len(records["time"])-1
            secondary["already_at_threshold"] = bool(records["secondary_progress"][-1] >= secondary["threshold"])
            secondary["target_contact_at_action_start"] = bool(records["secondary_contact"][-1])
            start_command = command.copy()
            close_command = closing_target(start_command)
            success = False
            reason = "secondary_motion_budget_exhausted"
            count = max(1, int(np.ceil(args.action_seconds / dt)))
            for i in range(1, count+1):
                q = start_command.copy()
                q[7:] += (close_command[7:]-start_command[7:]) * i/count
                if not step(q, "secondary_action"):
                    reason = "simulation_unstable"
                    break
                if records["secondary_progress"][-1] >= secondary["threshold"]:
                    reason = "joint_hold_failed"
                    streak = 0.0
                    for _ in range(max(1, int(np.ceil(args.hold_timeout / dt)))):
                        if not step(command, "joint_hold"):
                            reason = "simulation_unstable"
                            break
                        condition = (records["progress"][-1] >= threshold and
                                     records["secondary_progress"][-1] >= secondary["threshold"] and
                                     (not args.require_contact_hold or
                                      (records["target_contact"][-1] and records["secondary_contact"][-1])))
                        streak = streak+dt if condition else 0.0
                        if streak+1e-9 >= args.hold_seconds:
                            success, reason = True, "success"
                            break
                    break
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
        if not valid:
            summary.update(success=False, reason="invalid_physics")
    arrays.update(contact_points=np.asarray(contact_points).reshape(-1, 3),
                  contact_offsets=np.asarray(contact_offsets), precontact_qpos=pre,
                  metadata=np.array(json.dumps(summary)))
    return arrays, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--then-task", choices=[t for t in TASKS if not t.startswith("move_stick_")],
                        help="After the lever stage, hold TCP and close fingers for this task")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--hand", default="assets/hand/tianyi_right/right.xml")
    parser.add_argument("--selection", choices=["region-no-base", "region", "all"], default="region-no-base")
    parser.add_argument("--limit", type=int, default=10, help="First N sorted states; zero runs all")
    parser.add_argument("--pre-mode", choices=["native", "generate", "specified", "retreat"], default="native")
    parser.add_argument("--pre-state", type=Path)
    parser.add_argument("--pre-key", default="grasp_qpos")
    parser.add_argument("--pre-clearance", type=float, default=0.045)
    parser.add_argument("--retreat-direction", type=float, nargs=3)
    parser.add_argument("--approach-seconds", type=float, default=1.5)
    parser.add_argument("--settle-seconds", type=float, default=0.3)
    parser.add_argument("--action-seconds", type=float, default=2.0)
    parser.add_argument("--action-mode", choices=["auto", "fingers", "tcp"], default="auto",
                        help="Auto: lever uses TCP; buttons/trigger close all driven fingers with fixed TCP")
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
    args.output.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(env._model, str(args.output / "model.mjb"))
    results = []
    for i, path in enumerate(paths):
        state = np.load(path, allow_pickle=True).item()
        if str(args.scene or state["scene_path"]) != scene_path:
            raise ValueError("One execution batch must use one scene")
        arrays, result = execute(env, state, path, args, scene)
        trajectory = f"{i:04d}_{path.stem}.npz"
        np.savez_compressed(args.output / trajectory, **arrays)
        result["trajectory"] = trajectory
        results.append(result)
        print(f"{i+1}/{len(paths)} {path.name}: {result['reason']}; max={result['max_progress']:.6f} {result['units']}", flush=True)
    summary = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   scene_path=scene_path, model="model.mjb", total=len(results),
                   success=sum(r["success"] for r in results),
                   reasons=dict(Counter(r["reason"] for r in results)), results=results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("total", "success", "reasons")}), flush=True)


if __name__ == "__main__":
    main()
