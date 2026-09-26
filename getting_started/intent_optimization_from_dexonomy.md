# Intent 优化建议：借鉴 Dexonomy 的搜索与接触细化

日期：2026-09-24。本文依据当前本地实现的只读对照，不是耗时 profiling 结论；未修改 Intent 源码。

Dexonomy 根目录：`/mnt/ssd/Bennkyou/PROJECT/Dexonomy`。
Intent 根目录：`/mnt/ssd/Bennkyou/PROJECT/InHandMani/intent`。

## 目标与边界

保留 Intent 的任务语义、操作方向及功能执行能力，减少无效候选的精修成本。状态生成与执行评估继续分开，统一物体坐标系、手驱动顺序和状态接口。近接触、几何可行、可执行及功能成功必须分别统计。

## Dexonomy 的具体实现

| 策略 | 源码（相对 Dexonomy 根目录） | 实现方式及限制 |
|---|---|---|
| 模板手型 | `dexonomy/data/hand_loader.py`，`HandTemplateLoader._load_data/get_batched_data` | 保留模板关节构型和接触点；随机选一个模板接触点建立局部接触坐标系，批量抽取模板。不是随机初始化全部手指关节。 |
| 表面驱动初始化 | `dexonomy/data/obj_loader.py`，`sample_init_pose` | 表面采样，利用法向构造朝向，再采样面内旋转。铰接物体读取 `task.part_name` 指定的 `part_info` 网格。 |
| 低维批量匹配 | `dexonomy/op/gen_init.py`，`HandObjMatcher.forward` | 仅优化旋转四元数和平移，手指关节固定。用 Warp 查询目标最近点及法向，Adam 优化位置平方误差和法向向量误差。 |
| 初始化筛选 | 同上，`InitFilter`、`valid_index_padding` | 使用位姿、匹配误差、骨架碰撞、接触力 QP 和 FPS；候选不足时会补回未通过当前筛选的候选，因此 `Get N valid` 不等于严格有效 N 个。配置阈值须按代码实际量纲理解，不能只看 YAML 注释。 |
| 接触力细化 | `dexonomy/op/gen_grasp.py`，`_single_grasp`；`dexonomy/sim/mujoco_env.py`，`MuJoCo_OptEnv.apply_contact_forces` | 接触误差拆成法向和切向，分别乘 500 和 100，经过 `mj_applyFT` 转成广义力，配合短程仿真调整状态。默认 grasp 上限 20 轮、每轮 10 个 substep，实际可能提前结束。 |
| 输出筛选 | `dexonomy/op/gen_grasp.py`，`GraspFilter` | 检查关节范围、自碰撞、手物碰撞、必需接触组和 QP。必需接触组按手部 body 判断，不自动要求接触指定按钮或扳机。 |
| 控制与状态区分 | `dexonomy/sim/mujoco_env.py`，`get_squeeze_qpos` | 从接触力构造挤压控制目标；`squeeze_qpos` 不是已达到的物理状态。可视化和状态比较使用 `grasp_qpos`。 |

参考配置：`dexonomy/config/op/init.yaml`、`dexonomy/config/op/grasp.yaml`。
默认初始化每物体有 1024 个表面点、8 个面内旋转，匹配 100 步；不能只用最终保留的候选数计算总采样成本。

虚拟接触吸引力是生成工具，不是实际机器人可施加的外力。采用这种细化后，必须移除虚拟外力并通过统一执行评估，不能把仿真贴合直接解释为真实可执行。

## Intent 的可定位改进项

以下路径相对 Intent 根目录。先确认实际运行入口和命令覆盖参数，再确定改动范围。

### 1. 粗优化后筛选，再投入密集精修

`optim/batched_adam_single_hand.py::BatchedAdamSingleHandOptimizer.run` 中已有粗阶段排序和 `adam_fine_candidates` 截断。
但 `optim/batch_generation/optimize_pregrasp_seed_batch.py::PregraspSeedBatchOptimizer.optimize_group` 将全部 `coarse_u` 传入 fine 阶段，未执行同样的 Top-K。
`optimize_retrieved_stage_a_local_dual_sdf.py::optimize_group` 经 `super()` 复用该路径。

建议先按几何可行性、任务区域距离、法向和可达性过滤，再按任务部件、接触组、接近方向分层保留候选。避免仅按总损失排序造成模式坍缩。筛选后须同步切片参考位姿、局部上下界及 source IDs，特别是 local dual SDF 子类保存的逐候选状态。

### 2. 冻结候选后实际减少计算

`optim/batched_adam_single_hand.py::_optimize_stage` 每轮仍对完整 `u` 调用 `_loss`，再以 `active` 掩码控制目标和参数更新；冻结不等于停止该候选的 FK/SDF 查询。
开始冻结判定后，还会对更新后的状态再次调用 `_loss`，虽然没有反向图，仍有前向查询开销。

建议每若干步压缩活跃候选，保留原始索引、已完成输出及相应 Adam 一阶/二阶状态；冻结判定改为间隔执行。先单独测量双前向和活跃比例，避免为小批次引入超过收益的索引开销。

### 3. 粗阶段低维化与稀疏化

`optim/batched_adam_single_hand.py::_loss` 包括密集手表面 FK、目标 SDF、任务区域距离、法向和避碰查询。
`optim/batch_generation/optimize_retrieved_stage_a_local_dual_sdf.py::_collision_penetration` 对多个 avoidance link 查询手表面点。

建议最早的任务匹配阶段固定手型，只优化整体位姿；使用少量接触锚点、现有球体/骨架代理和保守的近邻部件筛选。通过后再开放相关的真实驱动关节，使用密集表面和完整几何做精修及最终检查。远场筛选不能漏掉底座等障碍；保留 mimic 关系，不能把从动关节当独立变量。

### 4. 缩小几何修复职责

`optim/batch_generation/optimize_stage_g_geometry.py::optimize_batch` 按固定 `steps` 执行，默认 110 步，其中前 45 步为 repair phase。

对深穿透、明显不可达、长时间无改善的候选提前淘汰；可行候选提前完成。几何修复应解决局部缺陷，不应替代重新采样或大范围重新构型。针对固定物体与中立状态，考虑复用已有的任务无关几何状态库，而不是每个任务重新完成同一轮修复。

### 5. 接触代理优化与最终判定分开

`optim/batched_adam_single_hand.py::_loss` 使用 soft assignment 聚合接触、区域及法向指标，`_surrogate_success` 判断这些代理量。

聚合指标适合可微优化，但可能掩盖局部接触问题。最终应检验具体点是否同时满足距离、区域、法向和穿透条件，再按实际手部接触组统计覆盖。MuJoCo 近接触、碰撞近似与视觉表面也要对齐；不能把间隙 2 mm 的近接触直接视作承载接触或功能成功。

### 6. 可选的短程物理贴合

先完成以上改动，再比较“现有精修”与“短程虚拟接触力贴合”。任务接触与支撑接触应区别处理，不能把所有手指吸向小按钮或扳机；保留任务方向、目标关节行程和执行可达性约束。生成阶段固定物体关节的结果，仅对应指定物体构型。

## 推荐实施与对比顺序

1. 在当前真实入口记录粗优化、几何修复、精修的时间及各阶段候选数。
2. 只加入粗到细候选筛选，保持其余设置不变。
3. 加入活跃批次压缩和间隔判定。
4. 对比固定手型的稀疏位姿匹配与当前粗优化。
5. 最后试验物理贴合，不同时替换全部阶段。

同时报告每秒有效任务候选数、部件接触覆盖率、多样性和独立功能成功率；初始化、资产/SDF 预处理与状态库构建成本应列出，复用成本与单次在线成本分开。不要只比较最终精修阶段耗时。

## 当前 Joystick baseline 的几何补全

旧匹配面 `assets/object/joystick_006/mesh/stick_link.obj` 不含活动按钮和扳机；完整 MuJoCo 碰撞模型含这些部件，因此此前扳机接触主要是生成后的附带结果。

新选项 `--surface complete` 使用 `stick_link` 及其子 body 的中立姿态视觉几何做布尔并集，输出 `mesh/neutral_complete.obj`；底座不纳入吸引表面，但保留在完整碰撞模型中。并集去除相交实体的内部面，不加入区域权重或手指指定。

该网格只是中立姿态的静态匹配代理，不替代完整 MJCF。移动关节后必须重建代理。OBJ 不保存部件标签，结果分析继续使用完整 MuJoCo 模型产生的 `ho_c.bn2`；未来弱区域引导需要独立的部件表面/区域数据，不能从并集 OBJ 假定标签仍然存在。

若补全后仍不能覆盖扳机，再对比全局/任务区域混合采样，以及小权重、允许范围内零惩罚的区域约束。原生版本与 `Dexonomy + region guidance` 分开报告，保持任务信息与执行评估条件一致。

### 补全几何版本的生成与查看命令

在 Dexonomy 根目录、`dexonomy` 环境中运行：

```bash
python assets/object/export_joystick_target.py --surface complete
python assets/object/prepare_joystick_scene.py --surface complete
dexrun hand=tianyi_right op=init exp_name=joystick006_complete tmpl_name=first_try 'init_gpu=[0]' n_worker=1 op.epoch=5 op.object.batch_size=1 op.object.n_cfg=1 op.filter.general.n_final=100 op.object.cfg_path=assets/object/joystick_006/scene_cfg/neutral_complete.npy
dexrun hand=tianyi_right op=grasp exp_name=joystick006_complete n_worker=1
python -m dexonomy.view_states --data output/joystick006_complete_tianyi_right/grasp_data --port 8083
```

打开 `http://127.0.0.1:8083`，用 `Contact body` 筛选 `obj-joystick_006trigger_00_link`。生成与查看均不调用 `op=eval`。已有实验可能因 `skip_done` 跳过处理；新一轮独立实验应在生成命令中使用一致的新 `exp_name`，并修改查看器的数据目录，不与旧结果混合。

直接查看匹配几何可用 CloudCompare 打开 `assets/object/joystick_006/mesh/neutral_complete.obj`。它仍在 stick_link 局部坐标系，与完整中立模型对照需要平移 `[0, 0.0368421, 0.142105]` 米。

本次接入测试：完整表面为 2353 顶点、4698 三角面，闭合且绕向一致。500 条初始化最终保存 195 个生成状态，其中 21 个记录到扳机近接触；旧杆体版本为 233 个生成状态，其中 1 个扳机近接触。两者仅为单轮观察，不是多随机种子结论，未执行功能评估。

新结果另有 10 个状态记录到底座近接触。底座参与碰撞避免不等于完全禁止近接触，原生筛选没有底座接触禁令；可在查看器筛选 `obj-joystick_006object_root` 检查，不应把这些状态自动认定为可用操作状态。

## 已实现：接触索引与可选区域优化

### 区域来源与坐标

`assets/object/import_joystick_regions.py` 读取 Intent 的 candidate_006 `link_sdf_all_2mm.npz` 中已投影的任务表面点云，包含 11 类任务。按中立 MJCF 正运动学转换到 stick_link 匹配坐标系，保存为 `assets/object/joystick_006/task_regions.npz`。不复制 SDF 网格，也不导入 Intent 的优化或执行策略。

```bash
python assets/object/import_joystick_regions.py --source /mnt/ssd/Bennkyou/PROJECT/InHandMani/intent/config/geometry/yaogan_curated_candidates_v6_upper_region_v1/candidate_006/link_sdf_all_2mm.npz
```

复用的是碰撞表面标注；抽样检查与当前碰撞网格最大距离约 0.045 mm，杆体与视觉表面有约 1.8 mm 的差异，源于凸碰撞近似。对象关节、根位姿、匹配坐标或缩放变化后不能直接复用该文件。区域加载时会校验这些场景条件。

### 原生结果的多任务接触索引

```bash
python -m dexonomy.label_states --data output/joystick006_complete_tianyi_right/grasp_data --regions assets/object/joystick_006/task_regions.npz
```

输出 `output/joystick006_complete_tianyi_right/contact_labels.json`，原状态 NPY 不变。`body_index` 和 `region_index` 的值均为相对于 `data_dir` 的状态路径列表，组合键按字母排序，并表示交集（允许额外接触）。默认近接触间隙 0~2 mm，任务区域点云距离不超过 4 mm；可通过 `--max-gap-mm`、`--max-penetration-mm`、`--region-tolerance-mm` 修改并重新生成索引。

`body_index["stick+trigger"]` 表示 body 级多部件候选，`stick` 包含头部和安装座。`region_index["move_stick_forward+pull_trigger"]` 要求实际接触分别落在对应 body 的 Intent 区域内。区域标签不测试法向力、操作方向、可达行程或功能成功；同一手指可能贡献多个接触。相同 body 上的不同方向区域不被单独列为跨部件多任务组合。`records` 保存接触位置、法向、手部 body、底座近接触及最小间隙。

读取示例：

```python
import json
from pathlib import Path
import numpy as np

index_path = Path("output/joystick006_complete_tianyi_right/contact_labels.json")
index = json.loads(index_path.read_text())
root = index_path.parent / index["data_dir"]
paths = index["region_index"].get("move_stick_forward+pull_trigger", [])
states = [np.load(root / p, allow_pickle=True).item() for p in paths]
```

### 初始化区域项（细化代码未修改）

`dexonomy/util/task_region.py::RegionLoss` 对每个请求任务选取 256 个确定性 FPS 区域点，计算所有模板接触点到区域点的最小距离 d。新增项为 `weight * 1000 * sum(relu(d - tolerance)^2)`；任务间求和，进入容差后为零，不指定手指，不要求不同任务使用不同手指。采用最近点分段可微距离，不包含任务法向或额外区域采样偏置。

`dexonomy/op/gen_init.py::HandObjMatcher.forward` 将它加入原生匹配目标；配置位于 `dexonomy/config/op/init.yaml::matcher.region`，默认 `weight=0`，不会读取区域文件，已做关闭时数值一致性测试。`n_points` 控制优化用区域离散精度，最终标签始终查询导入的完整区域点云。它们不是严格相同的距离离散化，不能把初始化距离直接当成最终实际接触。

单独扳机优化：

```bash
dexrun hand=tianyi_right op=init exp_name=joystick006_region_trigger tmpl_name=first_try 'init_gpu=[0]' n_worker=1 op.epoch=5 op.object.batch_size=1 op.object.n_cfg=1 op.filter.general.n_final=100 op.object.cfg_path=assets/object/joystick_006/scene_cfg/neutral_complete.npy op.matcher.region.path=assets/object/joystick_006/task_regions.npz 'op.matcher.region.tasks=[pull_trigger]' op.matcher.region.weight=1
dexrun hand=tianyi_right op=grasp exp_name=joystick006_region_trigger n_worker=1
python -m dexonomy.label_states --data output/joystick006_region_trigger_tianyi_right/grasp_data --regions assets/object/joystick_006/task_regions.npz
```

多任务优化：将初始化任务参数改成 `'op.matcher.region.tasks=[move_stick_forward,pull_trigger]'`，并统一将实验名改成 `joystick006_region_forward_trigger`。按钮任务可用 `press_button_01` 至 `press_button_05`；`press_button_center` 对应 `button_00_link`。可设置 `op.matcher.region.tolerance=0.004`，权重置零回退原生匹配。不同实验请使用不同名称，避免已有结果被跳过或混合。

NPY 中的 `region_guidance` 保存请求任务、权重、容差及初始化距离；最终 JSON 的 `all_requested_regions_contacted` 按真实近接触重新标注。不补救区域接触丢失，不修改 `gen_grasp.py` 的目标点、接触力或筛选器。

### 查看与本轮结果

```bash
python -m dexonomy.view_states --data output/joystick006_region_forward_trigger_tianyi_right/grasp_data --regions assets/object/joystick_006/task_regions.npz --port 8086
```

`Body combination` 筛选部件交集，`Required task regions` 复选框支持任意区域交集，`Exclude base contacts` 排除底座近接触；`Task region cloud` 显示所选区域。查看器实时计算标签，改变阈值不会自动覆写已保存 JSON，需要重新运行标签命令。

各版本均为 500 条初始化，以下仅是单轮观察，未运行功能评估：

| 版本 | 保存状态 | 扳机 body 接触 | 扳机区域接触 | 前推区域 + 扳机区域 | 底座近接触 |
|---|---:|---:|---:|---:|---:|
| 原生完整表面 | 195 | 21 | 20 | 7 | 10 |
| 扳机区域项 weight=1 | 273 | 238 | 227 | 101 | 70 |
| 前推 + 扳机区域项 weight=1 | 209 | 168 | 155 | 121 | 15 |

上述类别可重叠，统计未排除底座；双任务的 121 个区域交集候选排除底座近接触后为 120 个。接触覆盖改善不能替代功能评估，当前也不据此确定最优区域权重。
