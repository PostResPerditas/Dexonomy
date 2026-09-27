# Joystick006 离线操作评估与 Intent 接入说明

更新日期：2026-09-26。项目根目录：/mnt/ssd/Bennkyou/PROJECT/Dexonomy。

## 1. 适用范围与方法命名

本文记录当前 Tianyi right hand + joystick candidate_006 的执行配置与候选判据，供 Intent 对齐测试。最终主报告采用哪套配置与判据尚待用户根据不同方法的对比结果确定，不预先将 offline 质量门槛确定为唯一标准。它是低阻抗功能测试，不是对真实机构动力学、真实机器人可执行性或真实力控制的验证。

用户称呼的「Dexonomy + repre.」在本文具体指当前保存的区域引导候选：采用 Intent 的任务区域表示，在 Dexonomy 初始化匹配中加入 region guidance，然后沿用 Dexonomy 原生状态细化。保存状态的 region_guidance 指明 tasks、weight=1、tolerance_m=0.004。这不是不加任务信息的原生 Dexonomy，也不意味着已经独立评估了某个名为 repre 的模块。正式报告应给出这一操作性定义。

生成与执行完全分离。没有重新生成候选；本次补齐三个单任务在 offline 配置下的执行，多任务沿用同配置已完成测试。所有结果均可追溯至 summary.json 的 source 字段。

## 2. 当前统一配置结果

每项按任务区域标签交集筛选、排除底座近接触，再按路径字典序取前 10 个。不是随机抽样，不是全部候选测试。近接触标签不是功能成功标签。

| 任务 | 已生成状态 | 区域匹配且无基座标签候选 | 测试数 | task_success | physics_valid | 最终成功 | 成功率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lever forward | 217 | 171 | 10 | 10 | 10 | 10 | 100% |
| Button 5 | 212 | 166 | 10 | 5 | 10 | 5 | 50% |
| Trigger | 273 | 165 | 10 | 5 | 5 | 3 | 30% |
| Forward + trigger | 209 | 120 | 10 | 6 | 8 | 5 | 50% |
| Backward + button 2 | 191 | 32 | 10 | 1 | 7 | 1 | 10% |

task_success 与 physics_valid 是两个可重叠集合，最终成功取交集，不能取两个计数的最小值。

单任务合计 18/30 = 60%；双任务合计 6/20 = 30%；五项合计 24/50 = 48%。这些只适用于本次等量小样本组合，不是泛化成功率、全数据成功率或生成到执行端到端成功率。单 trigger 和双 trigger 来自不同候选池，不能将 30% 与 50% 解释成双任务更容易。每项只有 10 个样本，比例每个样本变化 10 个百分点。

最终失败原因：
- Forward：无。
- Button 5：3 个 motion_budget_exhausted，2 个 pretriggered。
- Trigger：5 个 invalid_physics，2 个 motion_budget_exhausted。
- Forward + trigger：2 个 invalid_physics，3 个 secondary_motion_budget_exhausted。
- Backward + button 2：3 个 invalid_physics，2 个 joint_hold_failed，3 个 pretriggered，1 个 secondary_motion_budget_exhausted。

invalid_physics 优先覆盖最终 reason，原任务结果保留在 task_reason。因此以上最终原因互斥，但物理异常和任务失败本身可能同时发生。

### 结果目录

所有路径相对于项目根目录，每个目录的 summary.json 是结果与实际命令参数的依据：

| 任务 | 离线结果目录 | 候选输入目录 |
| --- | --- | --- |
| Forward | output/joystick006_execution_a_offline | output/joystick006_task_a_forward_tianyi_right/grasp_data |
| Button 5 | output/joystick006_execution_b_offline | output/joystick006_task_b_button05_tianyi_right/grasp_data |
| Trigger | output/joystick006_execution_trigger_offline | output/joystick006_region_trigger_tianyi_right/grasp_data |
| Forward + trigger | output/joystick006_execution_c_offline | output/joystick006_task_c_forward_trigger_tianyi_right/grasp_data |
| Backward + button 2 | output/joystick006_execution_d_offline | output/joystick006_task_d_backward_button02_tianyi_right/grasp_data |

### 历史结果：不可混入当前结果表

旧 native 配置没有最终穿透/超程质量门槛，步长 4 ms，每 5 步更新控制：
- 前推 TCP：10/10，output/joystick006_execution_a_single。
- Button 5 TCP：5/10，output/joystick006_execution_b_single。
- Trigger TCP：0/10，output/joystick006_execution_trigger_single，TCP 圆弧预算 15 度。
- Button 5 全指闭合：8/10，output/joystick006_execution_button05_fingers。
- Trigger 全指闭合：9/10，output/joystick006_execution_trigger_fingers。
- Forward + trigger：4/10，output/joystick006_execution_c_dual。
- Backward + button 2：7/10，output/joystick006_execution_d_dual。

旧配置与新配置同时改变了离散化、接触/限位求解和判据，不能将比例变化单独归因于某一个参数。

### 三种统计口径与配置差异（其他窗口调用必读）

仿真中的穿透可能来自接触模型、控制推进和求解参数，也可能与生成状态有关，不能直接判定真实操作必然失败；穿透后的行程达标也不能证明真实可执行。保留以下三列，不以其中任一列替代其他列。

| 任务 | A：native 行程保持 | B：offline 行程保持 | C：offline 行程保持 + 质量检查 |
| --- | ---: | ---: | ---: |
| Lever forward | 10/10，100% | 10/10，100% | 10/10，100% |
| Button 5 | 8/10，80% | 5/10，50% | 5/10，50% |
| Trigger | 9/10，90% | 5/10，50% | 3/10，30% |
| Forward + trigger | 4/10，40% | 6/10，60% | 5/10，50% |
| Backward + button 2 | 7/10，70% | 1/10，10% | 1/10，10% |
| 单任务合计 | 27/30，90% | 20/30，66.67% | 18/30，60% |
| 双任务合计 | 11/20，55% | 7/20，35% | 6/20，30% |
| 全部合计 | 38/50，76% | 27/50，54% | 24/50，48% |

A 使用前推 a_single、按钮 button05_fingers、扳机 trigger_fingers、双任务 c_dual/d_dual 目录。排除早期按钮 b_single 和扳机 trigger_single 的 TCP-only 实验，因为其执行策略不同。

| 口径 | 仿真调用 | 逐候选统计字段 | 含义 |
| --- | --- | --- | --- |
| A | --physics-profile native --timestep 0.004 --substeps 5 | results[i].success | 原控制离散化和接触参数下的行程保持成功，不额外排除穿透/超程 |
| B | --physics-profile offline | results[i].task_success | 新仿真轨迹上的任务成功，质量异常不在本列额外剔除 |
| C | 与 B 同一次运行 | results[i].success | task_success AND physics_valid |

三个口径都排除接近/稳定阶段的提前触发，默认不要求持续接触。B 也不是只看最大位移：仍需遵循阶段顺序、保持判据和仿真运行条件。physics_valid 是独立数值诊断，不是任务成功，也不是物理真实性证明。

A 与 B 的区别是仿真配置改变，需要分别运行；不仅质量门槛不同。B 与 C 来自同一组 offline 轨迹，可以直接切换汇总字段，无需重新仿真，也不要用增大质量容差的方式伪装旧配置。两个物理配置默认执行策略一致：lever 使用 TCP 圆弧，button/trigger 使用全指闭合，双任务先 lever 后全指闭合。具体时长与阈值见第 5、6 节。这里 native 是执行器的配置名，不代表整个区域引导 baseline 等同于原生 Dexonomy 方法。

对比时先统一候选与 pre-contact，再分别运行 A、B/C。例如同一批 forward + trigger 候选：

```bash
cd /mnt/ssd/Bennkyou/PROJECT/Dexonomy
conda activate dexonomy

python -m dexonomy.execute_states \
  --data output/joystick006_task_c_forward_trigger_tianyi_right/grasp_data \
  --task move_stick_forward --then-task pull_trigger \
  --action-mode auto --pre-mode native --limit 10 \
  --physics-profile native --timestep 0.004 --substeps 5 \
  --output output/compare_c_native

python -m dexonomy.execute_states \
  --data output/joystick006_task_c_forward_trigger_tianyi_right/grasp_data \
  --task move_stick_forward --then-task pull_trigger \
  --action-mode auto --pre-mode native --limit 10 \
  --physics-profile offline \
  --output output/compare_c_offline
```

输出目录须未使用。Intent 替换 --data 并按第 7 节对齐输入，不更改任务执行策略。单任务删除 --then-task；button/trigger 保持 --action-mode auto，不传 --action-mode tcp。相同数据文件夹与标签必须保持不变，比较前核对两个 summary.json 的 results[*].source 列表一致。

从已有 offline 结果一次读取 B/C 的例子：

```python
import json
from pathlib import Path

folder = Path('output/joystick006_execution_c_offline')
summary = json.loads((folder / 'summary.json').read_text())
assert summary['settings']['physics_profile'] == 'offline'
rows = summary['results']
n = len(rows)  # 包含失败、pretriggered 和 invalid_physics，不删除这些样本
for label, field in [('B_task_only', 'task_success'),
                     ('C_task_and_quality', 'success')]:
    count = sum(bool(row[field]) for row in rows)
    print(label, count, n, f'{100 * count / n:.2f}%')
print('physics_valid_count', sum(bool(row['physics_valid']) for row in rows))
```

native 历史结果没有 task_success/physics_valid 字段，A 直接读 success；不要把缺失的 physics_valid 当 True，也不要从旧 native 的低频记录宣称完成了逐物理步质量检查。offline 的顶层 summary.success 是 C 的计数，不是 B；统计失败原因时 B 读 task_reason，C 读 reason。

--require-contact-hold 是另一个可选协议，不属于上述三列。它影响保持阶段和是否进入第二任务，启用后应为所有方法重新运行；不能仅用已有轨迹的末帧接触布尔值替代。调整保持时间、触发顺序、动作范围或预接触也可能改变控制流程，需要重跑。仅对同一保存轨迹改穿透/超程门槛可以后处理，但必须标注新门槛、重新检查完整轨迹且不覆盖原始判定。

回放的 --result success 按 C 筛选 offline 结果，不会显示 B 中 task_success=True、physics_valid=False 的轨迹。分析这类差异时使用默认 --result all，查看面板的 Task criterion 与 Physics valid，或定位 summary.json 的对应 trajectory。回放不决定 A/B/C 的统计。

比较不同方法时使用同一物理配置、控制策略、判据和统计分母。当前每项仅 10 个固定排序筛选候选，结果对配置敏感，应保留三列及失败原因；不根据哪个口径更有利于某个方法而选择主指标。最终采用的协议由用户后续确定，本次仅补充说明，不修改执行器或已有结果。

## 3. 实现与资产

- dexonomy/execute_states.py：候选筛选、pre-contact、单/双任务控制、offline 参数与判定。
- dexonomy/sim/mujoco_env.py：MuJoCo_BaseEnv、MuJoCo_EvalEnv、模型装配、mocap 和驱动映射、原生摩擦设置。
- dexonomy/view_execution.py：只回放已有轨迹，不重新仿真、不重新判定。
- dexonomy/label_states.py：生成候选的任务区域近接触标签。
- dexonomy/util/task_region.py：区域引导相关逻辑。
- assets/hand/tianyi_right/right.xml：手部模型、驱动与 mimic。
- assets/object/joystick_006/model.xml：完整可动物体碰撞模型，包括基座、杆体、按钮和扳机。
- assets/object/joystick_006/scene_cfg/neutral_complete.npy：评估场景入口。
- assets/object/joystick_006/task_regions.npz：任务区域点云。

MuJoCo 版本为 3.3.3。每个结果目录的 model.mjb 是实际编译并应用 offline 参数后的模型，比仅查看源 XML 更可靠。原生 Dexonomy 会在装配时覆盖部分 XML 属性，不能只复制物体 XML 的 option。

物体根固定，世界位姿为 xyz=(0,0,0)、wxyz=(1,0,0,0)，scale=1，9 个物体关节初始值均为 0。生成阶段 fix_joints_for_generation=True 不代表执行阶段也固定：执行模型恢复全部 9 个被动关节。

stick_link 匹配网格的局部位姿为 xyz=(0,0.0368421,0.142105)、单位旋转，不能将匹配网格坐标直接当物体根/世界坐标。Intent 应对手与物体整体使用一致坐标转换，避免重复施加历史模型偏移。

## 4. 固定的 offline 配置

必须显式传 --physics-profile offline；CLI 默认仍为 native。offline 会覆盖 --timestep 和 --substeps 为下表值。

| 参数 | 值 |
| --- | --- |
| timestep / 控制步 / 保存步 | 0.0005 s / 每物理步 / 每物理步 |
| substeps | 1 |
| integrator / solver | implicitfast / Newton |
| iterations / ls_iterations | 200 / 50 |
| tolerance / ls_tolerance | 1e-10 / 0.01 |
| cone / condim | elliptic / 4 |
| impratio | 10 |
| noslip_iterations / tolerance | 10 / 1e-6 |
| 碰撞 geom solref | [0.004, 1] |
| 碰撞 geom solimp | [0.99, 0.999, 0.0001, 0.5, 2] |
| 全部关节限位 solref / solimp | 同上 |
| 摩擦前两项 | [0.6, 0.02] |
| 第三摩擦项 | 保留模型值：手部 0.0001、物体 0.001；condim=4 不启用滚动摩擦自由度 |
| 重力 | mjDSBL_GRAVITY 禁用；编译模型 gravity 向量仍可能是 [0,0,-9.81] |
| 物体复位弹簧 / 外加操作力 | 无 / 无 |

不降低物体碰撞刚度来实现低阻抗。低阻抗指物体关节易动；接触约束仍需避免穿透。

物体关节 damping：tilt_x/y 为 0.08，其余为 0.05；frictionloss：tilt_x/y 为 0.01，其余为 0.005；stiffness 全为 0。保留质量、惯量、碰撞几何、margin 和 exclusions，不为提高通过率更换这些属性。

手部 6 个 position actuator：kp=5，forcerange=[-3,3]；12 个手指关节 damping=0.1。手根由 mocap weld 驱动，不是直接将真实手根 qpos 逐步写成目标位姿。mocap weld 的 solref=[0.02,1]、solimp=[0.9,0.95,0.001,0.5,2]；6 个 mimic 约束的 solref=[0.004,1]、solimp=[0.99,0.999,0.001,0.5,2] 不随 offline 接触设置覆盖。保留原生装配设置，包括手指 joint margin=0.1。

## 5. 执行过程

每个候选重新初始化手和中立物体，仅此时设置初始 qpos。随后物体只能由动力学与接触运动，不施加生成阶段的虚拟吸引力、不沿目标物体 qpos 直接播放。

1. 接近：pre-contact 路径到 grasp_qpos，总计 1.5 s；根姿态使用四元数插值，手指线性插值。
2. 稳定：0.3 s；squeeze_blend=0，不在此阶段额外 squeeze。
3. 单任务动作预算 2 s。Lever 用动作开始时真实世界关节轴/轴心的 TCP 圆弧，最大 25 度，手指指令固定。Button/trigger 固定 TCP 指令，全部 6 个驱动器同时向各自闭合上限渐进运动，close_fraction=1；不按初始距离挑选手指。mimic 由仿真约束联动。
4. 到达物体实际关节阈值就停止增加控制目标，最多等待 0.5 s，要求连续 0.2 s 达标。停止增加指令不等于物体立即停下，惯性与接触可能继续造成位移。
5. 双任务先执行 lever 并通过第一阶段保持，然后固定此时 TCP 指令，执行 2 s 全指闭合；最终要求两个物体任务同时达标并连续保持 0.2 s。第一阶段失败则不执行第二阶段。不是两个 TCP 动作叠加。

固定 TCP 指的是固定 mocap 目标；实际手根允许通过 weld 顺应偏移。默认没有持续接触要求。按动后脱离但物体仍保持位移的样本可能通过，这是当前明确的判据，而非持续力控制证明。

## 6. 阈值与最终判据

| 任务 | 关节 / 方向 | 中立到限位 | 半行程阈值 |
| --- | --- | --- | --- |
| Forward | joystick_tilt_x / 正向 | 0.55 rad | 0.275 rad，约 15.756 度 |
| Backward | joystick_tilt_x / 负向 | 0.55 rad | 0.275 rad，按负向进展计 |
| Button 5 | button_05_joint / 正向 | 0.003 m | 0.0015 m |
| Button 2 | button_02_joint / 正向 | 0.004 m | 0.002 m |
| Trigger | trigger_00_joint / 正向 | 0.35 rad | 0.175 rad，约 10.027 度 |

task_success：实际关节沿目标方向从初值的位移达到 success_fraction=0.5，并通过上述保持。接近/稳定期间任一请求任务达标则 pretriggered，不计成功。双任务第二项在 lever 阶段已达标不自动失败，记录 secondary.already_at_threshold；因此当前双任务协议不保证严格按顺序首次触发。

physics_valid：全过程包含 reset、approach、settle、action、hold：
- 无非有限 qpos/qvel、时间倒退/重置或 MuJoCo warning。
- 最大手与非手几何接触穿透 <= 0.002 m。
- 所有有限位物体关节的超程：slide <= 0.0002 m，hinge <= 0.02 rad，包含非任务按钮。
- 不合格统一 reason=invalid_physics；保留 task_reason、max_penetration、overtravel。

最终 success = task_success AND physics_valid。上述穿透是碰撞几何接触距离，不是视觉网格精确相交体积；没有全面检查手自碰撞、物体自碰撞或所有手部关节超程。2 mm、0.2 mm、0.02 rad 是当前人为协议阈值，不代表证明物理真实性。未进行步长/求解收敛研究。

base_contact_any 当前仅记录，不自动判失败；生成候选中的「无基座近接触」也不能替代执行全过程的基座检测。--require-contact-hold 可要求保持期接触，但本表全部未开启。开启后须为所有方法重测，不能和本表直接混用。

## 7. Intent 状态接口与公平比较

建议首先复用本项目执行器，只实现 Intent 状态导出；不要先各写一套控制器再直接比较成功率。

单状态 .npy 为 dict，至少包含 grasp_qpos，shape=(19,) 或 (1,19)，float：
- 前 3 维：手自由根世界 xyz，单位 m。
- 第 4-7 维：手自由根世界四元数 wxyz，归一化。
- 后 12 维：thumb_1、thumb_2、thumb_3、thumb_4、index_1、index_2、middle_1、middle_2、ring_1、ring_2、little_1、little_2，单位 rad。
- 使用当前 XML 的 free root 框架，不直接拿 Intent 的掌心位姿代替；应通过固定变换转换。当前 TCP 与 free root 重合，R_base_link 不是无旋转偏移的同一框架。
- 可含 scene_path；缺省时通过 --scene 指定上面的 neutral_complete.npy。
- 可含 pregrasp_qpos，shape=(N,19)，按接近顺序排列；程序最后追加 grasp_qpos。

pre-mode 支持：
- native：读 pregrasp_qpos，本表所有试验采用它。
- specified：--pre-state 外部 NPY；可以是 Nx19 数组，或 dict，--pre-key 指定键。
- retreat：从 grasp 根位置沿 --retreat-direction 后退 --pre-clearance（默认 0.045 m），再插值接近；无路径规划保证。没有 ho_c 的 Intent 状态必须明确给 retreat-direction。
- generate：用 Dexonomy margin 扩张生成 pre-contact，属于额外预处理成本，不应免费归入 Intent 状态质量。

没有 contact_labels.json 时，单状态文件可直接输入；文件夹必须 --selection all。同样比较时应统一候选预算、筛选信息与 pre-contact 策略，或明确分别报告差异。本表是 native-pre 下条件执行率；若 Intent 用 retreat，不应称已完全控制变量。推荐后续让两个方法都再跑同一个 pre-mode。

Intent 单状态调用示例（替换实际输入路径，不依赖默认区域标签）：

```bash
cd /mnt/ssd/Bennkyou/PROJECT/Dexonomy
conda activate dexonomy
python -m dexonomy.execute_states \
  --data /path/to/intent_state.npy \
  --scene assets/object/joystick_006/scene_cfg/neutral_complete.npy \
  --task move_stick_forward --then-task pull_trigger \
  --pre-mode specified --pre-state /path/to/intent_pre.npy \
  --physics-profile offline --output output/intent_forward_trigger_offline
```

评估其他单任务时删除 --then-task，替换 --task。输出必须用新目录。若 Intent 独立移植，需对齐模型、坐标、初态、预接触、控制顺序、驱动增益/力限、全部物理设置和判据；仅对齐 timestep 不足以保证公平。

## 8. 复测与回放命令

以下任务参数对应结果表；--output 换成未使用目录即可重跑。

```bash
python -m dexonomy.execute_states --data output/joystick006_task_a_forward_tianyi_right/grasp_data --task move_stick_forward --physics-profile offline --limit 10 --output output/retest_a_offline
python -m dexonomy.execute_states --data output/joystick006_task_b_button05_tianyi_right/grasp_data --task press_button_05 --physics-profile offline --limit 10 --output output/retest_b_offline
python -m dexonomy.execute_states --data output/joystick006_region_trigger_tianyi_right/grasp_data --task pull_trigger --physics-profile offline --limit 10 --output output/retest_trigger_offline
python -m dexonomy.execute_states --data output/joystick006_task_c_forward_trigger_tianyi_right/grasp_data --task move_stick_forward --then-task pull_trigger --physics-profile offline --limit 10 --output output/retest_c_offline
python -m dexonomy.execute_states --data output/joystick006_task_d_backward_button02_tianyi_right/grasp_data --task move_stick_backward --then-task press_button_02 --physics-profile offline --limit 10 --output output/retest_d_offline
```

输出：summary.json 保存设置与逐候选判定；model.mjb 保存实际模型；各 NPZ 保存逐物理步 time、phase、qpos/qvel、ctrl、mocap、command、progress、contact、metadata，双任务另含 secondary_progress/secondary_contact。

```bash
python -m dexonomy.view_execution --data output/joystick006_execution_c_offline --backend mujoco --sample 1
python -m dexonomy.view_execution --data output/joystick006_execution_d_offline --backend viser --port 8101
```

回放按约 60 Hz 显示采样，不改变物理记录或成功结果；逐帧仍可访问保存的物理步。MuJoCo 配色：手 #2C2BB8，杆体/平台/基座 #E9E9E9，按钮/扳机 #D56059，白背景。绘制效果不参与判定。

本文为当前配置与结果的交接入口。getting_started/intent_optimization_from_dexonomy.md 中早期「仅单任务、TCP 按钮」段落属于历史阶段，不应作为当前执行配置依据。
