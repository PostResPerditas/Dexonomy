# used

## 准备

1. 准备目标部件网格
2. 建立中立状态场景
3. 先跑通原生状态生成
4. 导出统一操作状态

conda activate dexonomy
cd /mnt/ssd/Bennkyou/PROJECT/Dexonomy
root=output/joystick006_final_freeze_v1

python -m dexonomy.view_execution --data "$root/forward" --backend mujoco
python -m dexonomy.view_execution --data "$root/backward" --backend mujoco
python -m dexonomy.view_execution --data "$root/button02" --backend mujoco
python -m dexonomy.view_execution --data "$root/button05" --backend mujoco
python -m dexonomy.view_execution --data "$root/trigger" --backend mujoco
python -m dexonomy.view_execution --data "$root/forward_trigger" --backend mujoco
python -m dexonomy.view_execution --data "$root/backward_button05" --backend mujoco
python -m dexonomy.view_execution --data "$root/backward_button02" --backend mujoco

## 训练
conda activate dexonomy
cd /mnt/ssd/Bennkyou/PROJECT/Dexonomy
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TAG=joystick006_128_e20_v1

TASKS=(
  "forward|move_stick_forward"
  "backward|move_stick_backward"
  "button02|press_button_02"
  "button05|press_button_05"
  "trigger|pull_trigger"
  "forward_trigger|move_stick_forward,pull_trigger"
  "backward_button05|move_stick_backward,press_button_05"
)

for item in "${TASKS[@]}"; do
  IFS='|' read -r name tasks <<< "$item"
  IFS=',' read -r primary secondary <<< "$tasks"
  old="output/$TAG/$name/final_freeze_up_to128"
  out="output/$TAG/$name/final_freeze_fast_up_to128"

  if [[ -f "$old/summary.json" || -f "$out/summary.json" ]]; then
    echo "跳过已完成任务：$name"
    continue
  fi

  extra=()
  [[ -n "$secondary" ]] && extra+=(--then-task "$secondary")
  [[ "$name" == button* || "$name" == trigger ]] && extra+=(--fix-other-object-joints)
  [[ "$name" == trigger ]] && extra+=(--trigger-contact-linear)

  python -m dexonomy.execute_final_state_freeze_fast \
    --data "output/$TAG/$name/selected_up_to128" \
    --selection all --limit 0 --task "$primary" "${extra[@]}" \
    --action-mode auto --pre-mode local --physics-profile offline \
    --workers 6 --record-every 10 --output "$out" || break
done