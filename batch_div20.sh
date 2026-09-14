#!/bin/bash
# 多样性批测 20 组 — 多出发位置 × 多目标类别 (2026-09-13, 用户指令)
# 覆盖: 14/14 类目全覆盖; 近目标脚边局 3 (0.06/0.59/1.16m) + 中距 7 + 远距 10
# 约定: 批跑期间不改 src (每次 bm_run 全新 import, 中途改动会使组间代码态不一致)
PY=/home/tao_h/miniconda3/envs/vlm_nav/bin/python
cd /home/tao_h/VLMnav

# GRP  category episode_idx  (geod, spawn) — 选点依据见 任务/项目进展报告 v19
run () {  # $1=GRP $2=CAT $3=IDX $4=备注
  echo "=== [$1] $2 idx$3 ($4) $(date +%H:%M:%S) ==="
  $PY -u bm_run.py "$1" "$2" "$3" > "logs/verify_$1_20260913.log" 2>&1
  grep -oE '"finish_status": "[a-z_]+", "goal_reached": (true|false), "final_distance": [0-9.]+' "logs/verify_$1_20260913.log" | tail -1
}

run d01 coca_cola_glass_bottle 5 "9.19m,(-2.4,-0.1)"
run d02 honey_bunches_of_oats_with_almonds 5 "10.48m,(0.9,0.3)"
run d03 nutrigrain_harvest_blueberry_bliss 3 "11.14m,(-0.2,0.3)"
run d04 crystal_hot_sauce 4 "10.50m,(1.4,0.3)"
run d05 softsoap_white 4 "9.39m,(0.9,0.3)"
run d06 progresso_new_england_clam_chowder 3 "10.45m,(0.9,0.3)"
run d07 honey_bunches_of_oats_honey_roasted 2 "10.50m,(1.3,0.2)"
run d08 quaker_chewy_low_fat_chocolate_chunk 1 "10.50m,(1.3,0.2)"
run d09 nature_valley_sweet_and_salty_nut_almond 2 "11.32m,(1.0,0.3)"
run d10 mahatma_rice 2 "6.17m,(1.2,0.0) 新出发点"
run d11 aunt_jemima_original_syrup 2 "2.22m,(2.3,0.0)"
run d12 paper_plate 2 "4.47m,(0.1,0.1)"
run d13 spongebob_squarepants_fruit_snaks 1 "2.12m,(-2.1,0.0)"
run d14 bumblebee_albacore 0 "2.13m,(-2.2,0.0)"
run d15 coca_cola_glass_bottle 1 "0.06m 脚边局"
run d16 softsoap_white 2 "1.34m,(-1.4,-0.2)"
run d17 crystal_hot_sauce 2 "0.59m,(-2.0,-0.2)"
run d18 nutrigrain_harvest_blueberry_bliss 2 "1.16m,(-2.5,-0.2)"
run d19 mahatma_rice 1 "5.80m,(2.7,-0.1) 第三出发点"
run d20 bumblebee_albacore 2 "5.79m,(0.2,0.2)"

echo "=== ALL 20 DONE $(date +%H:%M:%S) ==="
