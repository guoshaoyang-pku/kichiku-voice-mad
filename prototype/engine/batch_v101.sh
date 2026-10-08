#!/bin/bash
# v10.1 批处理：old3 重出 joint（play-full），new5 全流程 tune → joint
# 8 首歌 × 8 卡并行，日志在 out/batch_logs/
cd /data2/guoshaoyang/kichiku-voice-mad/prototype
PY=/data2/guoshaoyang/kichiku-venv/bin/python
mkdir -p out/batch_logs

run_old() {  # $1=name $2=gpu
  CUDA_VISIBLE_DEVICES=$2 $PY -u engine/auto_mad.py accomp --name $1 --skip-sep \
    > out/batch_logs/$1.accomp.log 2>&1
  echo "OLD $1 rc=$?"
}
run_new() {  # $1=name $2=gpu
  CUDA_VISIBLE_DEVICES=$2 $PY -u engine/auto_mad.py tune --name $1 \
    > out/batch_logs/$1.tune.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=$2 $PY -u engine/auto_mad.py accomp --name $1 --skip-sep \
    > out/batch_logs/$1.accomp.log 2>&1
  echo "NEW $1 rc=$?"
}

run_old haruhikage 0 &
run_old roundabout 1 &
run_old sonochi 2 &
run_new avemujica 3 &
run_new killkiss 4 &
run_new requiem 5 &
run_new bloodystream 6 &
run_new cnbt 7 &
wait
echo BATCH_DONE
