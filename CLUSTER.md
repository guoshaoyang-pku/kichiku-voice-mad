# 在 A100 集群上跑渲染（runbook）

落点：`a100_perm`（8×A100-40GB，root=/data2/guoshaoyang/kichiku-voice-mad）。
venv：/data2/guoshaoyang/kichiku-venv（torch 2.5.1+cu121）。

## 一次性准备

```bash
# 1. 代码 + 素材（~4G：lib/samples、anime 干声、contour_cache、歌曲+stems）
rsync -az --exclude='__pycache__' prototype/engine prototype/lib prototype/materials \
      prototype/make_review.py a100_perm:/data2/guoshaoyang/kichiku-voice-mad/prototype/
# 2. 关键：lib/*.json 里是本机绝对路径，必须重写
ssh a100_perm 'cd /data2/guoshaoyang/kichiku-voice-mad/prototype && \
  kichiku-venv路径/bin/python engine/relocate_lib.py \
  /Users/guoshaoyang/Desktop/workdir/Ideas/kichiku-voice-mad /data2/guoshaoyang/kichiku-voice-mad'
# 3. venv：torch 必须 cu121 且 torchaudio==2.5.1（更高版本要 CUDA13 会炸 libcudart.so.13）
pip install torch torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 跑法（每卡一首歌）

```bash
ssh a100_perm 'cd /data2/guoshaoyang/kichiku-voice-mad/prototype && \
  nohup env CUDA_VISIBLE_DEVICES=0 kichiku-venv/bin/python engine/token_dp.py \
    --mix materials/haruhikage_original.wav --stems materials/stems_ft/htdemucs_ft/haruhikage_original \
    --out out/v11_haruhikage_genshin --version v11 --variant genshin_lisa \
    --ref-name v9_haruhikage --singer 丽莎 --lib lib/library_genshin_full.json \
    --keyframe-hard --choke --w-onset 2 --c-skip 4 --w-keyalign=3 --w-pitch=2.5 \
    --topk 128 --lambdas 2.5 --shifts=-4,-3,-2,-1,0 --render 2.5 \
    > out/v11_genshin.log 2>&1 &'
# 拉回结果
rsync -az 'a100_perm:/data2/guoshaoyang/kichiku-voice-mad/prototype/out/v11_*' prototype/out/
```

## 实测（2026-10-08，haruhikage 60s 窗口，topk128）

- 本地 M 系列：~90s（topk32 口径）；A100 暖缓存：47s（topk128）。
- 瓶颈是 CPU 段（DP 求解、onset 检测），GPU 只吃 CREPE + 候选打分，
  所以加速 ~2-4×，不是数量级；真正价值在多卡并行 + 大码本（原神 5041 条一次跑）。
- contour_cache / clip_onsets 缓存可移植，首次跑新码本的轮廓提取 A100 上几分钟。

## 注意

- `library_*.json` 和 `hires_map.json` 含绝对路径：再同步 lib/ 后要重跑 relocate_lib.py。
- 渲染产物（wav/mp3）和素材不入 Git，只在集群/本地之间 rsync。
