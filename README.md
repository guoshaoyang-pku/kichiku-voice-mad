# 鬼畜调音自动化（Kichiku Voice MAD）

Render a song's vocal melody with an anime character's spoken lines — no vocoder, no pitch correction of the voice clips. 用动漫角色的台词原声当"码本"，靠搜索把旋律"选"出来，而不是"修"出来。

**Current engine (v8/v9)**: keyframe-constrained segment Viterbi DP — unit selection over raw clips ("unit-selection VQ"); v9 adds `auto_mad.py` 全自动管线（下载分离 → 码本变体 × λ_N 自动调参 → 渲染），配歌曲只需一条命令。当前目标曲《春日影》(MyGO!!!!!)、《ジョジョ~その血の運命~》、《Roundabout》，码本为长崎素世台词（低音区借椎名立希/八幡海铃/若叶睦）。

## How it works (v7/v8 formulation)

- **码本 C** = 台词原声切片（不变调、不变速、不过声码器；仅允许：常数增益 + 5ms 淡入淡出 + 音节边界裁剪 ≥70% + 低音句有限重采样降调 + 关键帧掐断）。
- **token** z_i = (选哪句台词 k, 起止时间 τ, 裁剪 [a,b], 常数增益 g)，y = Σ g_i·c_k[a:b] 直接叠加。
- **目标** = 原曲人声的帧级 F0/音量/有声轮廓（CREPE on Demucs vocal stem），滑音处自动降权重。
- **损失** = 音高（主）+ 音量包络 + 有声/无声 + λ_N·N（少即是多）+ 裁剪代价 + WavLM 元音匹配（可选）。
- **求解** = 分段 Viterbi DP，逐帧损失可加 ⇒ 全局精确最优（Hunt & Black 1996 unit selection 同构），不需要 MCTS。
- **v8 关键帧**：原曲人声起音 + 音高跳变点作为硬约束，每个音效的第一个强起音必须卡上（±10ms），密集段允许后音效掐断前一个。

实测（整首 4:20，λ_N=10）：音高 ±50 音分 60%、±100 85%；起音 ±30ms 内 46%（v7 为 19%）；低音区覆盖 82%；285 个音效、115 句不同台词。听感见仁见智，指标只负责兜底。

## Why not X

- **声码器（WORLD/PSOLA/RVC）**：v3–v5 试过，高频气声被抹掉，"听不出是素世"，弃。
- **MIDI 谱当目标**：点状音符 ⇒ 占空比错误的破碎音乐；必须对原曲人声轮廓优化。
- **VQ-VAE**：本方案即"不训练的 VQ"——码本固定为真实台词（要的就是听得出台词），恒等解码器，编码器 = 搜索。值得借的只有特征空间（WavLM 替 MFCC，kNN-VC 思路）。

## Repo layout

```
prototype/engine/token_dp.py   # 当前引擎（v8-v10），其他 *.py 为 v3-v7 演进与诊断工具
prototype/engine/auto_mad.py   # v9 自动调参 + v10 带伴奏拟合（accomp 子命令）
CLUSTER.md                     # A100 集群多卡运行 runbook
prototype/lib/                 # 素材库构建与索引（音频 samples/ 不在仓库内）
prototype/legacy/              # v1-v7 旧引擎与旧指标，仅存档
prototype/make_review.py       # 渲染结果 → review.html 验收页
scripts/                       # 素材下载（原神/星铁全量语音、番剧干声搜索）
materials/                     # 【不在仓库】原始素材 134GB，见下方素材章节重建
```

**音频不入库**：原曲、台词、渲染产物均有版权问题，仓库只含代码、元数据 JSON、指标与诊断图。

## Quickstart

Linux/CUDA 也可运行（见 CLUSTER.md）。

```bash
./setup.sh                    # venv + 依赖（macOS Apple Silicon 验证，MPS 加速）
# 全自动管线（v9）：下载/分离 + 码本变体 × λ_N 自动调参 + 渲染，一条命令
python3 prototype/engine/auto_mad.py prepare --name mysong --src "ジョジョ その血の運命 OP"
python3 prototype/engine/auto_mad.py tune --name mysong        # → out/v9_* + 调参报告
# 手动单渲染（引擎全参数）：
python3 prototype/engine/token_dp.py --song prototype/materials/<song>.wav ...
# 生成验收页：
python3 prototype/make_review.py   # → prototype/review.html
```

---

## 背景与调研（立项笔记）

想法：用一首钢琴曲的谱子（MIDI）驱动二次元角色语音素材，自动生成"音MAD/鬼畜"风格的曲子。
不只做单音映射：一个音效应可覆盖多个音高（采样器 key-range），和弦可多素材叠加，
音量/力度要与曲子演绎匹配（velocity → 增益 + 素材选择）。

## 原理确认

经典鬼畜调音确实是「曲子拆成音符 → 每个音符映射到一段素材语音 → 变调到目标音高」。两条主流路线：

1. **手工 DAW 流**（Vegas/REAPER/FL + Melodyne）：素材切片按网格摆放，逐条修音高。细节最好，但全手工。
2. **采样器流**（UTAU / SoundFont / sampler）：把素材做成音源库（每个键位映射一个语音样本），
   直接用 MIDI 演奏。这条路线天然适合自动化，就是本项目要做的。

手工细节的核心经验（来自音MAD教程，后面自动化工具要复刻）：
- 对齐以**元音**为准，辅音（さ行/た行等）允许提前出网格，不能削掉辅音；
- 音符时长不足时对素材做 time-stretch，过长时循环元音段或截断+淡出；
- 变调过大会有"花栗鼠/怪兽音"——经典鬼畜反而故意用重采样变调（音高时长一起变），
  要保真就用相位声码器 pitch-shift（保时长）或神经网络变调（RVC/DDSP，保音色）；
- 力度匹配：MIDI velocity → 增益包络；更好的做法是 **velocity layers**，
  强音用喊叫版台词、弱音用轻声版台词，而不只是调音量。

## 自动化流水线设计（Python）

1. **谱子 → 音符表**：已有 MIDI 直接用 `pretty_midi` 解析（音高/起止时间/velocity）；
   若只有 PDF 谱 → OMR 工具；若只有音频 → Basic Pitch / Demucs 转写。
2. **素材库构建**：
   - 下载/抓取角色语音（来源见下）；
   - 带 BGM 的用 Demucs/UVR 分离人声；
   - 按静音自动切片（librosa/pydub），得到短句样本；
   - 每条样本分析：自然音高（CREPE/pyin）、RMS 响度、时长、元音/喊叫类型标签。
3. **映射规划**：每个 MIDI 音符 → 选样本 + 变调半音数。约束：变调幅度尽量小（±7 半音内最自然）、
   一个样本可覆盖一段键位区间（key-range，即"一个音效覆盖多个音"）、和弦音符分配不同样本叠加。
   这一步可以用规则或让 LLM/优化器按"台词可读性 + 音域覆盖 + 力度匹配"打分来排。
4. **渲染**：
   - 方案 A：Python 直接合成（`librosa.effects.pitch_shift/time_stretch` + 按 onset 叠加混音）→ 出 WAV；
   - 方案 B：生成 SF2 SoundFont + 用 fluidsynth 演奏 MIDI（velocity layers 原生支持）；
   - 方案 C：导出 REAPER/DAW 工程文件，留给人工精修。
   建议 A 快速出 demo，B/C 出成品。
5. **动态处理**：velocity → gain + 样本层选择；整体 loudness normalize；可选 CC1 表情曲线。

## 素材库调研（2026-10-05）

**可直接批量下载的（游戏语音，社区已扒好）：**
- 原神全角色语音：GitHub `CSUSTers/mys-voice-genshin`（3178 个文件，按角色分目录，
  抓自米游社观测枢 wiki）；另有 B 站专栏 cv23965717「原神语音包 6.3 中日英韩持续更新」。
- 明日方舟全干员语音：GitHub `isHarryh/Ark-Voice`（OGG + `voice_data.json` 带时间戳索引，
  适合自动切片）。
- 蔚蓝档案：`bluearchive.wiki` Category:Characters audio；GameKee 碧蓝档案 wiki 全角色日语语音+翻译。
- BWIKI 原神 wiki「角色语音」页：逐角色在线音频，可脚本抓取。

**番剧台词（少女乐队/热门动漫）：**
- 没有官方语音包下载，通行做法是从正片提取：yt-dlp 拉片源 → Demucs 分离人声 → 切片。
- B 站搜「鬼畜素材库 / 素材包 配布」有 UP 主整理的无水印素材合集（专栏/网盘分发）；
  UTAU 式鬼畜音源配布列表见 B 站专栏 cv363982。
- niconico Commons（commons.nicovideo.jp）：官方授权免费素材，但多为泛用音声，番剧角色台词少。
- 受众友好度高的近作（单作品或少量作品集中用）：孤独摇滚、吉伊卡哇、葬送的芙莉莲、
  我推的孩子、赛马娘、Love Live / 轻音少女（少女乐队向经典）。

**版权注意**：米哈游/鹰角有二创指引，允许非商用同人创作；番剧音频属灰色地带，
B 站非商用鬼畜是长期被容忍的惯例——不接商单、不盈利发布、标注素材来源即可。

## 素材库状态（2026-10-06 更新）

已入库（materials/）：
- `anime/`：三部番剧干声共 24 个 wav、约 4.5 小时（MyGO 151min / Ave Mujica 105min / 孤独摇滚 15min），
  明细见 `anime/README.md` 与 `anime/inventory.json`。均为社区已分离/降噪的纯人声，可直接切片使用。
- `genshin/mys-voice-genshin/`：米游社 wiki 全量抓取（2022 快照，61 角色 8267 条 mp3/ogg，
  文本映射在 res/csv）。
- `genshin/voice/`：AI-Hobbyist 角色包 5 个（七七/丝柯克/久岐忍/丽莎/九条裟罗，约 3000 wav + .lab 文本）。
- `starrail/voice/Archer`：测试包（29 wav）。

完整包（ModelScope 下载，已完成入库）：
- `starrail/full_cn/`：星铁 4.2 中文全量语音，75,329 wav + .lab 文本，1617 个说话人目录（含 NPC）。
- `genshin/full_cn/`：原神 7.0 中文全量语音，193,353 wav + .lab 文本，3764 个说话人目录，92GB。
- 断点续传经验：pan.acgnai.top 直链慢（0.14MB/s），ModelScope CDN 快（~20MB/s）。

放弃的源：
- AI-Hobbyist 角色包直链（pan.acgnai.top）实测仅 0.14MB/s，206 个包需数天，不可行；
  脚本与接口保留在 `scripts/download_acgnai_packs.py`。
- HuggingFace（simon3000/starrail-voice，40 万条 wav）本机网络间歇性不可达。

番剧素材缺口：
- 孤独摇滚只有波奇的干声（15min），虹夏/凉/喜多无现成包 → 后续用正片 + Demucs 分离补齐。
- Our Notes 手游全语音（MyGO 288m / AveMujica 364m）体量大且混角色，暂未下载。

## Prototype 早期状态（2026-10-06，已被上方 v8 取代）

`prototype/` 下已搭建 MIDI→语音样本→渲染的自动化管线并迭代 v1-v6：
目标曲为 JoJo 经典（Il Vento D'oro / Roundabout，MIDI 来自 bitmidi），
样本库 = 原神 13 角色 4608 条 + 番剧干声 7280 条（CREPE 音高/响度/稳定性分析 + 元音核提取）。
验收页 `prototype/review.html`，报告 `prototype/REPORT.md`，渲染音频在 `prototype/out/`。

## 待定问题

- [ ] 用户手里的钢琴曲是什么（谱子格式：MIDI/PDF/音频）？
- [ ] 素材选哪部作品（决定素材获取路径：游戏=直接下载，番剧=扒片+分离）？
- [ ] 输出目标：纯音频 demo，还是要配画面的成品视频？
- [ ] 变调风格：经典重采样"电音感" vs 保时长/保音色的自然变调？

## 参考

- 音MAD 制作方法（REAPER 教程站）：ytpmv.info
- 个人的音MAD做法（辅音/元音对齐经验）：immortalt.hatenablog.com/entry/ar1845344
- 鬼畜新人入门（UTAU/Melodyne 原理）：zhuanlan.zhihu.com/p/145284283
- B 站调音教程：BV1UU4y1c7yL（Melodyne+UTAU 全流程）、BV1CB4y1D7zq（UTAU 零基础）
- pretty_midi 文档：craffel.github.io/pretty-midi/
