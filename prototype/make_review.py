#!/usr/bin/env python3
"""Generate prototype/review.html: v2 (original-audio target) on top, v1 below."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "out"

SONGS = {"roundabout.mid": "Roundabout", "il_vento_doro.mid": "Il Vento D'oro"}
PALETTES = {"anime": "MyGO / Ave Mujica", "genshin": "原神"}


def audio(name):
    return f'<audio controls preload="none" src="out/{name}"></audio>'


def image(name):
    return f'<img src="out/{name}" style="width:100%;max-width:980px;border:1px solid #ddd;display:block;margin:.4rem 0">'


def ref_list(name, limit=40):
    ref_dir = OUT / f"{name}_refs"
    if not ref_dir.exists():
        return ""
    files = sorted(ref_dir.glob("*.mp3"))[:limit]
    items = "".join(
        f'<li>{p.stem}<br><audio controls preload="none" src="out/{name}_refs/{p.name}"></audio></li>'
        for p in files
    )
    return f'<details><summary>逐条原声对照（{len(files)} 条）</summary><ol>{items}</ol></details>'


def fmt(v, suffix=""):
    return "—" if v is None else f"{v}{suffix}"


HEAD = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>鬼畜调音 · 验收</title>
<style>
body{font-family:-apple-system,"PingFang SC",sans-serif;max-width:920px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.6}
h1{font-size:1.45rem}h2{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;font-size:.84rem;margin:.5rem 0}
th,td{border:1px solid #ddd;padding:.3rem .5rem;text-align:left;vertical-align:top}th{background:#f6f6f6}
audio{width:100%;height:34px}.note{color:#555;font-size:.9rem}code{background:#f4f4f4;padding:.05rem .3rem;border-radius:3px}
</style></head><body>"""


def v8_section():
    rows = [(mf.name[:-len(".metrics.json")], json.load(open(mf))) for mf in sorted(OUT.glob("v8_*.metrics.json"))]
    if not rows:
        return ""
    ref = rows[0][1].get("ref_name", "v7_haruhikage")
    names = {"keyframe": "卡点", "keyframe+soyo_down": "卡点 + 素世变慢降调", "keyframe+soyo_down+low_others": "卡点 + 素世变慢降调 + 低音借他人"}
    h = ["""<h2>v8 · 关键帧卡点（最新）</h2>
<p class="note">在 v7 的严格 token 公式上加三处：①关键帧 = 原曲人声的起音点 + 音高跳变点；每个音效的第一个强起音必须落在关键帧 ±10 ms 内（硬约束），音效内部的起音节奏与原曲起音曲线逐帧比较（软约束）。②音效可以在自己第一个强起音前 20 ms 处起头（去掉拖沓的气口），后一个音效在关键帧进来时可以掐断前一个（至少放完 40%）。③低音：素世台词允许整体变慢降调 1–4 个半音（只用于本身偏低、降完落在 63 以下的句子，纯重采样，无声码器），或借用椎名立希/八幡海铃/若叶睦的原调低音台词（每次用他人有额外代价）。</p>"""]
    h.append(f"<p>原曲 {audio(ref + '_orig_mix.mp3')} 原曲人声 {audio(ref + '_orig_vocal.mp3')}</p>")
    h.append("<table><tr><th>渲染</th><th>实测指标</th><th>试听</th></tr>")
    for name, m in rows:
        b = m.get("pitch_bands", {}).get("band_0_61", {})
        mt = (f"λ_N={m['lambda_N']:g}，{m['n_tokens']} 个音效（掐断 {m.get('n_choked', 0)}，降调 {m.get('n_shifted', 0)}，他人 {m.get('n_other_char', 0)}）<br>"
              f"起音在 ±30 ms 内 {round(m['M3_onset_within_30ms']*100)}%，晚于 30 ms {round(m['M3_onset_late_gt30ms']*100)}%，早于 30 ms {round(m['M3_onset_early_gt30ms']*100)}%<br>"
              f"音高 ±50 音分 {round(m['M1_pitch_acc50']*100)}%，该唱在唱 {round(m['M2_voicing_recall']*100)}%<br>"
              f"低音区（MIDI&lt;61，占 {round(b.get('share', 0)*100)}%）覆盖 {round(b.get('covered', 0)*100)}%，准 {round(b.get('acc50', 0)*100)}%")
        png = f"diagnostics/{name}_placement.png"
        h.append(f"<tr><td><code>{name}</code><br>{names.get(m.get('variant'), m.get('variant'))}</td><td>{mt}</td><td><b>人声干声</b><br>{audio(name + '_vocal_dry.mp3')}<br><b>混音</b><br>{audio(name + '.mp3')}<br><a href=\"out/{name}_fragments/index.html\">逐音效碎片</a><br>{image(png) if (OUT / png).exists() else ''}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def v7_section():
    rows = [(mf.name[:-len(".metrics.json")], json.load(open(mf))) for mf in sorted(OUT.glob("v7_*.metrics.json"))]
    if not rows:
        return ""
    ref = rows[0][1].get("ref_name", "v7_haruhikage")
    h = ["""<h2>v7 · 严格 token 版：素世原声整句，不变调不变速</h2>
<p class="note">码本 = 长崎素世的原声台词（1823 句，只在音节边界裁剪，至少保留有声部分的 70%）。每个 token = 用哪句、放在哪、裁哪段、乘多大常数增益；解码就是直接相加，只有 5 ms 淡入淡出。损失逐帧对原曲人声计算：音高（两边都有声的帧，滑音/倚音处权重低）+ 增益闭式解后的响度残差 + 有声/无声错配，外加 λ_N·token 数 + 裁剪代价；用分段 Viterbi 动态规划精确求解。器乐是原曲分离出的伴奏，比人声低 6 dB。</p>"""]
    h.append(f"<p>原曲 {audio(ref + '_orig_mix.mp3')} 原曲人声 {audio(ref + '_orig_vocal.mp3')} 器乐 {audio(ref + '_instrumental.mp3')}</p>")
    sw = OUT / "v7_haruhikage_soyo_sweep.json"
    if sw.exists():
        h.append("<table><tr><th>λ_N</th><th>token 数</th><th>token 中位时长</th><th>音高 ±50 音分</th><th>±100 音分</th><th>该唱在唱</th><th>不该唱在响</th><th>用到台词数</th></tr>")
        for m in json.load(open(sw)):
            h.append(f"<tr><td>{m['lambda_N']:g}</td><td>{m['N']}</td><td>{m['dur_median']} 秒</td><td>{round(m['pitch_acc50']*100)}%</td><td>{round(m['pitch_acc100']*100)}%</td><td>{round(m['voicing_recall']*100)}%</td><td>{round(m['false_alarm']*100)}%</td><td>{m['distinct_lines']}</td></tr>")
        h.append("</table><p class=\"note\">上表是按解码结果直接算的预测值；下面渲染行里的指标是对渲染音频重新做音高检测得到的实测值。</p>")
        h.append(image("diagnostics/v7_haruhikage_soyo_tradeoff.png"))
    h.append("<table><tr><th>渲染</th><th>实测指标</th><th>试听</th></tr>")
    for name, m in rows:
        tag = f"λ_N={m['lambda_N']:g}" + ("，+元音匹配" if m.get("w_vowel", 0) > 0 else "")
        mt = (f"{m['n_tokens']} 个 token，中位 {m['token_dur_median']} 秒，{m['distinct_lines']} 句台词<br>"
              f"音高 ±50 音分 {round(m['M1_pitch_acc50']*100)}%，±100 {round(m['M1_pitch_acc100']*100)}%，中位 {m['M1_pitch_cents_median']} 音分<br>"
              f"该唱在唱 {round(m['M2_voicing_recall']*100)}%，不该唱在响 {round(m['M2_voicing_false_alarm']*100)}%<br>"
              f"起点 F1 {m['M3_onset_f1_50ms']}，强弱相关 {m['M5_dynamics_corr']}<br>增益中位 ±{m['gain_db_median_abs']} dB，平均裁掉 {round(m['crop_mean']*100)}%")
        png = f"diagnostics/{name}_placement.png"
        h.append(f"<tr><td><code>{name}</code><br>{tag}</td><td>{mt}</td><td><b>人声干声</b><br>{audio(name + '_vocal_dry.mp3')}<br>人声（轻混响）<br>{audio(name + '_vocal.mp3')}<br><b>混音</b><br>{audio(name + '.mp3')}<br><a href=\"out/{name}_fragments/index.html\">逐 token 碎片（素材 / 原曲 / 渲染）</a><br>{image(png) if (OUT / png).exists() else ''}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def v5_section():
    rows = [(mf.name[:-len(".metrics.json")], json.load(open(mf))) for mf in sorted(OUT.glob("v5_*.metrics.json"))]
    if not rows:
        return ""
    h = ["""<h2>v5 · 以原曲人声为优化目标：音节级单元选择 + 束搜索</h2>
<p class="note">目标直接取原曲人声：音高正则化为调内主音（短于 0.12 秒的倚音/滑音并入相邻长音，只保留 25% 的原唱偏差），按音节切分；素材单元是长崎素世台词里的真实音节（起音辅音 + 元音）。束搜索在全曲上联合优化：目标代价（移调、时长、元音音色、素材质量）+ 衔接代价（同一句台词里相邻音节连用有奖励、短时间内重复使用有惩罚）。器乐用原曲分离出的伴奏（鼓+贝斯+其他），比人声低 6 dB。</p>
<table><tr><th>渲染</th><th>“听起来对劲”指标</th><th>试听</th></tr>"""]
    lab = [("jianpu_bars9_16_match", "简谱 9–16 小节音程吻合"), ("M1_melody_acc50_vs_main_notes", "M1 主音 ±50 音分"),
           ("M2_voicing_recall", "M2 该唱的地方在唱"), ("M2_voicing_false_alarm", "M2 不该唱却在响"),
           ("M3_onset_f1_50ms", "M3 音节起点吻合 F1"), ("M4_note_stability_cents_median", "M4 音符内抖动（音分）"),
           ("M5_dynamics_corr", "M5 强弱走势相关"), ("M6_shift_semi_median_abs", "M6 移调中位（半音）"),
           ("M6_stretch_median", "M6 延长中位（倍）"), ("M7_contiguous_join_ratio", "M7 整句连用比例"),
           ("M7_distinct_lines", "M7 用到的台词数")]
    for name, m in rows:
        png = f"diagnostics/{name}_pianoroll.png"
        mt = "<br>".join(f"{l}: {m.get(k)}" for k, l in lab)
        h.append(f"<tr><td><code>{name}</code><br>{'、'.join(m['singers'])}<br>{m['n_syllables']} 音节 · {m['key']}</td><td>{mt}</td>"
                 f"<td>原曲<br>{audio(name + '_orig_mix.mp3')}<br>原曲人声<br>{audio(name + '_orig_vocal.mp3')}<br>正则化目标（平滑引导音）<br>{audio(name + '_target_guide.mp3')}<br><b>人声（干）</b><br>{audio(name + '_vocal_dry.mp3')}<br>人声（轻混响）<br>{audio(name + '_vocal.mp3')}<br><b>混音（+原曲器乐 -6 dB）</b><br>{audio(name + '.mp3')}<br>器乐<br>{audio(name + '_instrumental.mp3')}<br><a href=\"out/{name}_fragments/index.html\">逐音节碎片（素材 / 原曲 / 渲染）</a><br>{image(png) if (OUT / png).exists() else ''}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def v4_section():
    rows = [(mf.name[:-len(".metrics.json")], json.load(open(mf))) for mf in sorted(OUT.glob("v4_*.metrics.json"))]
    if not rows:
        return ""
    h = ["""<h2>v4 · 旋律优先：单角色码本 + PSOLA</h2>
<p class="note">先把原曲人声转写成按半音量化的音符（乐句内连奏），再用<b>一个角色</b>的 16 个长元音组成受限码本来唱。每个音符保留起音辅音，元音用 Praat PSOLA 拉平到准确音高并延长（保留共振峰），音符之间带短滑音并互相衔接，长音加轻微延迟颤音；只有很轻的混响。先听“转写音高”确认旋律对不对，再听干声旋律。</p>
<table><tr><th>渲染</th><th>转写</th><th>码本</th><th>旋律贴合</th><th>试听</th></tr>"""]
    for name, m in rows:
        png = f"diagnostics/{name}_pianoroll.png"
        h.append(
            f"<tr><td><code>{name}</code><br>{m['singer']}<br>{m['window'][0]}–{m['window'][1]} 秒</td>"
            f"<td>{m['n_notes']} 音符 / {m['n_phrases']} 乐句<br>{m['key']}<br>音符中位 {m['note_dur_median']} 秒<br>连奏 {round(m['legato_ratio']*100)}%<br>对原曲 ±50音分 {round(m['transcription_vs_orig_acc_50c']*100)}%</td>"
            f"<td>{m['codebook_size']} 个元音，用到 {m['codewords_used']}<br>移调中位 {m['shift_semi_median_abs']} 半音<br>延长中位 ×{m['stretch_median']}</td>"
            f"<td>音符内有声 {round(m['render_voiced_in_notes']*100)}%<br>中位偏差 {m['render_cents_median']} 音分<br>±50音分 {round(m['render_acc_50c']*100)}%</td>"
            f"<td>原曲混音<br>{audio(name + '_orig_mix.mp3')}<br>原曲人声<br>{audio(name + '_orig_vocal.mp3')}<br>转写音高（纯音）<br>{audio(name + '_transcription_tone.mp3')}<br><b>旋律（干声）</b><br>{audio(name + '_melody_dry.mp3')}<br>旋律（轻混响）<br>{audio(name + '_melody.mp3')}<br>混音（+贝斯 -10 dB）<br>{audio(name + '.mp3')}<br>码本原声（16 个元音）<br>{audio(name + '_codebook_raw.mp3')}<br><a href=\"out/{name}_fragments/index.html\">逐音符碎片对照</a><br>{image(png) if (OUT / png).exists() else ''}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def v3_section():
    rows = []
    for mf in sorted(OUT.glob("v3_*.metrics.json")):
        name = mf.name[:-len(".metrics.json")]
        diag_path = OUT / "diagnostics" / f"{name}.diagnostics.json"
        diag = json.load(open(diag_path)) if diag_path.exists() else {}
        rows.append((name, json.load(open(mf)), diag))
    if not rows:
        return ""
    h = ["""<h2>v3 · 分层采样器（Roundabout）</h2>
<p class="note">三层叠加、不停顿：<b>旋律层</b>由原曲主旋律（Melodia）切成连奏音符，每个音符由角色台词里挑出的真实元音核演唱（每乐句固定一个角色）；用 WORLD 只改基频、保留共振峰，音高偏移中位数很小，长音用元音中段来回循环延长。<b>梗层</b>是未经任何处理的整句台词，放在乐句间隙。<b>伴奏</b>是跟随原曲贝斯的合成低音，低于人声 10 dB。输出不含任何原曲音频。</p>
<table><tr><th>渲染</th><th>音符/乐句</th><th>占空比</th><th>旋律贴合（CREPE 对目标）</th><th>音高偏移</th><th>试听（先听原曲对照）</th></tr>"""]
    for name, m, diag in rows:
        png = f"diagnostics/{name}_vs_original.png"
        diag_html = image(png) if (OUT / png).exists() else ""
        h.append(
            f"<tr><td><code>{name}</code><br>{m.get('orig_window', ['—', '—'])[0]}–{m.get('orig_window', ['—', '—'])[1]} 秒</td>"
            f"<td>{m.get('n_notes')} 音符 / {m.get('n_phrases')} 乐句<br>音符中位 {fmt(m.get('note_dur_median'), ' 秒')}<br>梗层 {m.get('n_memes')} 句</td>"
            f"<td>{fmt(m.get('render_duty_cycle'))}</td>"
            f"<td>中位偏差 {fmt(m.get('melody_median_abs_cents'), ' 音分')}<br>±50音分 {fmt(round(m.get('melody_acc_50c', 0) * 100), '%')}<br>±100音分 {fmt(round(m.get('melody_acc_100c', 0) * 100), '%')}</td>"
            f"<td>中位 {fmt(m.get('pitch_shift_semi_median_abs'), ' 半音')}<br>P90 {fmt(m.get('pitch_shift_semi_p90_abs'), ' 半音')}</td>"
            f"<td>原曲混音<br>{audio(name + '_orig_mix_excerpt.mp3')}<br>原曲人声轨<br>{audio(name + '_orig_vocal_excerpt.mp3')}<br><b>v3 混音</b><br>{audio(name + '.mp3')}<br>旋律层<br>{audio(name + '_melody_layer.mp3')}<br>梗层<br>{audio(name + '_meme_layer.mp3')}<br>伴奏<br>{audio(name + '_backing.mp3')}<br>旋律层所用元音核（未处理原声）<br>{audio(name + '_nuclei_reference.mp3')}<br>梗层原声顺序对照<br>{audio(name + '_reference.mp3')}<br>{ref_list(name)}<br>chroma DTW {fmt(diag.get('chroma_dtw'))} · MFCC DTW {fmt(diag.get('mfcc_dtw'))}<br>{diag_html}</td></tr>"
        )
    h.append("</table>")
    return "\n".join(h)


def v2_section():
    rows = []
    for mf in sorted(OUT.glob("v2_*.metrics.json")):
        name = mf.name[:-len(".metrics.json")]
        m = json.load(open(mf))
        diag_path = OUT / "diagnostics" / f"{name}.diagnostics.json"
        diag = json.load(open(diag_path)) if diag_path.exists() else {}
        rows.append((name, m, diag))
    if not rows:
        return ""
    h = ["""<h2>v2 · 整句检索+强行改音高（已被 v3 取代）</h2>
<p class="note">不再使用 MIDI 谱面。对原曲做人声/贝斯分离（demucs），从人声轨提取连续 F0 轮廓与能量包络作为渲染目标：旋律是一条连续曲线，原曲换气/间奏处渲染也留白（占空比与原曲一致），不再有"点状音符"问题。素材仍按乐句整体匹配，F0 被引导到原曲连续轮廓上；伴奏为跟随原曲贝斯轨轮廓的合成低音（-10 dB），输出不含任何原曲音频。</p>
<table><tr><th>渲染</th><th>原曲窗口</th><th>乐句/素材</th><th>占空比</th><th>旋律贴合</th><th>试听（先听原曲对照）</th></tr>"""]
    for name, m, diag in rows:
        diag_html = ""
        png = f"diagnostics/{name}_vs_original.png"
        if (OUT / "diagnostics" / f"{name}_vs_original.png").exists():
            diag_html = image(png)
        h.append(
            f"<tr><td><code>{name}</code></td>"
            f"<td>{m.get('orig_window', ['—', '—'])[0]}–{m.get('orig_window', ['—', '—'])[1]} 秒</td>"
            f"<td>{m.get('n_phrases')} 乐句<br>素材中位 {fmt(m.get('phrase_dur_median'), ' 秒')}</td>"
            f"<td>原曲 {fmt(m.get('target_voiced_ratio'))}<br>渲染 {fmt(m.get('render_duty_cycle'))}</td>"
            f"<td>中位偏差 {fmt(m.get('melody_median_abs_cents_vs_orig'), ' 音分')}<br>±50音分 {fmt(round(m.get('melody_acc_50c_vs_orig', 0) * 100), '%')}</td>"
            f"<td>原曲混音<br>{audio(name + '_orig_mix_excerpt.mp3')}<br>原曲人声轨<br>{audio(name + '_orig_vocal_excerpt.mp3')}<br><b>v2 混音</b><br>{audio(name + '.mp3')}<br>v2 人声<br>{audio(name + '_vocal.mp3')}<br>v2 伴奏<br>{audio(name + '_backing.mp3')}<br>原声顺序对照<br>{audio(name + '_reference.mp3')}<br>{ref_list(name)}<br>chroma DTW {fmt(diag.get('chroma_dtw'))} · MFCC DTW {fmt(diag.get('mfcc_dtw'))}<br>{diag_html}</td></tr>"
        )
    h.append("</table>")
    return "\n".join(h)


def v1_section():
    rows = []
    for mf in sorted(OUT.glob("v1_*.metrics.json")):
        name = mf.name[:-len(".metrics.json")]
        diag_path = OUT / "diagnostics" / f"{name}.diagnostics.json"
        diag = json.load(open(diag_path)) if diag_path.exists() else {}
        rows.append((name, json.load(open(mf)), diag))
    if not rows:
        return ""
    h = ["""<h2>v1 · 以 MIDI 谱面为目标（已被 v2 取代）</h2>
<p class="note">谱面把音乐拆成点状音符，目标本身占空比低，听感破碎；保留在此仅作对照。GPU 批处理 CREPE 提取候选轮廓，声码器把选中素材音高向乐句引导（强度 1.0）；人声/伴奏分离，伴奏 -10 dB。旧版 v1–v7（逐音符密集映射）在 <code>legacy/</code>。</p>
<table><tr><th>渲染</th><th>长度</th><th>人声素材</th><th>素材时长中位</th><th>旋律引导</th><th>伴奏</th><th>试听</th></tr>"""]
    for name, m, diag in rows:
        diag_html = ""
        for png in (f"diagnostics/{name}_melody_compare.png", f"diagnostics/{name}_clip_compare.png"):
            if (OUT / png).exists():
                diag_html += image(png)
        h.append(
            f"<tr><td><code>{name}</code><br>{SONGS.get(m.get('song'), m.get('song'))} · {PALETTES.get(m.get('palette'), m.get('palette'))}</td>"
            f"<td>{fmt(m.get('dur_limit'), ' 秒')}</td>"
            f"<td>{m.get('n_vocal_clips')} 条 / {m.get('notes_covered')} 音<br>{m.get('vocal_clips_per_second')} 条每秒</td>"
            f"<td>{fmt(m.get('vocal_clip_dur_median'), ' 秒')}</td>"
            f"<td>强度 {fmt(m.get('pitch_strength'))}<br>中位偏差 {fmt(m.get('melody_median_abs_cents'), ' 音分')}<br>±50音分 {fmt(round(m.get('melody_acc_50c', 0) * 100), '%') if m.get('melody_acc_50c') is not None else '—'}</td>"
            f"<td>{m.get('backing_notes')} 音<br>{fmt(m.get('backing_vs_vocal_active_db'), ' dB')}</td>"
            f"<td>目标旋律<br>{audio(name + '_melody_reference.mp3')}<br>原声顺序对照<br>{audio(name + '_reference.mp3')}<br>混音<br>{audio(name + '.mp3')}<br>人声<br>{audio(name + '_vocal.mp3')}<br>伴奏<br>{audio(name + '_backing.mp3')}<br>{ref_list(name)}<br>MFCC match {fmt(diag.get('mfcc_aligned_cosine'))}<br>Chroma match {fmt(diag.get('chroma_aligned_cosine'))}<br>{diag_html}</td></tr>"
        )
    h.append("</table>")
    return "\n".join(h)


def main():
    h = [HEAD, "<h1>鬼畜调音 · 验收</h1>", v8_section(), v7_section(), v5_section(), v4_section(), v3_section(), v2_section(), v1_section(),
         '<p class="note">旧结果：<code>legacy/out/</code>；旧引擎：<code>legacy/engine/</code>。</p></body></html>']
    (ROOT / "review.html").write_text("\n".join(x for x in h if x))
    n2 = len(list(OUT.glob("v2_*.metrics.json"))) + len(list(OUT.glob("v3_*.metrics.json")))
    n1 = len(list(OUT.glob("v1_*.metrics.json")))
    print(f"review.html written: {n2} v2 renders, {n1} v1 renders")


if __name__ == "__main__":
    main()
