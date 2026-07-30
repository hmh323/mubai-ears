"""耳蜗（cochlea）—— 给生产会话的"我"造的近似人耳的本地听觉前端。

模型本体吃不进音频波形，但读图能力是现成的。于是照人耳耳蜗的原理，把一段声音
变成"一张图 + 一段数字"：
  - 外耳/中耳：采集与传导（这里 = 载入 + 必要时 ffmpeg 解码到 PCM）。
  - 内耳/耳蜗：把声波沿基底膜按频率展开成神经发放——对应**梅尔频谱**。
    梅尔刻度不是比喻：它本就是照人耳频率感知曲线（低频分辨细、高频分辨粗）设计的，
    这正是本方案的立足点。再叠上基频（音高/语调）与能量（响度）、停顿（节奏）。
  - 听觉皮层"听懂"这一步，交给读这张图 + 这段数字的模型完成——不在本文件里。

全程本地，声音不出这台机器。

用法：
    python cochlea.py <音频路径>

支持 webm / ogg / m4a / mp3 / wav。非 wav 先经 ffmpeg 转临时 wav 再进 librosa
（webm/opus 尤其依赖 ffmpeg）。产物落在 ears/out/<音频文件名主干>/：
    listen.png    —— 竖排双面板：梅尔频谱热力图 + 基频/能量曲线（停顿浅带标注）
    summary.json  —— 数字摘要（时长/有声占比/停顿/音高/能量/节奏估计…）
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无界面后端，纯出图
import matplotlib.pyplot as plt

import librosa
import librosa.display

# 中文标题字体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
OUT_ROOT = BASE_DIR / "out"

SR = 22050          # 统一采样率
HOP = 512           # 帧移（≈23ms @22050）
N_MELS = 128
FMAX = 8000         # 梅尔上限（语音有效带宽足够）
PITCH_FMIN = 65.0   # ≈ C2，成人男声下限
PITCH_FMAX = 500.0  # 语音基频上限（含女声/儿童）
PAUSE_MIN_SEC = 0.3
GRID_SEC = 0.1      # pitch_contour 降采样步长


def _find_ffmpeg():
    """PATH 里找 ffmpeg；找不到再看 scoop shims。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    scoop = Path("C:/Users/mh/scoop/shims/ffmpeg.exe")
    if scoop.exists():
        return str(scoop)
    return None


def load_audio(path: Path):
    """载入为单声道 float 波形。非 wav 优先 ffmpeg 转临时 wav，失败再直接交给 librosa。"""
    ext = path.suffix.lower()
    if ext == ".wav":
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        return y, sr

    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        tmp = Path(tempfile.gettempdir()) / f"cochlea_{os.getpid()}_{path.stem}.wav"
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-ar", str(SR), "-ac", "1", str(tmp)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            y, sr = librosa.load(str(tmp), sr=SR, mono=True)
            return y, sr
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    # 没有 ffmpeg：webm/opus 基本没救；mp3/ogg/m4a 可能靠 librosa 后端直读
    if ext == ".webm":
        raise RuntimeError(
            "解码 .webm 需要 ffmpeg，但没找到。请 `scoop install ffmpeg` 后重试。"
        )
    try:
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(
            f"无 ffmpeg 且 librosa 直读 {ext} 失败：{e}。建议 `scoop install ffmpeg`。"
        )


def _runs_below(mask: np.ndarray, times: np.ndarray, min_sec: float):
    """在布尔序列里找连续 True 段（低能量=停顿），返回 [(start, end), ...]，滤掉短的。"""
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            start = float(times[i])
            end = float(times[j - 1])
            if end - start >= min_sec:
                runs.append((round(start, 2), round(end, 2)))
            i = j
        else:
            i += 1
    return runs


def analyze(y: np.ndarray, sr: int) -> dict:
    duration = float(librosa.get_duration(y=y, sr=sr))

    # 能量（RMS）与时间轴
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_t = librosa.times_like(rms, sr=sr, hop_length=HOP)
    peak = float(rms.max()) if rms.size else 0.0
    thr = max(peak * 0.15, 1e-4)              # 有声阈值
    active = rms > thr
    speech_ratio = float(active.mean()) if active.size else 0.0
    pauses = _runs_below(~active, rms_t, PAUSE_MIN_SEC)

    # 基频（音高/语调），清音为 nan
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr, hop_length=HOP
    )
    f0_t = librosa.times_like(f0, sr=sr, hop_length=HOP)
    voiced = f0[~np.isnan(f0)]
    if voiced.size:
        pitch = {
            "min": round(float(np.min(voiced)), 1),
            "max": round(float(np.max(voiced)), 1),
            "mean": round(float(np.mean(voiced)), 1),
            "median": round(float(np.median(voiced)), 1),
        }
    else:
        pitch = {"min": None, "max": None, "mean": None, "median": None}

    # pitch_contour：每 0.1s 一点，取最近帧；清音=null
    contour = []
    frame_period = HOP / sr
    for t in np.arange(0.0, duration, GRID_SEC):
        idx = int(round(t / frame_period))
        if 0 <= idx < len(f0) and not np.isnan(f0[idx]):
            contour.append(round(float(f0[idx]), 1))
        else:
            contour.append(None)

    # 节奏估计（语音下仅供参考）
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr)
        tempo_estimate = round(float(np.atleast_1d(tempo)[0]), 1)
    except Exception:
        tempo_estimate = None

    return {
        "duration_sec": round(duration, 2),
        "speech_ratio": round(speech_ratio, 3),
        "pause_count": len(pauses),
        "pause_positions": pauses,
        "pitch": pitch,
        "pitch_contour": contour,
        "energy_mean": round(float(rms.mean()), 5) if rms.size else 0.0,
        "energy_peak": round(peak, 5),
        "tempo_estimate": tempo_estimate,
        "_rms": rms, "_rms_t": rms_t,          # 画图用，写 json 前会剔除
        "_f0": f0, "_f0_t": f0_t, "_pauses": pauses,
    }


def render(y, sr, info: dict, name: str, out_png: Path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, (ax_mel, ax_pit) = plt.subplots(
        2, 1, figsize=(14, 8), dpi=100, sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.suptitle(f"{name}   ·   {info['duration_sec']}s", fontsize=14)

    # 上：梅尔频谱热力图
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel", fmax=FMAX, ax=ax_mel,
    )
    fig.colorbar(img, ax=ax_mel, format="%+2.0f dB")
    ax_mel.set(title="梅尔频谱（照人耳频率感知）", ylabel="mel 频率")

    # 下：基频实线（清音留空）+ RMS 能量半透明填充（右轴）
    f0, f0_t = info["_f0"], info["_f0_t"]
    ax_pit.plot(f0_t, f0, color="#c0392b", linewidth=1.6, label="基频 f0")
    ax_pit.set(ylabel="基频 (Hz)", xlabel="时间 (s)")
    ax_pit.set_ylim(0, PITCH_FMAX)

    ax_e = ax_pit.twinx()
    rms, rms_t = info["_rms"], info["_rms_t"]
    ax_e.fill_between(rms_t, rms, color="#2980b9", alpha=0.25, label="能量 RMS")
    ax_e.set(ylabel="能量 RMS")

    # 停顿：两个面板都打浅色竖带
    for (s, e) in info["_pauses"]:
        for ax in (ax_mel, ax_pit):
            ax.axvspan(s, e, color="#f1c40f", alpha=0.18, lw=0)

    ax_pit.set_title("基频（语调）+ 能量（响度）+ 停顿（浅带）")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(out_png))
    plt.close(fig)


def main(argv):
    if len(argv) < 2:
        print("用法: python cochlea.py <音频路径>")
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"文件不存在: {path}")
        return 2

    y, sr = load_audio(path)
    info = analyze(y, sr)

    out_dir = OUT_ROOT / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    render(y, sr, info, path.name, out_dir / "listen.png")

    summary = {k: v for k, v in info.items() if not k.startswith("_")}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"耳蜗完成 → {out_dir}")
    print(f"  listen.png / summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
