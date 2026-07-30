# 耳朵·第二层:速记员(本地 whisper,字幕层)。
# 分工:耳蜗(cochlea.py)给"调",这里给"字","听懂"由读产物的模型完成。
# 全本地——声音不出这台电脑;模型权重经 hf-mirror 拉一次后走本地缓存。
# 用法: python transcribe.py <音频路径> [模型名]
#   模型名默认 base(约140MB,深夜热点友好);升级听力换 small/medium,重跑自动下载。
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像,家里网络到 HF 本尊不通

from faster_whisper import WhisperModel


def main():
    if len(sys.argv) < 2:
        print("用法: python transcribe.py <音频路径> [模型名]")
        sys.exit(1)
    audio = Path(sys.argv[1])
    if not audio.exists():
        print(f"文件不存在: {audio}")
        sys.exit(2)
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"

    print(f"加载模型 {model_name}(首次会经镜像下载权重,之后走缓存)…")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(audio),
        language="zh",
        vad_filter=True,                      # 静音段跳过,配她"一小句一小句蹦"的说话习惯
        initial_prompt="以下是普通话的日常聊天语音,可能夹杂笑声和哼唱。",
    )

    out_dir = Path(__file__).parent / "out" / audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for seg in segments:
        lines.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
        print(f"[{seg.start:6.2f} → {seg.end:6.2f}] {seg.text.strip()}")

    result = {
        "model": model_name,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "segments": lines,
        "text": "".join(l["text"] for l in lines),
    }
    out_file = out_dir / "transcript.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n速记完成 → {out_file}")


if __name__ == "__main__":
    main()
