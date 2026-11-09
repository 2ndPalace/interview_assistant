"""
Zoom 会議の文字起こしアプリ（Windows 用）

- スピーカー出力（ループバック＝相手の声）とマイク（自分の声）の2系統を同時に録音
- 発話区間ごとに Whisper（whisper.cpp または faster-whisper）で文字起こし
- 画面に「[HH:MM:SS] 相手: …」「[HH:MM:SS] 自分: …」を色分けして追記
- 開始時に transcripts/YYYYMMDD_HHMMSS.txt を作成し、1発話ごとに追記・flush

起動: python zoom_transcriber.py または start_zoom.bat
"""

from __future__ import annotations

import datetime as dt
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# soundcard より前に interview_assistant を import して NumPy 2 互換パッチを適用する
from interview_assistant import (
    BASE_DIR,
    COLORS,
    HALLUCINATIONS,
    LIVE_SILENCE_END,
    SegmentRecorder,
    Transcriber,
    normalize_level,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
ZOOM_SILENCE_END = LIVE_SILENCE_END  # この秒数黙ったら発話の区切りとみなす（既定 0.8 秒）
ZOOM_MIN_SPEECH = 0.4                # 相づちも拾えるように本番モード（0.8 秒）より短めにする
POLL_INTERVAL_MS = 100               # イベントキューのポーリング間隔（ミリ秒）

SPEAKERS = {
    "loopback": ("相手", "accent"),
    "mic": ("自分", "ok"),
}


class ZoomTranscriberApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.tr = Transcriber("")
        self.recorders: dict[str, SegmentRecorder] = {}
        self.running = False
        self.log_file: Path | None = None
        self.log_fp = None

        self.use_loopback = tk.BooleanVar(value=True)
        self.use_mic = tk.BooleanVar(value=True)

        self._build_ui()
        self.status("音声認識モデルを読み込み中…（初回はダウンロードで数分かかります）")
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(POLL_INTERVAL_MS, self._poll)

    # ---------- モデル読み込み・イベント処理 ----------
    def _load_model(self):
        self.tr.load()
        if self.tr.error:
            self.events.put(("status", f"モデル読み込み失敗: {self.tr.error}"))
        else:
            self.events.put(("status", f"準備完了（{self.tr.model_name} / {self.tr.device_name}）"))

    def status(self, msg: str):
        self.status_var.set(msg)

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                getattr(self, f"_on_{kind}")(payload)
        except queue.Empty:
            pass
        self.root.after(POLL_INTERVAL_MS, self._poll)

    def _on_status(self, msg: str):
        self.status(msg)

    def _on_level(self, rms: float):
        self.level["value"] = min(100, rms * 2000)

    # ---------- UI 構築 ----------
    def _build_ui(self):
        c = COLORS
        self.root.title("Zoom 文字起こし")
        self.root.configure(bg=c["bg"])
        self.root.geometry("780x600")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=c["bg"])
        style.configure("TLabel", background=c["bg"], foreground=c["fg"], font=("Yu Gothic UI", 11))
        style.configure("Sub.TLabel", background=c["bg"], foreground=c["sub"], font=("Yu Gothic UI", 10))
        style.configure("TCheckbutton", background=c["bg"], foreground=c["fg"], font=("Yu Gothic UI", 11))
        style.configure("TButton", font=("Yu Gothic UI", 11), padding=6)
        style.configure("Horizontal.TProgressbar", troughcolor=c["panel"], background=c["accent"])

        top = ttk.Frame(self.root, padding=(12, 10))
        top.pack(fill="x")

        self.start_btn = ttk.Button(top, text="開始", command=self.toggle_run)
        self.start_btn.pack(side="left")

        ttk.Checkbutton(
            top, text="相手（スピーカー）", variable=self.use_loopback, command=self._on_toggle_source
        ).pack(side="left", padx=(14, 6))
        ttk.Checkbutton(
            top, text="自分（マイク）", variable=self.use_mic, command=self._on_toggle_source
        ).pack(side="left", padx=6)

        ttk.Button(top, text="全文コピー", command=self.copy_all).pack(side="left", padx=(16, 4))
        ttk.Button(top, text="保存", command=self.save_as).pack(side="left", padx=4)

        self.level = ttk.Progressbar(top, length=110, maximum=100)
        self.level.pack(side="right")
        ttk.Label(top, text="音量", style="Sub.TLabel").pack(side="right", padx=6)

        body = ttk.Frame(self.root, padding=(12, 0))
        body.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(body, orient="vertical")
        self.text = tk.Text(
            body,
            wrap="word",
            bg=c["panel"],
            fg=c["fg"],
            insertbackground=c["fg"],
            relief="flat",
            padx=12,
            pady=10,
            font=("Yu Gothic UI", 13),
            yscrollcommand=scrollbar.set,
            state="disabled",
        )
        scrollbar.config(command=self.text.yview)
        scrollbar.pack(side="right", fill="y", pady=4)
        self.text.pack(side="left", fill="both", expand=True, pady=4)

        self.text.tag_configure("time", foreground=c["sub"])
        self.text.tag_configure("accent", foreground=c["accent"], font=("Yu Gothic UI", 13, "bold"))
        self.text.tag_configure("ok", foreground=c["ok"], font=("Yu Gothic UI", 13, "bold"))
        self.text.tag_configure("body", foreground=c["fg"])

        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, style="Sub.TLabel", padding=(12, 6)).pack(
            fill="x", side="bottom"
        )

    # ---------- 録音開始・停止 ----------
    def toggle_run(self):
        if not self.running:
            self.start()
        else:
            self.stop()

    def start(self):
        if not self.use_loopback.get() and not self.use_mic.get():
            messagebox.showwarning("Zoom 文字起こし", "「相手（スピーカー）」または「自分（マイク）」にチェックを入れてください。")
            return

        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = TRANSCRIPTS_DIR / f"{stamp}.txt"
        try:
            self.log_fp = self.log_file.open("a", encoding="utf-8")
        except OSError as e:
            messagebox.showerror("Zoom 文字起こし", f"保存ファイルを作成できませんでした: {e}")
            self.log_file = None
            return

        self.running = True
        self.start_btn.config(text="停止")
        self._sync_recorders()
        self.status(f"文字起こし中（自動保存: transcripts/{self.log_file.name}）")

    def stop(self):
        self.running = False
        for rec in list(self.recorders.values()):
            rec.stop()
        self.recorders.clear()
        self._close_log()
        self.start_btn.config(text="開始")
        self.level["value"] = 0
        if self.log_file:
            self.status(f"停止しました（保存先: transcripts/{self.log_file.name}）")
        else:
            self.status("停止しました")

    def _close_log(self):
        if self.log_fp is not None:
            try:
                self.log_fp.close()
            except OSError:
                pass
            self.log_fp = None

    def _on_toggle_source(self):
        if not self.running:
            return
        if not self.use_loopback.get() and not self.use_mic.get():
            self.stop()
            return
        self._sync_recorders()

    def _sync_recorders(self):
        wanted = {
            "loopback": self.use_loopback.get(),
            "mic": self.use_mic.get(),
        }
        for src, enabled in wanted.items():
            if enabled and src not in self.recorders:
                rec = SegmentRecorder(
                    src,
                    on_segment=lambda a, s=src: self.events.put(("segment", (s, a))),
                    silence_end=ZOOM_SILENCE_END,
                    min_speech=ZOOM_MIN_SPEECH,
                    one_shot=False,
                    on_level=lambda r: self.events.put(("level", r)),
                )
                self.recorders[src] = rec
                rec.start()
            elif not enabled and src in self.recorders:
                self.recorders.pop(src).stop()

    # ---------- 音声区間 → 文字起こし → 表示・保存 ----------
    def _on_segment(self, payload):
        source, audio = payload
        label, _ = SPEAKERS[source]
        if audio is None:
            rec = self.recorders.pop(source, None)
            err = rec.error if rec else ""
            self.status(f"{label}の音声デバイスを取得できませんでした: {err}")
            if not self.recorders and self.running:
                self.stop()
            return

        spoken_at = dt.datetime.now().strftime("%H:%M:%S")

        def work():
            target = normalize_level(audio) if source == "mic" else audio
            text = self.tr.transcribe(target, fast=True)
            if not text or text in HALLUCINATIONS:
                return
            self.events.put(("transcript", (source, spoken_at, text)))

        threading.Thread(target=work, daemon=True).start()

    def _on_transcript(self, payload):
        source, spoken_at, text = payload
        if not text:
            return
        label, tag = SPEAKERS[source]
        line = f"[{spoken_at}] {label}: {text}\n"

        self.text.configure(state="normal")
        self.text.insert("end", f"[{spoken_at}] ", "time")
        self.text.insert("end", f"{label}: ", tag)
        self.text.insert("end", f"{text}\n", "body")
        self.text.configure(state="disabled")
        self.text.see("end")

        if self.log_fp is not None:
            try:
                self.log_fp.write(line)
                self.log_fp.flush()
            except OSError as e:
                self.status(f"自動保存の書き込みに失敗しました: {e}")

    # ---------- コピー・保存・終了 ----------
    def copy_all(self):
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            self.status("コピーする文字起こし結果がありません")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status("全文をクリップボードにコピーしました")

    def save_as(self):
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Zoom 文字起こし", "保存する文字起こし結果がありません。")
            return
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        default_name = self.log_file.name if self.log_file else f"{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
        path = filedialog.asksaveasfilename(
            title="文字起こしを保存",
            initialdir=str(TRANSCRIPTS_DIR),
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
            self.status(f"保存しました: {path}")
        except OSError as e:
            messagebox.showerror("Zoom 文字起こし", f"保存できませんでした: {e}")

    def _on_close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    ZoomTranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
