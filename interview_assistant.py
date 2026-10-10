"""
面接アシスタント（練習モード／本番モード 統合版） Windows 用

- 練習モード: 想定質問を読み上げ → マイクで回答を録音 → 文字起こし → 要点の網羅度・時間・口癖を採点 → 記録
- 本番モード: Zoom/Teams/Meet などから流れてくる面接官の声（スピーカー出力）をループバックで取得
              → 文字起こし → 質問を判定 → 要点を大きく表示（練習で言い漏れた要点は色付き）

質問データは「プロファイル」単位で切り替える。
  profiles/_common.json   … どの面接でも聞かれる共通の質問（無ければ _common.example.json から作る）
  profiles/<会社>.json     … 会社固有の質問（共通と同じ id があれば上書き）
新しい面接用のプロファイルは  python new_profile.py <id> <表示名>  で作れる。
練習記録は practice_log.csv に全プロファイル分をまとめて保存する。
共通の質問の練習結果はどのプロファイルでも引き継がれる。

起動: python interview_assistant.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import queue
import random
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# NumPy 2 系との互換対策
# SoundCard の一部バージョンは numpy.fromstring のバイナリモード（NumPy では非推奨）を
# 使っているため、NumPy 2 系で録音時にエラーになることがある。frombuffer で置き換える。
# soundcard を import する前に適用する必要がある。
# ---------------------------------------------------------------------------
if not hasattr(np, "_ia_orig_fromstring"):
    _orig_fromstring = np.fromstring

    def _fromstring_compat(string, dtype=float, count=-1, *, sep="", like=None):
        if sep == "":
            return np.frombuffer(string, dtype=dtype, count=count).copy()
        return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)

    np._ia_orig_fromstring = _orig_fromstring
    np.fromstring = _fromstring_compat

import tkinter as tk
from tkinter import messagebox, ttk

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROFILES_DIR = BASE_DIR / "profiles"
COMMON_FILE = "_common.json"
COMMON_EXAMPLE = "_common.example.json"  # 共通の質問のひな形（Git 管理。_common.json は個人の回答なので管理外）
STATE_PATH = BASE_DIR / ".last_profile"
LOG_PATH = BASE_DIR / "practice_log.csv"

CAPTURE_SR = 48000          # 録音サンプリング周波数（Windows の既定に合わせる）
TARGET_SR = 16000           # Whisper に渡すサンプリング周波数
BLOCK_SEC = 0.1             # 1 回に読み取る長さ

# whisper.cpp（Vulkan 版 pywhispercpp）用のモデル。あれば GPU（AMD も可）で動かし、なければ下の faster-whisper を使う
WHISPER_CPP_MODEL = BASE_DIR / "models" / "ggml-large-v3-turbo.bin"
# 声のない区間で「ご視聴ありがとうございました」などを作り出さないための VAD モデル（無ければ VAD なし）
WHISPER_CPP_VAD_MODEL = BASE_DIR / "models" / "ggml-silero-v5.1.2.bin"
WHISPER_MODEL = "small"     # faster-whisper（CPU / NVIDIA GPU）用。精度を上げたいなら "medium"（重い）
LIVE_SILENCE_END = 0.8      # 本番: この秒数黙ったら質問の区切りとみなす
LIVE_MIN_SPEECH = 0.8       # 本番: これより短い発話は無視（相づち対策）
PRACTICE_SILENCE_END = 5.0  # 練習: この秒数黙ったら回答終了とみなす（考えながら話しても切れないように長め）
MIC_MIN_THRESHOLD = 0.003   # 練習: マイクで発話とみなす音量の下限（小さめの声・入力レベルの低いマイク向けに本番より低い）
MIC_MAX_GAIN = 20.0         # 練習: 文字起こしの前に小さい声を何倍まで増幅するか
MAX_SEGMENT_SEC = 30.0      # 1 区間の最大長
NOISE_WINDOW_SEC = 5.0      # 雑音レベルを測る期間
NOISE_PERCENTILE = 20       # この期間の音量の下位何 % を雑音レベルとみなすか
NOISE_MULT = 2.0            # 雑音レベルの何倍を超えたら発話とみなすか

# 声のない音に対して Whisper が作り出しやすい決まり文句。文字起こし結果がこれだけなら捨てる
HALLUCINATIONS = ["ご視聴ありがとうございました", "ご清聴ありがとうございました", "チャンネル登録よろしくお願いします",
                  "チャンネル登録お願いします", "最後までご視聴いただきありがとうございました"]

MATCH_THRESHOLD = 0.35      # これ未満は「該当なし」として扱う（質問でない発言は 0.35 未満、正しい質問の多くは 0.35 以上）
FILLERS = ["えー", "えっと", "えーと", "あのー", "あの", "まあ", "なんか", "そのー"]

COLORS = {
    "bg": "#1e2127",
    "panel": "#272b33",
    "fg": "#e6e6e6",
    "sub": "#9aa3ad",
    "accent": "#4ea1ff",
    "ok": "#4cc38a",
    "ng": "#ff6b6b",
    "warn": "#f0a43a",
    "todo": "#c792ea",
}


# ---------------------------------------------------------------------------
# 質問データ
# ---------------------------------------------------------------------------
@dataclass
class Point:
    text: str
    check: list[str]


@dataclass
class Question:
    id: str
    title: str
    variants: list[str]
    keywords: list[str]
    points: list[Point]
    full: str
    target_seconds: int
    scope: str = "common"   # "common"（共通）またはプロファイル id


def _parse_questions(data: dict, scope: str) -> list[Question]:
    qs = []
    for q in data.get("questions", []):
        qs.append(
            Question(
                id=q["id"],
                title=q["title"],
                variants=q["variants"],
                keywords=q.get("keywords", []),
                points=[Point(p["text"], p.get("check", [])) for p in q["points"]],
                full=q.get("full", ""),
                target_seconds=int(q.get("target_seconds", 40)),
                scope=scope,
            )
        )
    return qs


def list_profiles() -> list[tuple[str, str]]:
    """(id, 表示名) の一覧。_ で始まるファイルは共通データなので除く"""
    out = []
    for path in sorted(PROFILES_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            name = json.loads(path.read_text(encoding="utf-8")).get("name", path.stem)
        except json.JSONDecodeError:
            name = f"{path.stem}（JSONエラー）"
        out.append((path.stem, name))
    return out


def load_profile(profile_id: str) -> tuple[list[Question], str]:
    """共通の質問に会社別の質問を重ねる。同じ id は会社別が優先。"""
    common_path = PROFILES_DIR / COMMON_FILE
    example_path = PROFILES_DIR / COMMON_EXAMPLE
    if not common_path.exists() and example_path.exists():
        shutil.copyfile(example_path, common_path)  # 初回はひな形から作る
    common = json.loads(common_path.read_text(encoding="utf-8")) if common_path.exists() else {}
    prof = json.loads((PROFILES_DIR / f"{profile_id}.json").read_text(encoding="utf-8"))
    merged: dict[str, Question] = {}
    for q in _parse_questions(common, "common"):
        merged[q.id] = q
    for q in _parse_questions(prof, profile_id):
        merged[q.id] = q
    vocab = "、".join(v for v in (prof.get("vocabulary", ""), common.get("vocabulary", ""))
                     if v and not v.startswith("[要"))  # 未記入のひな形は文字起こしのヒントに入れない
    return list(merged.values()), vocab


# ---------------------------------------------------------------------------
# 質問判定（文字 n-gram の TF-IDF + キーワード加点）
# 日本語は単語の区切りがないので、形態素解析を使わずに文字の 1〜3 文字単位で比べる。
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[\s、。，．,.!?！？「」『』（）()・…ー〜~]+")
_COMPANY = re.compile(r"(弊社|御社|貴社|うち|わが社|我が社)")

# 面接でよく使う言い換えを 1 つの語にそろえる（文字の一致で比べるため、同義語のままだと外れる）
_SYNONYMS = [
    (re.compile(r"(欠点|弱点|弱み)"), "短所"),
    (re.compile(r"(強み|アピールポイント|良いところ|いいところ)"), "長所"),
    (re.compile(r"テスト"), "試験"),
    (re.compile(r"(土曜日?や日曜日?|土曜日?と日曜日?|土曜日?|日曜日?)"), "土日"),
    (re.compile(r"(お休みの日|休みの日)"), "休日"),
    (re.compile(r"(ひいき|贔屓)"), "応援"),
]


# どの質問にも出てくる丁寧表現。残すと「教えてください」だけで似ていると判定されるため除く
_POLITE = re.compile(
    r"(教えていただけますか|教えてください|教えて|お聞かせください|お願いします|お願いいたします|"
    r"していただけますか|いただけますか|してもらえますか|もらえますか|ください|"
    r"でしょうか|ですか|ますか|ですね|ありますか|そうですか|では|まず|ちなみに)"
)


def normalize(text: str) -> str:
    text = _COMPANY.sub("当社", text)
    text = _PUNCT.sub("", text)
    for pat, rep in _SYNONYMS:
        text = pat.sub(rep, text)
    text = _POLITE.sub("", text)
    return text.lower()


class Matcher:
    def __init__(self, questions: list[Question]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.questions = questions
        self.doc_owner: list[int] = []
        docs: list[str] = []
        for i, q in enumerate(questions):
            for v in q.variants + [q.title]:
                docs.append(normalize(v))
                self.doc_owner.append(i)
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
        self.mat = self.vec.fit_transform(docs)

    def match(self, text: str, top_k: int = 3) -> list[tuple[Question, float]]:
        norm = normalize(text)
        if not norm:
            return []
        v = self.vec.transform([norm])
        sims = (self.mat @ v.T).toarray().ravel()
        best: dict[int, float] = {}
        for doc_idx, s in enumerate(sims):
            qi = self.doc_owner[doc_idx]
            best[qi] = max(best.get(qi, 0.0), float(s))
        for qi, q in enumerate(self.questions):
            hits = sum(1 for k in q.keywords if normalize(k) in norm)
            best[qi] = best.get(qi, 0.0) + 0.12 * min(hits, 3)
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.questions[i], s) for i, s in ranked]


# ---------------------------------------------------------------------------
# 練習の採点と記録
# ---------------------------------------------------------------------------
@dataclass
class Evaluation:
    covered: list[bool]
    duration: float
    fillers: int
    coverage: float
    transcript: str
    gradable: int = 0


def evaluate(q: Question, transcript: str, duration: float) -> Evaluation:
    norm = normalize(transcript)
    covered, gradable, hit = [], 0, 0
    for p in q.points:
        if not p.check:
            covered.append(False)
            continue
        gradable += 1
        ok = any(normalize(c) in norm for c in p.check)
        covered.append(ok)
        hit += ok
    coverage = hit / gradable if gradable else 0.0
    # 長い語から順に照合して「えーと」を「えー」と二重に数えないようにする
    pattern = "|".join(sorted(map(re.escape, FILLERS), key=len, reverse=True))
    fillers = len(re.findall(pattern, transcript))
    return Evaluation(covered, duration, fillers, coverage, transcript, gradable)


class PracticeLog:
    FIELDS = ["timestamp", "profile", "qid", "coverage", "duration", "fillers", "missed", "transcript"]

    def __init__(self, path: Path = LOG_PATH):
        self.path = path
        self.profile = ""
        self.rows: list[dict] = []
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as f:
                self.rows = list(csv.DictReader(f))

    def add(self, q: Question, ev: Evaluation):
        missed = [p.text for p, c in zip(q.points, ev.covered) if p.check and not c]
        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "profile": self.profile,
            "qid": q.id,
            "coverage": f"{ev.coverage:.2f}",
            "duration": f"{ev.duration:.1f}",
            "fillers": str(ev.fillers),
            "missed": " / ".join(missed),
            "transcript": ev.transcript,
        }
        new = not self.path.exists()
        with self.path.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        self.rows.append(row)

    def history(self, q: Question) -> list[dict]:
        """共通の質問はどのプロファイルで練習した記録も使う。会社固有の質問はそのプロファイルの記録だけ。"""
        if q.scope == "common":
            return [r for r in self.rows if r["qid"] == q.id]
        return [r for r in self.rows if r["qid"] == q.id and r.get("profile") == q.scope]

    def weak_score(self, q: Question) -> float:
        """小さいほど苦手。未練習は -1 で最優先。"""
        h = self.history(q)[-3:]
        if not h:
            return -1.0
        return sum(float(r["coverage"]) for r in h) / len(h)

    def last_missed(self, q: Question) -> set[str]:
        h = self.history(q)
        if not h or not h[-1]["missed"]:
            return set()
        return set(h[-1]["missed"].split(" / "))

    def pick_next(self, questions: list[Question], exclude: str | None = None) -> Question:
        pool = [q for q in questions if q.id != exclude] or questions
        scored = sorted(pool, key=lambda q: (self.weak_score(q), random.random()))
        # 苦手上位 3 問からランダム（同じ質問ばかりにならないように）
        return random.choice(scored[:3])


# ---------------------------------------------------------------------------
# 音声入力（発話区間の切り出し）
# ---------------------------------------------------------------------------
def to_16k_mono(block: np.ndarray, loudest: bool = False) -> np.ndarray:
    """loudest=True ならチャンネルを平均せず、いちばん音の大きいチャンネルを使う。
    オーディオインターフェース（Yamaha AG06 など）のマイクは片方のチャンネルにしか入らないので、
    平均すると無音のチャンネルと混ざって声が半分の大きさになる"""
    if block.ndim == 2 and loudest:
        x = block[:, int(np.argmax(np.mean(block ** 2, axis=0)))]
    else:
        x = block.mean(axis=1) if block.ndim == 2 else block
    n = len(x) - len(x) % 3
    return x[:n].reshape(-1, 3).mean(axis=1).astype(np.float32)  # 48k → 16k


def normalize_level(audio: np.ndarray, target_peak: float = 0.5, max_gain: float = MIC_MAX_GAIN) -> np.ndarray:
    """小さく録れた声を、文字起こしの前に十分な大きさにそろえる（増幅は max_gain 倍まで）"""
    if not len(audio):
        return audio
    peak = float(np.percentile(np.abs(audio), 99.9))
    if peak <= 0:
        return audio
    gain = min(max_gain, target_peak / peak)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32) if gain > 1 else audio


class SegmentRecorder(threading.Thread):
    """
    source="loopback": スピーカー出力（面接官の声）を取得
    source="mic":      既定のマイク（自分の声）を取得
    音量で発話の始まりと終わりを判定し、区間ごとに on_segment(audio_16k) を呼ぶ。
    one_shot=True なら 1 区間で終了（練習の回答録音用）。
    """

    def __init__(self, source: str, on_segment, silence_end: float, min_speech: float,
                 one_shot: bool = False, on_level=None):
        super().__init__(daemon=True)
        self.source = source
        self.on_segment = on_segment
        self.on_level = on_level
        self.silence_end = silence_end
        self.min_speech = min_speech
        self.one_shot = one_shot
        # threading.Thread は内部で _stop() メソッドを使うので、同じ名前を付けると is_alive() / join() が壊れる
        self._stop_event = threading.Event()
        self._force_cut = threading.Event()
        self.error: Exception | None = None

    def stop(self):
        self._stop_event.set()

    def cut(self):
        """練習で「回答終了」ボタンが押されたとき、その時点までを区間として確定させる"""
        self._force_cut.set()

    def _device(self):
        import soundcard as sc

        if self.source == "loopback":
            spk = sc.default_speaker()
            # 名前で探すと、同名の入力デバイスを持つオーディオインターフェース（Yamaha AG06 など）で
            # ループバックではなく入力側が選ばれてしまうため、ID で指定する
            return sc.get_microphone(id=spk.id, include_loopback=True), spk.channels
        mic = sc.default_microphone()
        return mic, mic.channels

    def run(self):
        try:
            self._loop()
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.on_segment(None)

    def _loop(self):
        dev, _ = self._device()
        # WASAPI は 1 チャンネルだけを指定すると不正なデータになることがあるため、
        # チャンネル数は指定せずデバイス既定（全チャンネル）で録り、後で平均してモノラル化する
        frames = int(CAPTURE_SR * BLOCK_SEC)
        history: list[float] = []
        buf: list[np.ndarray] = []
        speaking = False
        speech_t = silence_t = 0.0
        with dev.recorder(samplerate=CAPTURE_SR, blocksize=frames) as rec:
            while not self._stop_event.is_set():
                x = to_16k_mono(rec.record(numframes=frames), loudest=self.source == "mic")
                rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
                if self.on_level:
                    self.on_level(rms)
                # 雑音レベル = 直近 NOISE_WINDOW_SEC の音量の下位 NOISE_PERCENTILE%。
                # loopback: 発話中も更新する。「発話していない間の平均」で追従すると、BGM や話し声まで雑音として
                #   学習して閾値が上がり続け、声を拾えなくなる（動画の評価で区間に入ったのが全体の 31% だけだった）
                # mic: 話し始める前の部屋の音だけで決め、発話中は固定する。発話中も更新すると、話し続けるうちに
                #   直近の音が全部自分の声になって閾値が声の大きさまで上がり、言いよどんだ瞬間に回答が切れる
                if self.source == "loopback" or not speaking:
                    history = (history + [rms])[-int(NOISE_WINDOW_SEC / BLOCK_SEC):]
                floor = 0.006 if self.source == "loopback" else MIC_MIN_THRESHOLD
                thr = max(floor, float(np.percentile(history, NOISE_PERCENTILE)) * NOISE_MULT)
                if not speaking:
                    buf = (buf + [x])[-3:]              # 話し始めの直前 0.3 秒を残す
                    if rms > thr:
                        speaking, speech_t, silence_t = True, 0.0, 0.0
                else:
                    buf.append(x)
                    speech_t += BLOCK_SEC
                    silence_t = silence_t + BLOCK_SEC if rms <= thr else 0.0
                    forced = self._force_cut.is_set()
                    if forced or silence_t >= self.silence_end or speech_t >= MAX_SEGMENT_SEC:
                        self._force_cut.clear()
                        audio = np.concatenate(buf)
                        buf, speaking = [], False
                        if speech_t - silence_t >= self.min_speech or forced:
                            self.on_segment(audio)
                            if self.one_shot:
                                return
                if self._force_cut.is_set() and not speaking:
                    # 何も話していないのに終了が押された
                    self._force_cut.clear()
                    self.on_segment(np.zeros(0, dtype=np.float32))
                    if self.one_shot:
                        return


# ---------------------------------------------------------------------------
# 文字起こし（whisper.cpp を優先し、使えなければ faster-whisper）
# ---------------------------------------------------------------------------
class Transcriber:
    def __init__(self, vocabulary: str):
        self.vocabulary = vocabulary
        self.model = None
        self.backend = ""           # "whispercpp" または "faster"
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.model_name = ""
        self.device_name = ""
        self._lock = threading.Lock()  # whisper.cpp のモデルは同時に呼べないため

    def load(self):
        try:
            if self._load_whispercpp():
                return
            from faster_whisper import WhisperModel

            try:
                self.model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
                self.device_name = "GPU"
            except Exception:  # noqa: BLE001
                self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
                self.device_name = "CPU"
            self.backend, self.model_name = "faster", WHISPER_MODEL
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            self.ready.set()

    def _load_whispercpp(self) -> bool:
        if not WHISPER_CPP_MODEL.exists():
            return False
        try:
            from pywhispercpp.model import Model
        except ImportError:
            return False
        # この Vulkan ビルドは複数候補を同時に扱うデコード（ビームサーチ、best_of>1 の温度フォールバック）で
        # access violation を起こすため、候補は常に 1 つにする
        params = dict(print_progress=False, print_realtime=False, no_context=True, suppress_nst=True,
                      greedy={"best_of": 1})
        if WHISPER_CPP_VAD_MODEL.exists():
            params.update(vad=True, vad_model_path=str(WHISPER_CPP_VAD_MODEL))
        try:
            self.model = Model(str(WHISPER_CPP_MODEL), redirect_whispercpp_logs_to=None, **params)
        except Exception:  # noqa: BLE001  読み込めなければ faster-whisper に切り替える
            self.model = None
            return False
        self.backend = "whispercpp"
        self.model_name = WHISPER_CPP_MODEL.stem.removeprefix("ggml-")
        self.device_name = "whisper.cpp"
        return True

    def transcribe(self, audio: np.ndarray, fast: bool) -> str:
        self.ready.wait()
        if self.model is None or len(audio) < TARGET_SR * 0.3:
            return ""
        if self.backend == "whispercpp":
            # GPU なら十分速いので、本番・練習とも同じ設定で動かす
            with self._lock:
                segments = self.model.transcribe(audio.astype(np.float32), language="ja",
                                                 initial_prompt=self.vocabulary)
        else:
            segments, _ = self.model.transcribe(
                audio,
                language="ja",
                beam_size=1 if fast else 5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=self.vocabulary,
            )
        text = "".join(s.text for s in segments).strip()
        rest = _PUNCT.sub("", text)
        for h in HALLUCINATIONS:
            rest = rest.replace(h, "")
        return text if rest else ""


# ---------------------------------------------------------------------------
# 読み上げ（pyttsx3 / Windows SAPI5）
# ---------------------------------------------------------------------------
def speak_async(text: str, done):
    def worker():
        try:
            try:
                import pythoncom  # pywin32

                pythoncom.CoInitialize()
            except ImportError:
                pass
            import pyttsx3

            eng = pyttsx3.init()
            for v in eng.getProperty("voices"):
                tag = f"{v.name} {v.id} {getattr(v, 'languages', '')}".lower()
                if "japan" in tag or "haruka" in tag or "ayumi" in tag or "ichiro" in tag or "ja-jp" in tag:
                    eng.setProperty("voice", v.id)
                    break
            eng.setProperty("rate", 170)
            eng.say(text)
            eng.runAndWait()
            eng.stop()
        except Exception:  # noqa: BLE001  読み上げできない環境では表示だけで進める
            pass
        done()

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.profiles = list_profiles()
        if not self.profiles:
            raise FileNotFoundError("profiles")
        ids = [p[0] for p in self.profiles]
        last = STATE_PATH.read_text(encoding="utf-8").strip() if STATE_PATH.exists() else ""
        self.profile_id = last if last in ids else ids[0]
        self.log = PracticeLog()
        self.tr = Transcriber("")
        self._load_profile_data()
        self.events: queue.Queue = queue.Queue()
        self.recorder: SegmentRecorder | None = None
        self.mode = tk.StringVar(value="practice")
        self.current_q: Question | None = None
        self.live_candidates: list[tuple[Question, float]] = []
        self.answer_started = 0.0

        self._build_ui()
        self.status("音声認識モデルを読み込み中…（初回はダウンロードで数分かかります）")
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(100, self._poll)

    # ---------- プロファイル ----------
    def _load_profile_data(self):
        self.questions, vocab = load_profile(self.profile_id)
        self.matcher = Matcher(self.questions)
        self.tr.vocabulary = vocab  # 文字起こしのヒント語もプロファイルに合わせる
        self.log.profile = self.profile_id
        try:
            STATE_PATH.write_text(self.profile_id, encoding="utf-8")
        except OSError:
            pass

    def _on_profile_selected(self, _event=None):
        idx = self.profile_box.current()
        if idx < 0 or self.profiles[idx][0] == self.profile_id:
            return
        self._stop_recorder()
        self.profile_id = self.profiles[idx][0]
        try:
            self._load_profile_data()
        except (json.JSONDecodeError, KeyError) as e:
            messagebox.showerror("面接アシスタント", f"プロファイルを読み込めませんでした: {e}")
            return
        self.current_q = None
        self.live_candidates = []
        self.q_label.config(text="「次の質問」を押すと始まります")
        self.ans_btn.config(state="disabled", text="回答開始")
        self.live_btn.config(text="聞き取り開始")
        self._write(self.result, [])
        self._write(self.live_view, [])
        self.heard.config(text="")
        for w in self.cand_row.winfo_children():
            w.destroy()
        n_own = sum(q.scope == self.profile_id for q in self.questions)
        self.status(f"{self.profiles[idx][1]} に切り替えました（固有 {n_own} 問＋共通 {len(self.questions) - n_own} 問）")

    # ---------- 共通 ----------
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
        self.root.after(100, self._poll)

    def _on_status(self, msg):
        self.status(msg)

    def _on_level(self, rms):
        self.level["value"] = min(100, rms * 2000)

    def _stop_recorder(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

    # ---------- UI 構築 ----------
    def _build_ui(self):
        c = COLORS
        self.root.title("面接アシスタント")
        self.root.configure(bg=c["bg"])
        self.root.geometry("760x620")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=c["bg"])
        style.configure("TLabel", background=c["bg"], foreground=c["fg"], font=("Yu Gothic UI", 11))
        style.configure("Sub.TLabel", foreground=c["sub"], font=("Yu Gothic UI", 10))
        style.configure("TRadiobutton", background=c["bg"], foreground=c["fg"], font=("Yu Gothic UI", 11))
        style.configure("TButton", font=("Yu Gothic UI", 11), padding=6)
        style.configure("Horizontal.TProgressbar", troughcolor=c["panel"], background=c["accent"])

        top = ttk.Frame(self.root, padding=(12, 10))
        top.pack(fill="x")
        ttk.Label(top, text="モード:").pack(side="left")
        ttk.Radiobutton(top, text="練習", value="practice", variable=self.mode,
                        command=self._switch_mode).pack(side="left", padx=6)
        ttk.Radiobutton(top, text="本番", value="live", variable=self.mode,
                        command=self._switch_mode).pack(side="left", padx=6)
        ttk.Label(top, text="面接先:").pack(side="left", padx=(16, 4))
        self.profile_box = ttk.Combobox(top, state="readonly", width=24,
                                        values=[name for _, name in self.profiles])
        self.profile_box.current([p[0] for p in self.profiles].index(self.profile_id))
        self.profile_box.bind("<<ComboboxSelected>>", self._on_profile_selected)
        self.profile_box.pack(side="left")
        self.level = ttk.Progressbar(top, length=120, maximum=100)
        self.level.pack(side="right")
        ttk.Label(top, text="入力レベル", style="Sub.TLabel").pack(side="right", padx=6)

        self.body = ttk.Frame(self.root, padding=(12, 0))
        self.body.pack(fill="both", expand=True)

        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, style="Sub.TLabel",
                  padding=(12, 6)).pack(fill="x", side="bottom")

        self._build_practice()
        self._build_live()
        self._switch_mode()

    def _text(self, parent, height, size=12):
        t = tk.Text(parent, height=height, wrap="word", bg=COLORS["panel"], fg=COLORS["fg"],
                    insertbackground=COLORS["fg"], relief="flat", padx=10, pady=8,
                    font=("Yu Gothic UI", size))
        for name, color in [("ok", "ok"), ("ng", "ng"), ("warn", "warn"), ("todo", "todo"),
                            ("sub", "sub"), ("accent", "accent")]:
            t.tag_configure(name, foreground=COLORS[color])
        t.tag_configure("h", font=("Yu Gothic UI", size, "bold"))
        t.configure(state="disabled")
        return t

    @staticmethod
    def _write(t: tk.Text, chunks: list[tuple[str, str | tuple]]):
        t.configure(state="normal")
        t.delete("1.0", "end")
        for text, tag in chunks:
            t.insert("end", text, tag)
        t.configure(state="disabled")

    # ---------- 練習モード ----------
    def _build_practice(self):
        f = self.practice_frame = ttk.Frame(self.body)
        self.q_label = ttk.Label(f, text="「次の質問」を押すと始まります", font=("Yu Gothic UI", 15, "bold"),
                                 wraplength=700)
        self.q_label.pack(anchor="w", pady=(4, 8))

        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        self.read_aloud = tk.BooleanVar(value=True)
        ttk.Button(row, text="次の質問（苦手優先）", command=self.next_question).pack(side="left")
        self.ans_btn = ttk.Button(row, text="回答開始", command=self.toggle_answer, state="disabled")
        self.ans_btn.pack(side="left", padx=6)
        ttk.Button(row, text="模範回答を見る", command=self.show_model).pack(side="left")
        ttk.Checkbutton(row, text="質問を読み上げる", variable=self.read_aloud).pack(side="left", padx=10)

        self.result = self._text(f, height=20)
        self.result.pack(fill="both", expand=True, pady=8)

    def next_question(self):
        self._stop_recorder()
        prev = self.current_q.id if self.current_q else None
        q = self.log.pick_next(self.questions, exclude=prev)
        self.current_q = q
        phrasing = random.choice(q.variants)
        self.q_label.config(text=phrasing)
        hist = self.log.history(q)
        info = f"練習回数 {len(hist)} 回" + (f"／前回の網羅度 {float(hist[-1]['coverage']):.0%}" if hist else "")
        scope = "共通" if q.scope == "common" else "この面接先固有"
        self._write(self.result, [(f"［{q.title}・{scope}］ {info}\n", "sub"),
                                  (f"目安 {q.target_seconds} 秒。準備ができたら「回答開始」。\n", "sub")])
        self.ans_btn.config(state="disabled")
        if self.read_aloud.get():
            self.status("読み上げ中…")
            speak_async(phrasing, lambda: self.events.put(("tts_done", None)))
        else:
            self._on_tts_done(None)

    def _on_tts_done(self, _):
        self.ans_btn.config(state="normal", text="回答開始")
        self.status("「回答開始」を押して話してください")

    def toggle_answer(self):
        if self.recorder is None:
            self.answer_started = time.time()
            self.recorder = SegmentRecorder(
                "mic",
                on_segment=lambda a: self.events.put(("answer_audio", a)),
                silence_end=PRACTICE_SILENCE_END,
                min_speech=0.5,
                one_shot=True,
                on_level=lambda r: self.events.put(("level", r)),
            )
            self.recorder.start()
            self.ans_btn.config(text="回答終了")
            self.status(f"録音中…（{PRACTICE_SILENCE_END} 秒黙るか「回答終了」で確定）")
        else:
            self.recorder.cut()

    def _on_answer_audio(self, audio):
        rec, self.recorder = self.recorder, None
        self.ans_btn.config(text="回答開始", state="disabled")
        if audio is None:
            self.status(f"マイクを開けませんでした: {rec.error if rec else ''}")
            self.ans_btn.config(state="normal")
            return
        self.status("文字起こし中…")
        q = self.current_q

        def work():
            text = self.tr.transcribe(normalize_level(audio), fast=False)
            self.events.put(("answer_text", (q, text, len(audio) / TARGET_SR)))

        threading.Thread(target=work, daemon=True).start()

    def _on_answer_text(self, payload):
        q, text, dur = payload
        ev = evaluate(q, text, dur)
        if text:
            self.log.add(q, ev)
        chunks: list[tuple[str, str | tuple]] = [("あなたの回答\n", "h"), ((text or "（聞き取れませんでした）") + "\n\n", ())]
        chunks.append(("要点チェック\n", "h"))
        for p, c in zip(q.points, ev.covered):
            if "[要" in p.text:
                chunks.append((f"  － {p.text}\n", "todo"))
            elif not p.check:
                chunks.append((f"  － {p.text}（自動採点なし）\n", "sub"))
            else:
                chunks.append((f"  {'○' if c else '×'} {p.text}\n", "ok" if c else "ng"))
        diff = ev.duration - q.target_seconds
        t_tag = "ok" if abs(diff) <= q.target_seconds * 0.3 else "warn"
        chunks.append((f"\n時間 {ev.duration:.0f} 秒（目安 {q.target_seconds} 秒）", t_tag))
        chunks.append((f"　口癖 {ev.fillers} 回", "ok" if ev.fillers <= 2 else "warn"))
        if ev.gradable:
            chunks.append((f"　網羅度 {ev.coverage:.0%}\n", "ok" if ev.coverage >= 0.67 else "ng"))
        self._write(self.result, chunks)
        self.ans_btn.config(state="normal")
        self.status("記録しました。「回答開始」でやり直し、「次の質問」で次へ")

    def show_model(self):
        q = self.current_q
        if not q:
            return
        self.result.configure(state="normal")
        self.result.insert("end", "\n模範回答\n", "h")
        for part in re.split(r"(\[要[^\]]*\])", q.full):
            self.result.insert("end", part, "todo" if part.startswith("[要") else ())
        self.result.insert("end", "\n")
        self.result.configure(state="disabled")
        self.result.see("end")

    # ---------- 本番モード ----------
    def _build_live(self):
        f = self.live_frame = ttk.Frame(self.body)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 6))
        self.live_btn = ttk.Button(row, text="聞き取り開始", command=self.toggle_live)
        self.live_btn.pack(side="left")
        ttk.Button(row, text="カメラ下に配置", command=self.place_compact).pack(side="left", padx=6)
        ttk.Label(row, text="透明度", style="Sub.TLabel").pack(side="left", padx=(12, 4))
        self.alpha = tk.DoubleVar(value=1.0)
        ttk.Scale(row, from_=0.4, to=1.0, variable=self.alpha, length=110,
                  command=lambda v: self.root.attributes("-alpha", float(v))).pack(side="left")
        self.topmost = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="最前面", variable=self.topmost,
                        command=lambda: self.root.attributes("-topmost", self.topmost.get())).pack(side="left", padx=8)

        self.heard = ttk.Label(f, text="", style="Sub.TLabel", wraplength=700)
        self.heard.pack(anchor="w")

        self.cand_row = ttk.Frame(f)
        self.cand_row.pack(fill="x", pady=4)

        self.live_view = self._text(f, height=14, size=17)
        self.live_view.pack(fill="both", expand=True, pady=4)

        manual = ttk.Frame(f)
        manual.pack(fill="x", pady=(2, 4))
        ttk.Label(manual, text="手入力:", style="Sub.TLabel").pack(side="left")
        self.manual_entry = tk.Entry(manual, bg=COLORS["panel"], fg=COLORS["fg"], relief="flat",
                                     insertbackground=COLORS["fg"], font=("Yu Gothic UI", 11))
        self.manual_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.manual_entry.bind("<Return>", lambda e: self._on_live_text(self.manual_entry.get()))

        for i in range(3):
            self.root.bind(f"<F{i + 1}>", lambda e, i=i: self.pick_candidate(i))
        self.root.bind("<Escape>", lambda e: self._write(self.live_view, []))

    def toggle_live(self):
        if self.recorder is None:
            self.recorder = SegmentRecorder(
                "loopback",
                on_segment=lambda a: self.events.put(("live_audio", a)),
                silence_end=LIVE_SILENCE_END,
                min_speech=LIVE_MIN_SPEECH,
                on_level=lambda r: self.events.put(("level", r)),
            )
            self.recorder.start()
            self.live_btn.config(text="聞き取り停止")
            self.status("聞き取り中（スピーカー出力を取得しています）")
        else:
            self._stop_recorder()
            self.live_btn.config(text="聞き取り開始")
            self.status("停止しました")

    def _on_live_audio(self, audio):
        if audio is None:
            err = self.recorder.error if self.recorder else ""
            self.recorder = None
            self.live_btn.config(text="聞き取り開始")
            self.status(f"スピーカー出力を取得できませんでした: {err}")
            return
        self.status("文字起こし中…")

        def work():
            text = self.tr.transcribe(audio, fast=True)
            if text:
                self.events.put(("live_text", text))
            else:
                self.events.put(("status", "聞き取り中"))

        threading.Thread(target=work, daemon=True).start()

    def _on_live_text(self, text: str):
        self.heard.config(text=f"聞き取り: {text}")
        self.live_candidates = self.matcher.match(text)
        for w in self.cand_row.winfo_children():
            w.destroy()
        for i, (q, s) in enumerate(self.live_candidates):
            ttk.Button(self.cand_row, text=f"F{i + 1} {q.title}（{s:.2f}）",
                       command=lambda i=i: self.pick_candidate(i)).pack(side="left", padx=(0, 6))
        if self.live_candidates and self.live_candidates[0][1] >= MATCH_THRESHOLD:
            self.pick_candidate(0)
        else:
            self._write(self.live_view, [
                ("該当する想定質問なし\n", "warn"),
                ("結論 → 理由 → 具体例 → この仕事でどう活かすか\n", ()),
                ("わからない場合は「確認させてください」と聞き返してよい\n", "sub"),
            ])
        self.status("聞き取り中" if self.recorder else "待機中")

    def pick_candidate(self, i: int):
        if self.mode.get() != "live" or i >= len(self.live_candidates):
            return
        q = self.live_candidates[i][0]
        missed = self.log.last_missed(q)
        chunks: list[tuple[str, str | tuple]] = [(f"{q.title}　（目安 {q.target_seconds} 秒）\n", "accent")]
        for p in q.points:
            if "[要" in p.text:
                tag = "todo"
            elif p.text in missed:
                tag = "warn"   # 前回の練習で言い漏れた要点
            else:
                tag = ()
            chunks.append((f"・{p.text}\n", tag))
        if missed:
            chunks.append(("\nオレンジ = 前回の練習で言い漏れた要点\n", "sub"))
        self._write(self.live_view, chunks)

    def place_compact(self):
        """画面上部中央（Web カメラの真下）に小さめに配置。視線のずれを小さくするため。"""
        w, h = 720, 420
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+0")
        self.root.attributes("-topmost", True)
        self.topmost.set(True)

    # ---------- モード切替 ----------
    def _switch_mode(self):
        self._stop_recorder()
        if self.mode.get() == "practice":
            self.live_frame.pack_forget()
            self.practice_frame.pack(fill="both", expand=True)
            self.live_btn.config(text="聞き取り開始")
            self.root.attributes("-topmost", False)
        else:
            self.practice_frame.pack_forget()
            self.live_frame.pack(fill="both", expand=True)
            self.root.attributes("-topmost", self.topmost.get())
            self.ans_btn.config(text="回答開始")


def main():
    root = tk.Tk()
    try:
        App(root)
    except FileNotFoundError:
        messagebox.showerror("面接アシスタント",
                             "profiles フォルダに面接先のファイルがありません。new_profile.py で作成してください。")
        return
    root.mainloop()


if __name__ == "__main__":
    main()
