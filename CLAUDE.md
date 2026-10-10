# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

Windows 専用の Tkinter デスクトップアプリ。面接の「練習モード」（質問読み上げ → マイク録音 → 文字起こし → 採点）と「本番モード」（会議アプリのスピーカー出力をループバック取得 → 文字起こし → 質問判定 → 要点表示）を1本の `interview_assistant.py` で提供する。利用者向けの説明は README.md / GUIDE.md にあり、ドキュメント・コメント・UI 文言はすべて日本語。

## コマンド（PowerShell）

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
python interview_assistant.py                                   # アプリ起動（初回は Whisper small をダウンロード）
python new_profile.py <id> "<表示名>" --type baito|shinsotsu|intern [--force]   # プロファイルのひな形生成
```

テスト・リンター・ビルド設定は存在しない。動作確認は実際にアプリを起動して行う（`soundcard` / `pyttsx3` / `pywin32` は Windows の音声デバイスと SAPI5 前提）。GUI を使わずにロジックだけ確かめるなら、`load_profile` / `Matcher` / `evaluate` / `PracticeLog` は tkinter 以外の依存が軽いので import して直接呼べる。

## アーキテクチャ

### プロファイル（質問データ）
- `profiles/_common.json`（共通）に `profiles/<id>.json`（面接先固有）を `id` 単位で重ねる（`load_profile`）。同じ `id` は面接先側が優先。`_` で始まるファイルは面接先一覧に出ない。`_common.json` も面接先固有のプロファイルも、個人の回答や選考情報を含むため `.gitignore` 済み。Git 管理するのはひな形の `_common.example.json` だけで、`_common.json` が無ければ `load_profile` がこれをコピーして作る。公開リポジトリなので、個人情報（名前・学校・経歴・勤務先など）や選考で知った情報を Git 管理下のファイルに書かないこと。
- 各 `Question` は `scope` を持つ：共通由来なら `"common"`、面接先由来ならプロファイル id。これが練習記録の引き継ぎ範囲を決める（下記）。
- `vocabulary`（面接先＋共通を連結）は Whisper の `initial_prompt` として渡される。`[要` で始まる未記入のひな形は除外される。
- `[要記入]` は「利用者が自分で埋める箇所」のマーカー。練習モードの採点表示でも特別扱い（紫）される。
- `new_profile.py` の `BASE` は共通の `self_intro` / `motivation` などと同じ `id` を使い、会社ごとに上書きさせる設計。`EXTRA` は `--type` ごとの追加質問。
- 最後に選んだプロファイルは `.last_profile` に保存（Git 管理外）。

### 質問判定（本番モード）
`normalize()` で「御社/弊社 → 当社」の統一、句読点除去、同義語の統一（`_SYNONYMS`：欠点/弱み→短所 など）、丁寧表現（「教えてください」等）の除去を行ってから、文字 1〜3-gram の TF-IDF コサイン類似度（`variants` と `title` が文書）＋ `keywords` 一致 1 件あたり 0.12 加点（最大 3 件）でスコアリング。`MATCH_THRESHOLD` 未満は「該当なし」。形態素解析は使っていない。

### 採点と練習記録
- `evaluate()`：各要点の `check` 語のいずれかが（正規化後の）文字起こしに含まれれば「言えた」。`check` が空の要点は採点対象外。口癖は `FILLERS` を長い語優先で数える。
- `PracticeLog` は `practice_log.csv`（UTF-8 BOM 付き、Git 管理外）に全プロファイル分を追記。`history()` は scope が `common` の質問なら全プロファイルの記録を、固有質問ならそのプロファイルの記録だけを返す。`pick_next()` は直近 3 回の網羅度平均が低い（未練習は -1 で最優先）上位 3 問からランダム。

### スレッドと UI
- `SegmentRecorder`（スレッド）が `soundcard` から 48kHz で読み、16kHz モノラルに変換、0.1 秒ブロックの RMS が「直近 `NOISE_WINDOW_SEC` の RMS の下位 `NOISE_PERCENTILE`% × `NOISE_MULT`」を超えたら発話として区間を切り出してコールバック。発話していない間の平均で雑音を追従する方式に戻さないこと（BGM のある音声で閾値が上がり続け、声を拾えなくなる）。ただし `source="mic"`（練習）は発話中に雑音履歴を更新せず、下限も `MIC_MIN_THRESHOLD` と低い。発話中も更新すると、話し続けるうちに閾値が自分の声の大きさまで上がり、回答が 4〜5 秒で切れる。マイクは `to_16k_mono(loudest=True)` で最も大きいチャンネルを使い（AG06 はマイクが CH1 だけに入るので平均すると半分になる）、練習の回答は `normalize_level` で音量をそろえてから文字起こしする。`source="loopback"` は既定スピーカーのループバック、`"mic"` は既定マイク。練習の回答録音は `one_shot=True`。
- `Transcriber` は別スレッドでロードし、バックエンドを2段で選ぶ：`WHISPER_CPP_MODEL` があり `pywhispercpp`（Vulkan 版を自前ビルド。AMD GPU 用）を import できれば whisper.cpp、無ければ faster-whisper（CUDA float16 → CPU int8）。whisper.cpp のモデルはスレッドセーフでないため `_lock` で直列化し、`greedy={"best_of": 1}` 固定（自前の Vulkan ビルドはビームサーチや best_of>1 の温度フォールバックで access violation を起こす。音楽・雑音の多い音声でだけフォールバックが走るので、読み上げ音声のテストでは再現しない）。本番・練習とも同じ設定（`WHISPER_CPP_VAD_MODEL` があれば Silero VAD を有効化）。faster-whisper は本番 `beam_size=1`、練習 5。どちらも結果が `HALLUCINATIONS`（「ご視聴ありがとうございました」等）だけなら空文字にする。
- `SegmentRecorder` の停止フラグは `_stop_event`。`threading.Thread` 内部の `_stop()` と衝突するので `_stop` という名前は使わない。
- ループバックデバイスは `default_speaker().id` で指定する。名前で探すと、出力と同名の入力を持つオーディオインターフェース（Yamaha AG06 など）で入力側が選ばれて無音になる。
- ワーカースレッドから Tk ウィジェットを直接触らない。`App.events` キューに `(kind, payload)` を積み、メインスレッドの `_poll()`（100ms ごと）が `_on_<kind>(payload)` にディスパッチする。新しいイベントを足すときはこの命名規約に従う。
- 読み上げは `speak_async`（pyttsx3、スレッド内で `pythoncom.CoInitialize()`）。失敗しても黙って続行する。

### 注意点
- ファイル冒頭で `soundcard` の import 前に `np.fromstring` をパッチしている（NumPy 2 系で SoundCard が落ちる対策）。import 順を変えないこと。
- 調整用の定数（`WHISPER_CPP_MODEL`, `WHISPER_MODEL`, `LIVE_SILENCE_END`, `PRACTICE_SILENCE_END`, `MATCH_THRESHOLD` など）は `interview_assistant.py` 上部にまとまっており、README の設定表と対応している。変更したら README も合わせる。
- プロファイル JSON は UTF-8・`ensure_ascii=False`・インデント 2 で書く。壊れた JSON は一覧で「（JSONエラー）」と表示される。
