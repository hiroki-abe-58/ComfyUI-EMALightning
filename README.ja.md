# EMA Lightning for ComfyUI（日本語概要）

**EMA Lightning**（[canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning)、コード
[canberk7/ema-lightning](https://github.com/canberk7/ema-lightning)）で、ComfyUI 上でトルコ語の音声合成を行うノードパックです。
8.6M パラメータ（34 MB）の小さなモデルで 48 kHz の音声を生成し、数字・日付・時刻・金額はトルコ語のテキスト正規化で読み上げ、各単語の
発話時刻も返します。公式の `ema-lightning` パッケージ経由で **ComfyUI のプロセス内**（GPU または CPU）で動作し、ノードが何かを
ダウンロードすることはありません。

> **非公式**のインテグレーションです。EMA Lightning の作者、Comfy Org とは関係がなく、承認も受けていません。
> 詳細は英語版 [README.md](README.md) と [docs/VERIFICATION.md](docs/VERIFICATION.md) を参照してください。

## ノード

| ノード | 内容 |
| --- | --- |
| **EMA Lightning Loader** | `models/ema_lightning` のモデル、デバイス（GPU / CPU）、スケジューラのバッチサイズ、高速な lightning モード |
| **EMA Lightning Text to Speech** | テキスト、シード、速度（0.25〜4）、サンプルレート（48 / 24 / 16 / 8 kHz）→ `AUDIO`、単語の時刻（JSON）、レポート |

音声は ComfyUI 標準の **Save Audio (Advanced)**（FLAC / MP3 / Opus）や **Preview Audio** に接続します。

## 確認した範囲

- **GPU で公式パッケージとビット単位で一致**：別環境での公式の使い方（`EMA(device).say(...)`）と、波形・単語時刻が 9 件中 9 件で
  完全に一致（短文・長文、数字と日付、3 種のシード、速度 0.75 / 1.0 / 1.5、48 / 24 / 16 kHz、バッチサイズ auto / 1 / 8）。
- **CPU** でも同じ条件のプロセス同士では 9 件中 9 件一致。ただし CPU の浮動小数点の丸めは、プロセスが何を import したかで最大約
  6e-7（-124 dB）変わることがあります（公式パッケージ単体でも同様）。
- **実際の ComfyUI キュー**：同梱ワークフロー 2 本が動作し、保存された 16 ビット FLAC は公式の浮動小数点の音声を 16 ビットに丸めた
  ものと一致（最大差は 16 ビットの半ステップ）。

## インストール

```
git clone https://github.com/hiroki-abe-58/ComfyUI-EMALightning custom_nodes/ComfyUI-EMALightning
python -m pip install -r custom_nodes/ComfyUI-EMALightning/requirements.txt
python custom_nodes/ComfyUI-EMALightning/tools/download_model.py --models-dir models/ema_lightning
```

モデル（約 34 MB）は確認済みのリビジョンに固定し、SHA-256 を検証します。`models/ema_lightning/ema-lightning/` に入ります。

## 注意点

- **batch_size**：`auto` はパッケージ自身の既定（デバイスごとに一度だけスループットを測り、パッケージのキャッシュフォルダに保存）。
  テキストが長いとバッチサイズでごくわずかに音声が変わることがあります（最大 9e-4。バッチ 1 とそれ以外で差が出ます）。
- **lightning**（GPU のみ、既定オフ）：パッケージの高速モード。この Windows 環境では Triton がなく、コンパイルなしの CUDA Graph
  記録になり、初回に 158 秒かかりました。その後は 7.9 秒のアナウンスを 12 ms で生成。通常経路との差は最大 6.8e-4 でビット一致では
  ありません。また開始時にプロセス全体の cuDNN ベンチマークモードを有効にします（他のモデルにも影響）。
- テキストはトルコ語の規則で小文字化されます（例：「ComfyUI」→「comfyuı」）。
- モデルはセッション中ずっと読み込まれたままです（34 MB）。

## ライセンス

- このノードパック：**MIT**（[LICENSE](LICENSE)）。第三者のコードは含みません。
- `ema-lightning`・`normalizer-tr`（インストールされるパッケージ）とモデル（同梱せず、`tools/download_model.py` 実行時のみ取得）：
  **Apache-2.0**。モデルカードは、合成音声の開示や同意に関する規則を含め、利用者が適法に使う責任を負うと述べています。
  [NOTICE](NOTICE) を参照してください。
