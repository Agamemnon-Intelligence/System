# Artemis Plasma Pro

> Sitede yayında olan **Artemis Plasma Pro**, Qwen3.5-0.8B (Apache 2.0) üzerine Artemis kimliği
> eklenmiş modeldir ve `artemis/pro-worker.js` ile WebGPU'da çalışır; bu klasördeki dosyalara ihtiyaç duymaz.
> Buradaki eğiticiler, Plasma Pro'nun ilk planı olan sıfırdan eğitilen 54M'lik modeli üretir.

Sıfırdan eğitilen, ~54 milyon parametreli Türkçe + İngilizce sohbet modeli. Eğitim bittiğinde
`artemis/artemis-2.0-pro.bin` dosyası oluşur. Model tamamen tarayıcıda çalışır ve eğitim sayfasında denenebilir.

Sitede şu an yayında olan **Artemis 2.0 Plasma** (`artemis/artemis-2.0.bin`) ise ayrı bir modeldir:
Artemis 1.0 ile 1.1'in tek bir 27 katmanlı, 23,5M parametreli modelde birleştirilip Türkçe/İngilizce
sohbet ve Vikipedi ile eğitilmeye devam edilmiş hâli. `train_from_11.py` ile üretilir
(`python artemis/plasma/train_from_11.py --minutes 120`; GPU gerekmez).

| | |
|---|---|
| Mimari | 12 katman, d=512, 8 baş, MLP 2048, bağlam 512 token |
| Parametre | 54,5M (gömme 8,4M + çıktı 8,4M + katmanlar 37,7M) |
| Tokenizer | 16.384 token, Türkçe/İngilizce bayt düzeyinde BPE (`tokenizer.json`) |
| Dosya | int8 ağırlıklar (satır ölçekli), ~55 MB |

## Python'suz: tarayıcıda eğitim (WebGPU)

`artemis/plasma/index.html` sayfasını açın (siteden: `/artemis/plasma/`). Süreyi seçin,
**Eğitimi başlat**'a basın. Sayfa veriyi Hugging Face'ten indirir, GPU'da eğitir, sonunda
`artemis-2.0-pro.bin`'i indirmenizi ve aynı sayfada denemenizi sağlar.

- Güncel Chrome / Edge (ya da WebGPU destekli Safari) ve ayrı bir GPU gerekir.
- "GPU belleği" ayarı mikro-yığın boyutudur: ~1 GB / ~2 GB / ~3,5 GB.
- Kendi `.txt` ya da `.jsonl` dosyalarınızı ekleyebilirsiniz. `.jsonl` satırları
  `{"text": …}`, `{"messages": [{"role": "user", …}, …]}` ya da `{"prompt": …, "response": …}` olabilir.
- Var olan bir `artemis-2.0-pro.bin`'den devam edebilirsiniz (ince ayar).

Tarayıcı fp32 ile çalışır, yani PyTorch'tan yavaştır. Aynı 20 dakikada daha az token görür.
En iyi model için Python yolunu kullanın.

## Python ile eğitim (en hızlısı)

```bash
pip install torch numpy tokenizers pyarrow huggingface_hub
python artemis/plasma/train.py --minutes 20
```

Tek komut veriyi indirir (`data/`, bir kez), modeli **tam 20 dakika** eğitir (veri hazırlama
ve ilk derleme hariç), doğrulama kaybını ve örnek yanıtları yazdırır, `artemis/artemis-2.0-pro.bin`'i
üretir. Ardından dosyayı commit edip yayımlayın.

| GPU | Yaklaşık hız | 20 dakikada |
|---|---|---|
| RTX 4090 / A100 | ~150–250k token/sn | ~200–300M token |
| RTX 3060 / T4 (Colab) | ~40–70k token/sn | ~50–80M token |

Sayılar tahmindir; betik gerçek hızı (token/sn ve TFLOP/s) ekrana yazar.

Seçenekler: `--minutes 60` (daha iyi model), `--micro 64` (24 GB+ GPU'da daha hızlı),
`--micro 16` (bellek yetmezse), `--scale 0.25` (daha az veri indir), `--resume` (son
`plasma.pt`'den devam), `--export-only`.

### Neden bu kadar hızlı?

modded-nanogpt hız denemelerinde kendini kanıtlamış teknikler:

- **Muon** optimizer (Newton–Schulz ile dikleştirilmiş momentum) gizli matrislerde, Adam
  yalnızca gömme ve çıktı katmanında. Aynı kayba AdamW'den belirgin biçimde daha az adımda iner.
- **RoPE + QK-norm + ReLU² + parametresiz RMSNorm + logit softcap**, ayrıca sıfırdan
  başlayan çıkış katmanları. Yüksek öğrenme hızında kararlı kalır.
- **bf16 + `torch.compile` + FlashAttention.**
- **Süreye bağlı plan.** Öğrenme hızı ve veri karışımı ilerlemeye değil geçen süreye göre
  ayarlanır, yani GPU ne olursa olsun eğitim tam `--minutes` dakikada biter. Son %40'ta öğrenme
  hızı sıfıra iner ve karışım sohbet verisine kayar (tavlama). Kayıp yalnızca asistan
  yanıtlarında hesaplanır, kullanıcı mesajları maskelenir.

## Veri (hepsi açık lisanslı)

| Kaynak | İçerik | Lisans |
|---|---|---|
| FineWeb-2 (tur_Latn) | Türkçe web metni | ODC-By |
| Vikipedi 20231101.tr | Türkçe ansiklopedi | CC BY-SA 4.0 |
| FineWeb-Edu | İngilizce eğitim metni | ODC-By |
| InstrucTurca + Aya (Türkçe) | Türkçe talimat/yanıt | CC BY-SA 4.0 / Apache 2.0 |
| smol-smoltalk | İngilizce sohbet | Apache 2.0 |
| Kimlik örnekleri | "Sen kimsin?" vb. | `prepare.py` içinde |

## Dosyalar

- `prepare.py`: tokenizer ve veri hazırlama
- `train.py`: PyTorch eğitimi ve `.bin` dışa aktarma
- `train_from_11.py`: Artemis 2.0 Plasma'yı (1.1'in üzerine kurulu, ART1 biçiminde) eğitir
- `index.html`, `plasma.js`, `trainer.js`: WebGPU eğitici. `trainer.js`, `train.py` ile aynı ileri/geri
  yayılımı ve optimizer'ı WGSL'de uygular; gradyanları ve Muon/Adam adımları PyTorch ile ~1e-6
  farkla doğrulandı.
- `vendor/parquet.js`: tarayıcıda parquet okumak için hyparquet (MIT)
- `../artemis-engine.js`: çıkarım motoru. Plasma (ART2) dosyasını int8 olarak bellekte tutar ve
  sohbet boyunca KV önbelleğini yeniden kullanır.
