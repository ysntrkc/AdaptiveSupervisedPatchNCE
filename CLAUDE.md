# ASP — Adaptive Supervised PatchNCE (karşılaştırma yöntemi)

Fork: `ysntrkc/AdaptiveSupervisedPatchNCE` ← upstream `lifangda01/AdaptiveSupervisedPatchNCE`
(Li et al., MICCAI 2023, MIST veri seti). CUT (Contrastive Unpaired Translation) kod tabanı üzerine kurulu.
Tezde H&E→IHC için karşılaştırma yöntemi / DWT kaybının ekleneceği ikinci mimari olarak kullanılacak.
Çalışma alanı kuralları, veri setleri ve run düzeni için `../CLAUDE.md`'yi de oku.

**Durum (2026-10-29):** Bu makinede henüz tam bir eğitim yapılmadı. Önceki deneme (`runs/_archive/asp_her2_asp_failed`)
torch 1.12 RTX 5070'te çalışmadığı için çöktü. uv ortamında (torch 2.9.1+cu128) kısa train + test smoke testi geçti;
kodda değişiklik gerekmedi.

## Kurulum ve komutlar

```bash
uv sync                                  # repo kökünde; Python 3.12, torch 2.9.1+cu128

# yol 1: upstream launcher (tmux + GPUtil; experiments/mist_launcher.py içindeki seçenekleri kullanır)
uv run python -m experiments mist train 0     # mist_launcher'daki 0. deney
uv run python -m experiments mist test 0      # test: phase=val → <dataroot>/valA,valB gerekir; crop 1024 (tam görüntü), num_test 1000
uv run python -m experiments mist dry 0       # sadece komutu yazdır

# yol 2: doğrudan (mist_launcher ile aynı seçenekler; çıktılar runs/ düzenine)
uv run python train.py --dataroot <view_dir> --checkpoints_dir /mnt/d/thesis/runs \
    --name asp/lambda_linear/run_YYYYMMDD_HHMMSS_MIST_HER2 \
    --model cpt --CUT_mode FastCUT --netG resnet_6blocks --netD n_layers --n_layers_D 5 --ndf 32 \
    --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 \
    --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --lambda_gp 10.0 \
    --gp_weights '[0.015625,0.03125,0.0625,0.125,0.25,1.0]' --lambda_asp 10.0 --asp_loss_mode lambda_linear \
    --dataset_mode aligned --direction AtoB --batch_size 1 --load_size 1024 --crop_size 512 --preprocess crop \
    --n_epochs 30 --n_epochs_decay 10 --save_epoch_freq 5 --display_id -1
uv run python test.py <aynı model seçenekleri> --checkpoints_dir /mnt/d/thesis/runs --results_dir /mnt/d/thesis/runs \
    --name asp/lambda_linear/run_... --phase test --crop_size 1024 --num_test 1000
```

`experiments/mist_launcher.py` içinde `dataroot="datasets/MIST/HER2/TrainValAB"` ve `checkpoints_dir='checkpoints'`
göreli; yol 1'i kullanacaksan bunları `/mnt/d/thesis/...` yollarına çevir.

## Veri formatı (BCI'den farklı!)

`dataset_mode aligned` burada **ayrı klasörler** bekler: `<dataroot>/{trainA,trainB,testA,testB}` (veya `valA/valB`;
`phase=test` iken `testA` yoksa `valA` kullanılır). Dosya adları A ve B'de aynı olmalı. BCI'deki yan yana AB formatını kullanmaz.

D:'deki MIST HER2 `HE/{train,test}` + `IHC/{train,test}` şeklinde. Kopyalamadan bir görünüm klasörü kur:
```bash
V=<view_dir>; M=/mnt/d/thesis/datasets/MIST/HER2; mkdir -p $V
ln -sfn $M/HE/train $V/trainA; ln -sfn $M/IHC/train $V/trainB; ln -sfn $M/HE/test $V/testA; ln -sfn $M/IHC/test $V/testB
```
BCI ve HER2match için aynısı: `BCI/{HE,IHC}/{train,test}`, `HER2match/{HE,IHC}/{train,val,test}`.

## Mimari

| Dosya | İçerik |
|---|---|
| `models/cpt_model.py` | `--model cpt`: CUT/FastCUT + ASP (eşli PatchNCE, `lambda_asp`) + Gauss piramidi L1 (`lambda_gp`, `gp_weights`) |
| `models/asp_loss.py` | `AdaptiveSupervisedPatchNCELoss`; `--asp_loss_mode <scheduler>_<lookup>` (scheduler: sigmoid/linear/lambda/zero, lookup: top/linear/bell/uniform). Makaledeki en iyi: `lambda_linear` |
| `models/patchnce.py` | standart PatchNCE (eşsiz, `lambda_NCE`) |
| `models/gauss_pyramid.py` | `Gauss_Pyramid_Conv(num_high=5)` → 6 seviye |
| `models/networks.py` | G: `resnet_{9,6,4}blocks`, `unet_*`, `stylegan2`, `fdlresnet`, `fdlunet`…; F: `mlp_sample`; D: `basic`, `n_layers`, … |
| `models/cut_model.py` | orijinal CUT (`--model cut`) |
| `experiments/` | `mist_launcher.py` (makale ayarları), `tmux_launcher.py` (komut üretimi, GPU seçimi), `__main__.py` (CLI) |
| `evaluate.py` | PHV (ResNet50 perceptual hash), FID (pytorch-fid, standart), KID (`util/kid_score.py`, subprocess), PSNR/SSIM |
| `util/kid_score.py`, `util/fid.py`, `util/perceptual.py`, `util/fdlutil.py` | metrik/FDL yardımcıları |

`--CUT_mode FastCUT` varsayılanları ezer: `nce_idt=False, lambda_NCE=10, n_epochs=20, n_epochs_decay=10`
(launcher bunları 30+10 ile tekrar ezer).

## Bilinen tuzaklar

- `--asp_loss_mode lambda_*` ile **`--n_epochs_decay 0` sıfıra böler** (`asp_loss.py`: `m = 1/(1-k)`).
  Smoke testte en az 1 decay epoch ver.
- Değerlendirme: `uv run python evaluate.py --real_dir <run>/<test_dir>/images/real_B --fake_dir <run>/<test_dir>/images/fake_B`
  (test.py görselleri `images/{real_A,fake_B,real_B}/` alt klasörlerine aynı adlarla yazar). Sonuçları dosyaya
  değil stdout'a yazar; `> <run>/<test_dir>/evaluation.txt` ile kaydet. 2026-10-29'da düzeltildi: argümanlar
  (eskiden sabit `/path/to/...`), SSIM `channel_axis=2` (skimage 0.26), KID `python -m util.kid_score` olarak
  çağrılıyor (script olarak çağrılınca `util/html.py` stdlib `html`'i gölgeleyip torchvision importunu kırıyordu).
- `scipy<1.18` sabit: pytorch-fid 0.3.0 `linalg.sqrtm(disp=...)` kullanıyor, scipy 1.18 bunu kaldırdı.
- `test.py`'de `--num_test` varsayılanı **50**; tam test için `--num_test 1000` ver.
- `test.py` `<checkpoints_dir>/<name>/test_opt.txt`'nin üzerine yazar.
- `--dataroot` varsayılanı `placeholder`, `--dataset_mode` varsayılanı `unaligned`, `--model` varsayılanı `cut` — her zaman açıkça ver.
- ASP otomatik run klasörü oluşturmaz; benzersiz adı `--name` içine sen koy (`asp/<variant>/run_<tarih>_<veri>`).
- `experiments/pretrained_launcher.py` upstream'in cityscapes/horse2zebra örnekleri; tezle ilgisi yok.

## Değişiklik yaparken

- Upstream stili: tek tırnak, kısa satır içi yorumlar, CUT kod tabanı düzeni. Gereksiz refactor yapma.
- DWT kaybı eklenecekse `cpt_model.py:compute_G_loss` içine ayrı bir terim + `modify_commandline_options`'a ağırlık
  argümanı olarak ekle (BCI'deki `haar_wavelet_decomposition` ile aynı tanımı kullan ki sonuçlar karşılaştırılabilir olsun).
