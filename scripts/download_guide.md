# AMASS + BABEL Download Guide

## Adım 1: Kayıt Ol

### AMASS
1. https://amass.is.tue.mpg.de/register.php adresine git
2. Kaydol (isim, email, affiliation)
3. Lisans sözleşmesini kabul et (non-commercial research only)
4. Email onayını bekle

### BABEL
1. https://babel.is.tue.mpg.de/register.php adresine git
2. Kaydol
3. Lisans sözleşmesini kabul et
4. Email onayını bekle

## Adım 2: BABEL Label'larını İndir

1. https://babel.is.tue.mpg.de/download.php adresine git (login ol)
2. `babel_v1.0_release` klasörünü indir
3. Aşağıdaki dosyaları içerecek:
   - `train.json`
   - `val.json`
   - `test.json`
   - `extra_train.json`
   - `extra_val.json`
4. Bu klasörü `data/babel_labels/` altına koy:
   ```
   data/babel_labels/babel_v1.0_release/
       train.json
       val.json
       test.json
       extra_train.json
       extra_val.json
   ```

**BABEL boyutu: ~50-100 MB** (sadece JSON)

## Adım 3: Hedef Action'ları Filtrele

```bash
python scripts/filter_babel_actions.py
```

Bu script:
- BABEL JSON'larını okur
- walk, run, jog, jump, sit down, stand up, bend, crouch, turn, fall action'larını arar
- Hangi AMASS .npz dosyalarının gerektiğini listeler
- `data/babel_labels/needed_amass_files.txt` oluşturur

## Adım 4: Sadece Gerekli AMASS Dosyalarını İndir

1. https://amass.is.tue.mpg.de/download.php adresine git (login ol)
2. `needed_amass_files.txt` içindeki path'lere bak
3. SADECE o path'lerin ait olduğu alt-dataset'leri indir
   - Örn: CMU, KIT, BMLrub en hareketli olanlar
4. İndirilen npz'leri `data/amass_npz/` altına (alt-dataset yapısını koruyarak) koy:
   ```
   data/amass_npz/
       CMU/
       KIT/
       BMLrub/
       ...
   ```

**Hedefli indirme ile ~3-8 GB** (tüm AMASS 16-20 GB yerine)

## Adım 5: Mini Dataset Oluştur

```bash
python scripts/build_mini_amass.py
```

Bu script:
- BABEL label'ları ile AMASS npz'leri eşleştirir
- Hedef segmentleri keser
- SMPL/H joint trajectory çıkarır
- Normalize eder (root=0, orientation normalize, 64 frame resample)
- `data/mini_amass/` altına kaydeder

## Hedef Action Listesi

```
walk
run
jog
jump
sit down
stand up
bend
crouch
squat
turn
fall
kick          (optional)
throw         (optional)
```

## Dizin Yapısı (son durum)

```
AGI_visualize_humanmotion/
├── data/
│   ├── babel_labels/
│   │   └── babel_v1.0_release/
│   │       ├── train.json
│   │       ├── val.json
│   │       ├── test.json
│   │       ├── extra_train.json
│   │       └── extra_val.json
│   ├── amass_npz/
│   │   ├── CMU/
│   │   ├── KIT/
│   │   ├── BMLrub/
│   │   └── ... (sadece gerekenler)
│   └── mini_amass/
│       └── (oluşturulacak)
├── scripts/
│   ├── download_guide.md
│   ├── filter_babel_actions.py
│   └── build_mini_amass.py
└── repos/
    ├── amass/
    └── BABEL/
```
