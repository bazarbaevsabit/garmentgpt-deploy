# GarmentGPT Deployment

Развёртывание open-source ML-пайплайна **GarmentGPT** для генерации 3D-геометрии одежды из одной 2D-фотографии.

## Что делает проект

GarmentGPT превращает фото одежды (футболка, платье, брюки) в **структурированную 3D-геометрию** — панели, швы, изгибы, положение в пространстве. Результат сохраняется в формате **GCD (Garment Code Data)** — JSON, который можно открыть в Blender, Clo3D, Unity.

### Пайплайн

1. **VLM-инференс** — визуально-языковая модель (LLaVA-7B) анализирует фото и генерирует последовательность токенов с описанием панелей и рёбер.
2. **Парсинг** — токены превращаются в структурированные индексы.
3. **Геометрическое декодирование** — VQ-VAE-декодеры превращают индексы в реальные координаты вершин, кривых и 3D-позиций.

### Стек

- Python 3.10
- PyTorch 2.8.0
- vLLM 0.10.2
- transformers 4.56.2
- CUDA 12.1
- LLaVA-1.5-7B

## Требования

- **GPU с 24+ ГБ VRAM** (RTX 4090, A5000, A100)
- Docker + nvidia-container-toolkit
- ~50 ГБ свободного места на диске

## Быстрый старт


### 1. Клонировать репозиторий

```bash
git clone https://github.com/bazarbaevsabit/garmentgpt-deploy.git
cd garmentgpt-deploy
```

### 2. Скачать чекпоинты

```bash
pip install "huggingface_hub>=0.34.0,<1.0"
huggingface-cli download ChimerAI/GarmentGPT --local-dir ./checkpoints
```

### 3. Собрать Docker-образ

```bash
docker build -t garmentgpt:latest .
```

### 4. Запустить инференс

```bash
docker run --gpus all -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/test.jpg:/app/test.jpg \
  garmentgpt:latest \
  python main.py \
    --llm_model_path "checkpoints/vlm/checkpoint-12844" \
    --codec_config_path "configs/config_vq1024_resres_aug_decay0.99_q5_gcd_nl8_ld512.yaml" \
    --rt_config_path "configs/config_rt_euler.yaml" \
    --image_path "test.jpg" \
    --output_path "./my_garment.json" \
    --device "cuda:0"
```
## Готовый Docker-образ

Образ доступен на Docker Hub:
```
docker pull bazarbaevsabit/garmentgpt:latest
```

## Что решено в этом форке

- Убрана зависимость от **LLaMA-Factory** (свежая версия требует Python 3.11+, а проект — 3.10).
- Устранены конфликты `numpy 1.x vs 2.x`, `huggingface_hub<1.0`.
- Переписан `main.py` для работы с `transformers` напрямую.
- Собран **Docker-образ** с полным окружением для воспроизводимого развёртывания.
- Почищены чекпоинты: 90 ГБ → 14 ГБ (удалены DeepSpeed-состояния оптимизатора).

## Структура проекта

```
.
├── main.py                      # Основной скрипт инференса
├── Dockerfile                   # Сборка образа
├── requirements_frozen.txt      # Зафиксированные версии зависимостей
├── configs/                     # YAML-конфиги codec и RT-моделей
└── checkpoints/                 # Модели (скачиваются отдельно)
```

## Ссылки

- Оригинальный проект: [ChimerAI-MMLab/Garment-GPT](https://github.com/ChimerAI-MMLab/Garment-GPT)
- Модели на Hugging Face: [ChimerAI/GarmentGPT](https://huggingface.co/ChimerAI/GarmentGPT)

## Лицензия

Apache 2.0 (как в оригинальном проекте).
