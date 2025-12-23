# Food Object Detection (PyTorch, Faster R‑CNN)

Учебный проект по детекции объектов на изображениях еды с использованием
`PyTorch` и модели **Faster R‑CNN (ResNet‑50 FPN)** на датасете  
**MM‑Food‑100K** (HuggingFace).

Основные идеи:

- загрузка метаданных датасета с HuggingFace;
- скачивание небольшого поднабора изображений (≈200 штук);
- свой класс `Dataset` и `DataLoader` для задачи детекции;
- дообучение предобученной модели Faster R‑CNN;
- сохранение чекпоинтов (возможность продолжить обучение);
- сравнение инференса **до** и **после** обучения.

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/Jorjanoo/food-cv.git
cd food-cv

# 2. (рекомендуется) создать виртуальное окружение
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Установить зависимости (CPU-версия PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas matplotlib opencv-python tqdm requests "huggingface_hub[cli]" fsspec jupyterlab

Доступ к датасету HuggingFace
CSV с метаданными берётся из датасета Codatta/MM-Food-100K:

hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv

Запуск ноутбука

jupyter lab
# или
jupyter notebook
Затем в браузере открыть файл:

food_object_detection_pytorch.ipynb

И выполнить ячейки по порядку сверху вниз.

Что делает ноутбук
Краткая структура:

Imports and Environment Setup
Импорт библиотек, выбор устройства (cuda / cpu).

Dataset Loading
Загрузка CSV с HuggingFace и выбор небольшого поднабора данных.

Image Download & image_path
Скачивание картинок в папку images/ и создание столбца image_path
(относительный путь к файлу).

Custom FoodDetectionDataset
Класс Dataset, который:

читает изображения с диска;
создаёт простую разметку bounding box для объекта еды;
возвращает данные в формате, совместимом с моделями детекции torchvision.
DataLoader
Обёртка над датасетом + функция collate_fn для детекционных моделей.

Model Initialization

загрузка предобученной Faster R‑CNN ResNet‑50 FPN;
перенастройка выходного слоя под 2 класса (фон + еда);
перенос модели на device.
Baseline Inference (Before Training)
Инференс на текущих весах до обучения.

Training Loop c чекпоинтами

обучение на указанное число эпох;
сохранение чекпоинта после каждой эпохи в checkpoints/fasterrcnn_food.pt;
при наличии чекпоинта обучение автоматически продолжается с последней эпохи;
при ручной остановке (Stop / KeyboardInterrupt) текущий прогресс
также сохраняется.
Training Progress Visualization
График loss по эпохам.

Inference After Training
Инференс после обучения и сравнение с базовой моделью.

Контрольные точки (checkpoints)
Директория: checkpoints/
Файл модели: fasterrcnn_food.pt
Веса не коммитятся в репозиторий (папка добавлена в .gitignore).
Если файл чекпоинта существует:

загружается состояние модели и оптимизатора;
обучение продолжается с эпохи last_epoch + 1.

Примечание
Проект носит учебный характер:

используется небольшой поднабор датасета (по умолчанию ~200 изображений);
bounding box для объекта задаётся упрощённо;
нет полноценного разбиения на train/val и продвинутых метрик.
Тем не менее, ноутбук покрывает типичный end‑to‑end workflow
для обучения модели детекции объектов в PyTorch.

