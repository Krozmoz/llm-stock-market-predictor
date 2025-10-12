# ===============================================================
#  Dockerfile — Образ для запуска LLM-экспериментов с CUDA
# ---------------------------------------------------------------
#  Включает:
#   • Ubuntu + CUDA 12.8 + cuDNN 9
#   • Python 3.11 + PyTorch 2.7.0 (GPU)
#   • Поддержку RTX 50xx / Blackwell (sm_120)
#
#  Используется для полностью воспроизводимого окружения.
#
#  Автор: Михаил Шардин
#  Онлайн-визитка: https://shardin.name/?utm_source=python
# 
#  Репозиторий: https://github.com/empenoso/llm-stock-market-predictor
# ===============================================================


# Базовый образ с поддержкой CUDA 12.8 и cuDNN 9
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl vim && \
    rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip install --upgrade pip

# Создаём рабочую директорию
WORKDIR /workspace

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
COPY . /workspace/
