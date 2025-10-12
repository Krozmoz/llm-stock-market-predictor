#!/bin/bash
# ===============================================================
#  run.sh — Запуск эксперимента LLM Finance Predictor
# ---------------------------------------------------------------
#  Собирает и запускает Docker-контейнер с поддержкой GPU:
#   • Проверяет наличие CUDA и видеокарты
#   • Монтирует локальную папку проекта внутрь контейнера
#   • Запускает тренировку и анализ результатов
#
#  Пример:
#     bash run.sh
#
#  Смотреть нагрузку на видеокарту: $ watch -n 5 -c --no-wrap nvidia-smi
#
#  Автор: Михаил Шардин
#  Онлайн-визитка: https://shardin.name/?utm_source=python
# 
#  Репозиторий: https://github.com/empenoso/llm-stock-market-predictor
# ===============================================================

set -e

IMAGE_NAME="llm_predictor"
IMAGE_TAG="latest"

echo "#####################################################"
echo "### Сборка Docker-образа: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "#####################################################"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "#####################################################"
echo "### Запуск эксперимента в Docker-контейнере...     ###"
echo "#####################################################"

# Создаём папку для кешей на хосте (вне проекта)
mkdir -p ~/.cache/llm_predictor

docker run --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/Data:/workspace/Data" \
  -v "$(pwd)/results:/workspace/results" \
  -v "$HOME/.cache/llm_predictor:/workspace/.cache" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  -e TORCH_HOME=/workspace/.cache/torch \
  -e MPLCONFIGDIR=/workspace/.config/matplotlib \
  -e TMPDIR=/tmp \
  -e TEMP=/tmp \
  -e TMP=/tmp \
  --rm \
  "${IMAGE_NAME}:${IMAGE_TAG}" \
  python multi_ticker_experiment.py

echo ""
echo "#####################################################"
echo "### Эксперимент завершен. Результаты в папке ./results"
echo "#####################################################"

