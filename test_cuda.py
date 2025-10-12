# ===============================================================
#  test_cuda.py — Проверка доступности GPU и версии CUDA
# ---------------------------------------------------------------
#  Запускается внутри Docker для валидации окружения:
#   • torch.cuda.is_available()
#   • torch.cuda.get_device_name(0)
#   • torch.version.cuda
#
#  Используется для диагностики перед запуском эксперимента.
#
#  Автор: Михаил Шардин
#  Онлайн-визитка: https://shardin.name/?utm_source=python
# 
#  Репозиторий: https://github.com/empenoso/llm-stock-market-predictor
# ===============================================================

import torch

def print_info():
    print("=====================================")
    print("PyTorch версия:", torch.__version__)
    print("CUDA версия PyTorch:", torch.version.cuda)
    print("CUDA доступна:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Имя устройства:", torch.cuda.get_device_name(0))
        print("Поддерживаемые архитектуры:", torch.cuda.get_arch_list())
        x = torch.randn(3, 3, device="cuda")
        y = x * 2
        print("Результат вычисления на GPU:\n", y)
    print("=====================================")

if __name__ == "__main__":
    print_info()
