import torch
import os
import sys

# Установка кодировки для вывода
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Создание устройства (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Загрузка модели
# Для четного номера в списке группы
model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                      "cifar100_mobilenetv2_x0_5",
                      pretrained=True)

print("Модель успешно загружена")

# Загрузка модели на устройство
model.to(device)
model.eval()  # Установка модели в режим оценки (не обучения)

# Создаем входной тензор для экспорта
x = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)

# Путь для сохранения модели
onnx_model_path = "media/models/cifar100.onnx"

# Создаем директорию, если она не существует
os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

try:
    # Экспортируем модель в формат ONNX с более высокой версией opset
    torch.onnx.export(model,  # модель
                     x,  # входной тензор
                     onnx_model_path,  # путь для сохранения
                     export_params=True,  # сохраняет веса обученных параметров
                     opset_version=18,  # повышенная версия ONNX
                     do_constant_folding=True,  # укорачивание констант для оптимизации
                     input_names=['input'],  # имя входного слоя
                     output_names=['output'])  # имя выходного слоя
                     # убираем dynamic_axes, так как они могут вызывать проблемы

    print(f"Модель успешно экспортирована в {onnx_model_path}")

    # Проверяем, что файл был создан
    if os.path.exists(onnx_model_path):
        print(f"Файл модели успешно создан по пути: {onnx_model_path}")
        print(f"Размер файла: {os.path.getsize(onnx_model_path) / (1024*1024):.2f} МБ")
    else:
        print(f"Ошибка: файл {onnx_model_path} не был создан")

except Exception as e:
    print(f"Произошла ошибка при экспорте модели: {e}")