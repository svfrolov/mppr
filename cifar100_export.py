import torch
import os
import onnx

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

# Экспортируем модель в формат ONNX
torch.onnx.export(model,  # модель
                 x,  # входной тензор
                 onnx_model_path,  # путь для сохранения
                 export_params=True,  # сохраняет веса обученных параметров
                 opset_version=9,  # версия ONNX
                 do_constant_folding=True,  # укорачивание констант для оптимизации
                 input_names=['input'],  # имя входного слоя
                 output_names=['output'],  # имя выходного слоя
                 dynamic_axes={'input': {0: 'batch_size'},  # динамичные оси
                             'output': {0: 'batch_size'}})

print(f"Модель успешно экспортирована в {onnx_model_path}")

# Загружаем модель ONNX для проверки
onnx_model = onnx.load(onnx_model_path)

# Проверяем модель
onnx.checker.check_model(onnx_model)
print("Модель ONNX проверена и корректна!")