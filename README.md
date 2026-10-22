# ML

ML-проекты, курс машинного обучения, ИТМО.

| Проект | Задача | Подход | Результат |
|---|---|---|---|
| [Image Classification](image-classification/) | Классификация фото: собака / утка, собственный датасет ~2 900 изображений | CNN с нуля vs ResNet18 transfer learning, t-SNE эмбеддингов | ResNet18: accuracy 0.98, F1 0.98 (CNN: 0.89 / 0.90) |
| [Icon Autoencoder](icon-autoencoder/) | Сжатие и генерация иконок, ~4 800 иконок Tabler (SVG → PNG) | Свёрточный автоэнкодер, latent_dim 128, генерация из случайного z, интерполяция | Val L1 loss 0.028 |

## Стек

Python, PyTorch, torchvision, scikit-learn, pandas, matplotlib.

## Запуск

Каждый проект независим: свои зависимости в `requirements.txt` и инструкция в README.
Команды выполняются из папки проекта.

Датасеты и веса моделей в репозиторий не входят — они скачиваются и создаются скриптами.
