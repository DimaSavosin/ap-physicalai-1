# Отчёт по практике 2 Physical AI

[Параметры и их значения по умолчанию](parameters.md) · [Сравнение и выводы по всем опытам](comparison.md) · [Подробные результаты](experiments.md)

Во всех пяти опытах использованы 512 параллельных сред, seed 1, обучение с нуля на GPU T4. Уменьшение масштаба с дефолтных 4096 явно зафиксировано; результаты относятся к 512 средам. Проверка каждого финального чекпоинта выполнена на CPU в 10 эпизодах с одинаковым критерием доставки (расстояние до цели менее 0.05 м в конце). Новые видео по решению автора не записывались.

| Опыт | Блокнот с кодом и выводом | Финальный чекпоинт |
|---|---|---|
| baseline | [baseline.ipynb](baseline.ipynb) | [model_1000.pt](checkpoints/baseline/model_1000.pt) |
| max_iters_500 | [max_iters_500.ipynb](max_iters_500.ipynb) | [model_499.pt](checkpoints/max_iters_500/model_499.pt) |
| box_target_strong | [box_target_strong.ipynb](box_target_strong.ipynb) | [model_1000.pt](checkpoints/box_target_strong/model_1000.pt) |
| qpos_strong | [qpos_strong.ipynb](qpos_strong.ipynb) | [model_1000.pt](checkpoints/qpos_strong/model_1000.pt) |
| learning_rate_3e4 | [learning_rate_3e4.ipynb](learning_rate_3e4.ipynb) | [model_1000.pt](checkpoints/learning_rate_3e4/model_1000.pt) |

Блокноты раздельных опытов собраны из фактического кода и сохранённого вывода выполненных ячеек Colab; это не прямой экспорт всего исходного блокнота. Номера выполнения не восстановлены. Для повторения установку выполнять в свежей среде Colab с GPU, затем запускать только нужный опыт. Исходные записи вывода также сохранены отдельными TXT. Конфиги лежат рядом с финальными чекпоинтами; TensorBoard events — в events/<имя опыта>/. CSV и график воспроизводятся командой `python results/practice2/plot_learning.py` из корня проекта при установленном matplotlib.

## Повторение инференса

Откройте Docker Desktop. Для существующего контейнера:

```powershell
docker start mujoco-sim-cpu
docker exec -it -w /workspace mujoco-sim-cpu python3 inference.py --device cpu --checkpoint results/practice2/checkpoints/learning_rate_3e4/model_1000.pt --episodes 10
```

Путь можно заменить на любой чекпоинт из таблицы. Просмотр: http://localhost:6080/vnc.html. Команда не записывает видео. Инференс создаёт среду со стандартными весами награды: mean_reward инференса не подменяет mean_reward обучения, особенно для двух опытов с изменёнными весами.

Предварительные гипотезы сохранены в [плане изменения масштаба](max-iters-500-plan.md), [плане изменения награды](reward-experiments-plan.md), [плане изменения learning rate](learning-rate-3e4-plan.md). В сравнение включены фактические результаты; иллюстративные числа для 4096 сред из обсуждения не использованы.
