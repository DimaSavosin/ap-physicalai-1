# Параметры обучения PandaPickCube

Таблица по уроку 2 практики 2. Столбцы соответствуют шаблону `find-the-parameters.xlsx` из урока.

| Параметр | Файл / способ доступа | Имя ключа / флага | Значение по умолчанию |
|---|---|---|---|
| num_envs | `train.py`: `parse_args()`, затем `device_utils.py`: `default_num_envs()` | `--num_envs` | Аргумент `None`, фактически 4096 при CUDA, 64 на CPU/MPS |
| learning_rate | `config.py`: `manipulation_params.rsl_rl_config("PandaPickCube").to_dict()` | `cfg_dict["algorithm"]["learning_rate"]` | 0.001 — подтверждено выводом установленной библиотеки из Docker |
| max_iters | `train.py`: `parse_args()` | `--max_iters` | 1001 |
| gripper_box | `registry.get_default_config("PandaPickCube")`; переопределение в `train.py` перед `registry.load(...)` | `env_cfg.reward_config.scales.gripper_box` | 4.0 — подтверждено выводом установленной библиотеки из Docker |

Проверенные в Docker веса остальных компонент: `box_target=8.0`, `no_floor_collision=0.25`, `robot_target_qpos=0.3`. В отдельных опытах изменяли box_target и robot_target_qpos; точные изменения и результаты указаны в comparison.md.

Новый запуск `baseline` выполнен в Colab с явным `NUM_ENVS=512`, `MAX_ITERS=1001`, `SEED=1`, `SAVE_INTERVAL=50`. Это отклонение от дефолтного GPU-масштаба 4096, которое необходимо учитывать в сравнении.
