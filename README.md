# SR-SAC

Проект реализует continuous-control часть статьи `Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier`. Основная идея, которую здесь реализуем: `SR-SAC`, то есть `SAC` с высоким replay ratio и периодическими полными hard reset-ами параметров агента.

## Запуски

Один запуск:

```bash
python main.py --config configs/pendulum_rr32_srsac.json
```

Все выбранные эксперименты:

```bash
python run_selected_experiments.py
```

Быстрая локальная проверка:

```bash
python tools/smoke_test.py
```

Так же некоторые эксперименты есть в ipynb ноутбуках.

## Почему выбраны именно эти эксперименты

1. `DMC Pendulum Benchmark`
   Это одна из самых дешёвых сред из статьи, на ней уже на маленьком бюджете шагов видно, что high replay ratio начинает работать.

2. `Replay Ratio Scaling`
   Это главный тезис статьи: при наличии reset-ов можно поднимать replay ratio заметно выше обычного SAC.

3. `Reset Ablation`
   Самая важная локальная sanity-check проверка: при одинаковом replay ratio сравниваем вариант без reset-ов и с reset-ами.

4. `LunarLanderContinuous-v3`
   Эта среда обязательна по заданию и нужна как внешний тест на переносимость реализации вне DMC.

## Что сохраняется

Для каждого прогона в `runs/<run_name>_seed<seed>/` сохраняются:

- `evaluations.csv`
- `training_episodes.csv`
- `summary.json`
- `checkpoint.pt`
- `learning_curve.png`
- `final_episode.mp4`

После `python run_selected_experiments.py` дополнительно создаются:

- `runs/selected_runs_summary.csv`
- `runs/selected_experiment_table.csv`
- markdown-версии этих таблиц

## Сравнение со статьёй

В `article_refs.json` лежат reference-значения, заранее собранные из официального репозитория авторов:

- https://github.com/proceduralia/high_replay_ratio_continuous_control
- статья: https://openreview.net/forum?id=OpC-9aBBVJe

Это не полная реплика всех таблиц статьи, а компактный набор reference-метрик на том же бюджете шагов, который используется в этом проекте для дешёвого воспроизведения.
