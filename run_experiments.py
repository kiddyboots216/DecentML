import ray
from ray import tune

from experiment import Experiment

config = {
    "notes": "",
    "num_clients": 100,
    "model_type": 'perceptron',
    "dataset_type": 'iid',
    "fraction": 0.2,
    "max_rounds": 1000,
    "batch_size": 10,
    "epochs": 1,
    "learning_rate": tune.grid_search([]),
    "save_dir": './results_tune/',
    "goal_accuracy": 0.97,
    "lr_decay": 0.99,
}

def run_experiment(config, reporter):
    experiment = Experiment(config, reporter)
    experiment.run()

tune.register_trainable("run_experiment", run_experiment)
ray.init()

tune.run_experiments({
    "mlp_experiment": {
        "run": "run_experiment",
        "stop": { "val_accuracy": 0.97 },
        "config": config,
    }
})
