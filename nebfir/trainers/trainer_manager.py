from .base_trainer import BaseTrainer



def get_trainer(trainer_key: str):
    trainers = {
        "base_dataloader": BaseTrainer,
    }
    assert trainer_key in list(trainers.keys()), f"Unknown trainer {trainer_key}, choose from [ {list(trainers.keys())} ]"
    return trainers[trainer_key]

