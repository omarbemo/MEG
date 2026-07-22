import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from multiprocessing import freeze_support
import torch
from pathlib import Path
from anomalib.data.utils import read_image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
import anomalib.models as models

torch.set_float32_matmul_precision("medium")

BACKBONE = "resnet18"
LAYERS = [f"layer{i}" for i in range(1, 4)]
INPUT_SIZE = (256, 256)

datamodule = Folder(
    root=Path.cwd() / "labeled_dataset",
    normal_dir="good",
    abnormal_dir=["badly_formed", "cleaning", "collision"],
    normal_split_ratio=0.2,
    image_size=INPUT_SIZE,
    train_batch_size=32,
    eval_batch_size=32,
    task=TaskType.CLASSIFICATION,
)

datamodule.setup()  # Split the data to train/val/test/prediction sets.
datamodule.prepare_data()  # Create train/val/test/predic dataloaders

from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode

SHARED_CALLBACKS = [
    MetricsConfigurationCallback(
        task=TaskType.CLASSIFICATION,
        image_metrics=["AUROC"],
    ),
    ModelCheckpoint(
        mode="max",
        monitor="image_AUROC",
    ),
    PostProcessingConfigurationCallback(
        normalization_method=NormalizationMethod.MIN_MAX,
        threshold_method=ThresholdMethod.ADAPTIVE,
    ),
    MinMaxNormalizationCallback(),
    # ExportCallback(
    #     input_size=(256, 256),
    #     dirpath=str(Path.cwd()),
    #     filename="model",
    #     export_mode=ExportMode.TORCH,
    # ),
]


def build(name: str, model):
    callbacks = SHARED_CALLBACKS.copy()
    callbacks.append(
        ExportCallback(
            input_size=INPUT_SIZE,
            dirpath=str(Path.cwd() / name),
            filename=name,
            export_mode=ExportMode.OPENVINO,
        )
    )

    trainer = Trainer(
        callbacks=callbacks,
        accelerator="cpu",
        auto_scale_batch_size=False,
        check_val_every_n_epoch=1,
        devices=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
        log_every_n_steps=4,
    )

    trainer.fit(model=model, datamodule=datamodule)
    results = trainer.test(model=model, datamodule=datamodule)
    return results


all_models = {
    "dfm": models.Dfm(backbone=BACKBONE, layer=LAYERS[0], input_size=INPUT_SIZE),
    "cflow": models.Cflow(input_size=INPUT_SIZE, backbone=BACKBONE, layers=LAYERS),
    "dfkde": models.Dfkde(backbone=BACKBONE, layers=LAYERS),
    "fastflow": models.Fastflow(input_size=INPUT_SIZE, backbone=BACKBONE),
    "padim": models.Padim(backbone=BACKBONE, input_size=INPUT_SIZE, layers=LAYERS),
    "patchcore": models.Patchcore(
        input_size=INPUT_SIZE, backbone=BACKBONE, layers=LAYERS
    ),
    "stfpm": models.Stfpm(input_size=INPUT_SIZE, backbone=BACKBONE, layers=LAYERS),
}

if __name__ == "__main__":
    freeze_support()

    # model_name: str = "cflow"
    # model_name: str = "dfkde"
    # model_name: str = "fastflow"
    # model_name: str = "padim"
    # model_name: str = "patchcore"
    # model_name: str = "stfpm"
    model_name: str = "dfm"
    # with open("results.json", "r", encoding="utf-8") as file:
    #     data = json.load(file)

    data = {}
    data[model_name] = build(model_name, all_models[model_name])

    # with open("results.json", "w", encoding="utf-8") as file:
    #     json.dump(data, file)
