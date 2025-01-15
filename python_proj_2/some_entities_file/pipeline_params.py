import yaml
from pydantic import BaseModel
from .data_params import DataParams
from .train_params import TrainParams

from marshmallow_dataclass import class_schema

# class TrainParams(BaseModel):
#     train_data: torch.Tensor
#     epochs: int
#     optimizer: torch.optim
#     loss_function: type
#     batch_size: int = 1
#
#
# class DataParams(BaseModel):
#     shape: tuple
#     data_type: float


class PipelineParams(BaseModel):
    train_params: TrainParams
    data_params: DataParams
    random_state: int


PipelineParamsSchema = class_schema(PipelineParams)


def read_pipeline_params(path: str) -> PipelineParams:
    with open(path, 'r') as input_steam:
        schema = PipelineParamsSchema()
        return schema.load(yaml.safe_load(input_steam))

