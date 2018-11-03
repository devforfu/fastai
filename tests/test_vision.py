import pytest
from fastai import *
from fastai.vision import *
from torchvision.models import squeezenet1_1, resnet18

@pytest.fixture(scope="module")
def path():
    path = untar_data(URLs.MNIST_TINY)
    return path

def test_path_can_be_str_type(path):
    assert ImageDataBunch.from_csv(str(path))

def test_create_cnn_supports_squeeze_net(path):
    data = ImageDataBunch.from_folder(path)
    # learn = create_cnn(data, resnet18)
    learn = create_cnn(data, squeezenet1_1)
    learn.fit_one_cycle(5)
