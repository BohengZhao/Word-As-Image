import yaml
import os


def read_yaml(path):
    with open(path, 'r') as f:
        try:
            params = yaml.load(f, Loader=yaml.FullLoader)
            return params
        except yaml.YAMLError as exc:
            print(exc)


def write_yaml(path, data):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile)
        outfile.close()


if __name__ == '__main__':
    parameter = [
        {
            'ProjectName': 'FontClassifier',
            'RunName': 'Resnet50 with learnable CLIP, Case-insensitive Classification',
            'ProjectPath': '/scratch/gilbreth/zhao969/FontClassifier',
            'CheckpointPath': '/scratch/gilbreth/zhao969/FontClassifier/checkpoint',
            'NeedDataGeneration': False,
            'InputPath': '/scratch/gilbreth/zhao969/Fonts/**/*.ttf',
            'OutputPath': '/scratch/gilbreth/zhao969/FontClassifier/data/output',
            'use_wandb': True,
            'NumFonts': 1000,
            'learning_rate': 0.0001,
            'epochs': 100,
            'batch_size': 32
        },
        {
            'Number_of_Classes': 26,
            'train_CLIP': False
        }
    ]
    write_yaml(
        '/scratch/gilbreth/zhao969/FontClassifier/config/config.yaml', parameter)

    print(read_yaml(
        '/scratch/gilbreth/zhao969/FontClassifier/config/config.yaml'))
