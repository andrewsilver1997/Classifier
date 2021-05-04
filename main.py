import argparse
from solver import Solver
from data_loader import get_train_loader, get_test_loader


def main(config):
    if config.mode == 'train':
        loader = get_train_loader(config.batch_size, config.num_workers)
    elif config.mode == 'test':
        loader = get_test_loader(config.batch_size, config.num_workers)

    solver = Solver(config, loader)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--model', default='densenet169', choices=['resnet50',
                                                                   'alexnet',
                                                                   'densenet169'])

    # Training configuration
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--EPOCH', default=30)

    # Directories
    # TODO: Modify model save path here
    parser.add_argument('--model_save_path', default='classifier/models')

    # Steps
    parser.add_argument('--model_save_step', default=10)

    # Miscellaneous
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--num_workers', default=1)

    config = parser.parse_args()
    print(config)
    main(config)
