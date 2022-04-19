import pathlib


def default_args(parser, checker):
    # general configs
    parser.add_argument('--seed', type=int, default=-1, help='seed for RNG, negative for no seed')

    # dataset configs
    parser.add_argument('--labeled_dataset', type=str, default='cifar10',
                        choices=('cifar10', 'cifar100', 'mnist', 'svhn', 'svhn_extra', 'custom'),
                        help='name of labeled dataset, set to custom for custom dataset')
    parser.add_argument('--labeled_mode', type=str, choices=("npy", "npy_folder"),
                        required=checker.get_requirement('--labeled_dataset custom'),
                        help='one of "npy", "npy_folder", must be provided when labeled_dataset is custom')
    # all custom dataset paths should be relative to ./custom_data/
    parser.add_argument('--labeled_train', type=pathlib.PurePath,
                        required=checker.get_requirement('--labeled_dataset custom'),
                        help='path to custom data features for training, '
                             '.npy file if mode is "npy", parent directory if mode is "npy_folder", '
                             'relative to working_dir/custom_data/')
    parser.add_argument('--labeled_train_y', type=pathlib.PurePath,
                        required=checker.get_requirement('--labeled_dataset custom', '--custom_mode npy'),
                        help='path to custom data targets(labels) for training, '
                             '.npy file if mode is "npy", '
                             'ignored if mode is "npy_folder" (labels implied), '
                             'relative to working_dir/custom_data/')
    parser.add_argument('--labeled_val', type=pathlib.PurePath,
                        help='path to custom data features for val, '
                             '.npy file if mode is "npy", parent directory if mode is "npy_folder", '
                             'relative to working_dir/custom_data/')
    parser.add_argument('--labeled_val_y', type=pathlib.PurePath,
                        required=checker.get_requirement('--labeled_mode npy',
                                                         '--labeled_val'),
                        help='path to custom data targets(labels) for val, '
                             '.npy file if mode is "npy", '
                             'ignored if mode is "npy_folder" (labels implied), '
                             'path relative to working_dir/custom_data/')
    # dataset configs
    parser.add_argument('--unlabeled_dataset', type=str, default='cifar10',
                        choices=('cifar10', 'cifar100', 'mnist', 'svhn', 'svhn_extra', 'custom'),
                        help='name of labeled dataset, set to custom for custom dataset')
    parser.add_argument('--unlabeled_mode', type=str, choices=("npy", "npy_folder"),
                        required=checker.get_requirement('--unlabeled_dataset custom'),
                        help='one of "npy", "npy_folder", must be provided when unlabeled_dataset is custom')
    # all custom dataset paths should be relative to ./custom_data/
    parser.add_argument('--unlabeled_train', type=pathlib.PurePath,
                        required=checker.get_requirement('--unlabeled_dataset custom'),
                        help='path to custom data features for training, '
                             '.npy file if mode is "npy", parent directory if mode is "npy_folder", '
                             'relative to working_dir/custom_data/')
    parser.add_argument('--unlabeled_val', type=pathlib.PurePath,
                        required=checker.get_requirement('--unlabeled_dataset custom', '--labeled_val'),
                        help='path to custom data features for val, '
                             '.npy file if mode is "npy", parent directory if mode is "npy_folder", '
                             'relative to working_dir/custom_data/')

    # model configs
    parser.add_argument('--model', type=str,
                        choices=('resnet', 'resnet_reduced',
                                 'resnetV2', 'resnetV2_reduced',
                                 'bottleneck_resnet', 'bottleneck_resnet_reduced',
                                 'bottleneck_resnetV2', 'bottleneck_resnetV2_reduced',
                                 'wide_resnet', 'wide_resnet_reduced',
                                 'pretrained'),
                        default='resnet_reduced',
                        help='model for training classifier, set to pretrained for loading pretrained model')
    parser.add_argument('--model_save_path', type=pathlib.PurePath,
                        help='path to save model after training')
    parser.add_argument('--pretrained_path', type=pathlib.PurePath,
                        required=checker.get_requirement('--model pretrained'),
                        help='path to pretrained model to load for classifier, '
                             'ignored when model is not set to pretrained, '
                             'path relative to working_dir/saved_classifiers/')
    # all below model configs will be ignored if set to pretrained
    parser.add_argument('--model_config', type=int, nargs='+', default=[3, 3, 3],
                        help='model configuration for each stage')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--model_width', type=float, default=0.25,
                        help='width multiplier applied on width of original architecture')

    # transforms / data augmentation
    parser.add_argument('--other_transforms', type=int, default=1,
                        help='allow all augmentations other than cutout, autoaug, normalize_input, and toTensor')
    parser.add_argument('--cutout', type=int, default=0)
    parser.add_argument('--autoaug', type=int, default=0)
    parser.add_argument('--normalize_input', type=int, default=1)

    # general training configs
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.1)
    parser.add_argument('--lr_schedule', type=str,
                        choices=('trades', 'trades_fixed', 'cosine', 'wrn'), default='cosine')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--val_freq', type=int, default=1, help='frequency to validate on validation set in epochs')
    parser.add_argument('--initial_epochs', type=int, default=1,
                        help='index of epochs to start with (for pretrained model)')
    parser.add_argument('--epochs', type=int, default=50, help='number of total epochs')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes excluding the shifted class')


def detector_args(parser, checker):
    # a placeholder for future implementation specific to training detector, has no effect at this time
    pass
