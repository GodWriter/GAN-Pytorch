import argparse


def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "Pytorch implementation of Unit"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=100, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--dataset', type=str, default='apple2orange', help='name of the dataset')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=50, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads during batch generation')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--lambda_adv', type=float, default=10.0, help='adversarial loss weight')
    parser.add_argument('--lambda_KL1', type=float, default=0.1, help='encoded images loss weight')
    parser.add_argument('--lambda_id', type=float, default=100, help='identity loss weight')
    parser.add_argument('--lambda_KL2', type=float, default=0.1, help='encoded translated images loss weight')
    parser.add_argument('--lambda_cyc', type=float, default=100, help='cycle loss weight')
    parser.add_argument('--n_downsample', type=int, default=2, help='number downsampling layers in encoder')
    parser.add_argument('--n_upsample', type=int, default=2, help='number sampling layers in decoder')
    parser.add_argument('--dim', type=int, default=64, help='number of filters in first encoder layer')
    parser.add_argument('--sample_interval', type=int, default=100, help='interval between image samples')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='interval between saving models')
    parser.add_argument('--load_model', type=str, default='checkpoints/apple2orange/*_done.pth', help='model to load')
    parser.add_argument('--test_img', type=str, default='images/test.jpg', help='image to test')

    opt = parser.parse_args()
    print(opt)

    return opt
