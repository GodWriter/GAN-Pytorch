import argparse


def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "Pytorch implementation of StarGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=100, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--dataset', type=str, default='celeba', help='name of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=80, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads during batch generation')
    parser.add_argument('--img_height', type=int, default=128, help='size of image height')
    parser.add_argument('--img_width', type=int, default=128, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--residual_blocks', type=int, default=6, help='number of residual blocks in generator')
    parser.add_argument('--lambda_cls', type=float, default=1.0, help='classification loss weight')
    parser.add_argument('--lambda_rec', type=float, default=10.0, help='reconstruction loss weight')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='gp loss weight')
    parser.add_argument('--sample_interval', type=int, default=100, help='interval between image samples')
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help='interval between saving models')
    parser.add_argument('--load_model', type=str, default='checkpoints/apple2orange/*_done.pth', help='model to load')
    parser.add_argument('--test_img', type=str, default='images/test.jpg', help='image to test')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training iterations for WGAN discriminator')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"])
    opt = parser.parse_args()
    print(opt)

    return opt
