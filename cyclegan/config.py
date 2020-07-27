import argparse


def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "Pytorch implementation of Pix2Pix"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=200, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dataset', type=str, default='facades', help='name of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads during batch generation')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--lambda_cyc', type=float, default=10.0, help='cycle loss weight')
    parser.add_argument('--lambda_id', type=float, default=5.0, help='identity loss weight')
    parser.add_argument('--num_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
    parser.add_argument('--sample_interval', type=int, default=100, help='interval between image samples')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between saving models')
    parser.add_argument('--load_model', type=str, default='checkpoints/generator_done.pth', help='model to load')
    parser.add_argument('--test_img', type=str, default='images/test.jpg', help='image to test')

    opt = parser.parse_args()
    print(opt)

    return opt
