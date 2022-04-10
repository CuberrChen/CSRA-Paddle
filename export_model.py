import os
import argparse

import paddle

from models.resnet_csra import ResNet_CSRA

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam", default=0.1, type=float)
    parser.add_argument("--num_cls", default=20, type=int)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)

    return parser.parse_args()


def main(args):
    # model
    if args.model == "resnet101":
        net = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, depth=101,
                            pretrained=False)
    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        print('Loaded trained params of model successfully.')

    shape = [-1, 3, args.img_size, args.img_size]

    new_net = net

    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)
    print(f'Model is saved in {args.save_dir}.') # model.pdiparams|info|model


if __name__ == '__main__':
    args = parse_args()
    main(args)