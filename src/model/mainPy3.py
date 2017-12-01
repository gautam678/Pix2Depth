import os
import argparse


def launch_training(**kwargs):

    # Launch training
    trainPy3.train(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('patch_size', type=int, nargs=2, action="store", help="Patch size for D")
    parser.add_argument('--backend', type=str, default="theano", help="theano or tensorflow")
    parser.add_argument('--generator', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="facades", help="facades")
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--epoch', default=10, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--nb_classes', default=2, type=int, help="Number of classes")
    parser.add_argument('--do_plot', action="store_true", help="Debugging plot")
    parser.add_argument('--bn_mode', default=2, type=int, help="Batch norm mode")
    parser.add_argument('--img_dim', default=64, type=int, help="Image width == height")
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('--use_label_smoothing', action="store_true", help="Whether to smooth the positive labels when training D")
    parser.add_argument('--label_flipping', default=0, type=float, help="Probability (0 to 1.) to flip the labels when training D")
    parser.add_argument('--lastLayerActivation', type=str, default='tanh', help="Activation of the lastLayer")
    parser.add_argument('--PercentageOfTrianable', type=int, default=70, help="Percentage of Triantable Layers")
    args = parser.parse_args()

    # Set the backend by modifying the env variable
    os.environ["KERAS_BACKEND"] = "tensorflow"

    # Import the backend
    import keras.backend as K

    # manually set dim ordering otherwise it is not changed

    image_data_format = "channels_last"
    K.set_image_data_format(image_data_format)

    import trainPy3

    # Set default params
    d_params = {"dset": args.dset,
                "generator": args.generator,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "nb_epoch": args.nb_epoch,
                "model_name": "ResNet",
                "epoch": args.epoch,
                "nb_classes": args.nb_classes,
                "do_plot": args.do_plot,
                "image_data_format": image_data_format,
                "bn_mode": args.bn_mode,
                "img_dim": args.img_dim,
                "use_label_smoothing": args.use_label_smoothing,
                "label_flipping": args.label_flipping,
                "patch_size": args.patch_size,
                "use_mbd": args.use_mbd,
                "lastLayerActivation":args.lastLayerActivation,
                "PercentageOfTrianable":args.PercentageOfTrianable

                }

    # Launch training
    launch_training(**d_params)
