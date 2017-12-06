import os
import argparse


def launch_Depthtraining(**kwargs):

    # Launch training
    trainDepthMap.trainDepthMap(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DepthTrain model')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--nb_train_samples', default=1000, type=int, help="Number of training epochs")
    parser.add_argument('--nb_validation_samples', default=400, type=int, help="Number of validatio sample")
    parser.add_argument('--nb_epoch', default=40, type=int, help="Number  epoch")
    parser.add_argument('--lastLayerActivation', type=str, default='relu', help="Activation of the lastLayer")
    parser.add_argument('--PercentageOfTrianable', type=int, default=50, help="Percentage of Triantable Layers")
    parser.add_argument('--SpecificPathStr', type=str, default='Org', help="PathStr")
    parser.add_argument('--lossFunction', type=str, default="mean_absolute_error", help="Loss function")
    parser.add_argument('--bnAtTheend', type=str, default="True", help="have batch normaliation at the last layer?")
    args = parser.parse_args()

    # Set the backend by modifying the env variable
    os.environ["KERAS_BACKEND"] = "tensorflow"

    # Import the backend
    import keras.backend as K

    # manually set dim ordering otherwise it is not changed

    image_data_format = "channels_last"
    K.set_image_data_format(image_data_format)

    import trainDepthMap

    # Set default params
    d_params = {
                "batch_size": args.batch_size,
                "nb_train_samples": args.nb_train_samples,
                "nb_validation_samples": args.nb_validation_samples,
                "nb_epoch": args.nb_epoch,
                "model_name": "ResNet",
                "lastLayerActivation":args.lastLayerActivation,
                "PercentageOfTrianable":args.PercentageOfTrianable,
                "SpecificPathStr":args.SpecificPathStr,
                "bnAtTheend":args.bnAtTheend,
                "lossFunction":args.lossFunction

                }

    # Launch training
    launch_Depthtraining(**d_params)
