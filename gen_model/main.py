import torch
import sys
import os

from torch import optim

from san_model.model.trainer import Trainer

# Aggiungi la directory superiore al percorso di ricerca dei moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gen_model.feat_extr.feature_extractor import JointFeatureLearningNetwork, FeatureExtractor
from gen_model.feat_extr.vgg import VGG
import tensorflow as tf
from san_model.model.data import CrossViewDataset, ImageTypes
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
# from a2seg import a2seg
# from seg2sa import seg2sa


def train_joint_feature_learner(device):
    dataset_path = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/"
    trainCSV = "/Volumes/SALVATORE R/Università/CV/hw_data/cvusa/CVUSA_subset/CVUSA_subset/train500-19zl.csv"

    batch_size = 8
    epochs = 30

    full_dataset = CrossViewDataset(trainCSV, base_path=dataset_path, device=device, normalize_imgs=True,
                                    dataset_content=[ImageTypes.Sat, ImageTypes.SyntheticSat,
                                                     ImageTypes.Ground])
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1])

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=int(0.1 * len(train_dataset)))
    valid_sampler = RandomSampler(validation_dataset, replacement=False,
                                  num_samples=int(0.05 * len(validation_dataset)))
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=valid_sampler)

    gen_model = FeatureExtractor(device)
    gen_model.load_weights(
        '/Volumes/SALVATORE R/Università/CV/hw_data/saved_models/models_gen/jfl_1717343102687.pt',
        '/Volumes/SALVATORE R/Università/CV/hw_data/saved_models/models_gen/ff_1717343102687.pt'
    )
    trainer = Trainer(gen_model, device=device)
    trainer.train(training_dataloader,
                  validation_dataloader,
                  epochs=epochs,
                  loss_function=gen_model.triplet_loss,
                  optimizer=optim.Adam,
                  learning_rate=10e-4, weight_decay=0.01)


def three_stream_joint_feat_learning(x_sat=None, x_grd=None, x_grd_gan=None, trainable=True):
    def fc_layer(x, input_dim, output_dim, init_dev, init_bias,
                 trainable, name='fc_layer', activation_fn=tf.nn.relu):
        weight = tf.Variable(initial_value=tf.zeros([input_dim, output_dim]), name='weights',
                             shape=[input_dim, output_dim], trainable=trainable,
                             initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=init_dev))
        bias = tf.Variable(initial_value=tf.zeros([output_dim]), name='biases', shape=[output_dim],
                           trainable=trainable, initializer=tf.constant_initializer(init_bias))

        if activation_fn is not None:
            out = tf.keras.backend.bias_add(tf.keras.backend.dot(x, weight), bias)
            out = activation_fn(out)
        else:
            out = tf.keras.backend.bias_add(tf.keras.backend.dot(x, weight), bias)

        return out

    x_grd = tf.random.uniform(shape=[10, 53760])  # ground img
    x_sat = tf.random.uniform(shape=[10, 43008])
    x_grd_gan = tf.random.uniform(shape=[10, 43008])

    fc_grd = fc_layer(x_grd, 53760, 1000, 0.005, 0.1, trainable, 'fc2', activation_fn=None)
    grd_global = tf.nn.l2_normalize(fc_grd, dim=1)

    fc_sat = fc_layer(x_sat, 43008, 1000, 0.005, 0.1, trainable,
                      'fc1', activation_fn=None)
    sat_global = tf.nn.l2_normalize(fc_sat, dim=1)

    fc_sat_synth = fc_layer(x_grd_gan, 43008, 1000, 0.005, 0.1,
                            trainable, 'fc1', activation_fn=None)
    grd_sat_gan = tf.nn.l2_normalize(fc_sat_synth, dim=1)

    return sat_global, grd_global, grd_sat_gan


def eight_layer_conv_multiscale(x=None, keep_prob=0.5, trainable=True, name=None):
    # x = x if x is not None else tf.random.uniform(shape=[10, 512, 512, 3])  # sat img
    x = tf.random.uniform(shape=[10, 224, 1232, 3])  # ground img

    # x = tf.random.uniform(shape=[10, 512, 512, 3])

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1],
                            padding='SAME')

    def conv_layer(x, kernel_dim, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        weight = tf.Variable(initial_value=tf.zeros([kernel_dim, kernel_dim, input_dim, output_dim]), name='weights',
                             shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                             trainable=trainable, initializer=tf.keras.initializers.GlorotUniform())
        bias = tf.Variable(initial_value=tf.zeros([output_dim]), name='biases', shape=[output_dim],
                           trainable=trainable, initializer=tf.keras.initializers.GlorotUniform())

        if activated:
            out = activation_function(conv2d(x, weight) + bias)
        else:
            out = conv2d(x, weight) + bias

        return out

    # layer 1: conv 3-64
    layer1_output = conv_layer(x, 4, 3, 64, trainable, True, 'conv1')

    # layer 2: conv 64 - 256
    layer2_output = conv_layer(layer1_output, 4, 64, 128, trainable, True, 'conv2')

    # layer 3: conv 256 - 512
    layer3_output = conv_layer(layer2_output, 4, 128, 256, trainable, True, 'conv3')

    # layer 4: conv 512 - 512
    layer4_output = conv_layer(layer3_output, 4, 256, 512, trainable, True, 'conv4')

    # layer 5: conv 512 - 512
    layer5_output = conv_layer(layer4_output, 4, 512, 512, trainable, True, 'conv5')

    # layer 6: conv 512 - 512
    layer6_output = conv_layer(layer5_output, 4, 512, 512, trainable, True, 'conv6')
    layer6_output = tf.nn.dropout(layer6_output, keep_prob, name='conv6_dropout')
    # layer6_b_output = tf.layers.flatten(layer6_output, 'reshape_feats_6')
    layer6_b_output = tf.keras.layers.Flatten(name='reshape_feats_6')(layer6_output)

    # layer 7: conv 512 - 512
    layer7_output = conv_layer(layer6_output, 4, 512, 512, trainable, True, 'conv7')
    layer7_output = tf.nn.dropout(layer7_output, keep_prob, name='conv7_dropout')
    # layer7_b_output = tf.layers.flatten(layer7_output, 'reshape_feats_7')
    layer7_b_output = tf.keras.layers.Flatten(name='reshape_feats_7')(layer7_output)

    # layer 8: conv 512 - 512
    layer8_output = conv_layer(layer7_output, 4, 512, 512, trainable, True, 'conv8')
    layer8_output = tf.nn.dropout(layer8_output, keep_prob, name='conv8_dropout')
    # layer8_b_output = tf.layers.flatten(layer8_output, 'reshape_feats_8')
    layer8_b_output = tf.keras.layers.Flatten(name='reshape_feats_8')(layer8_output)

    layer9_output = tf.concat([layer6_b_output, layer7_b_output, layer8_b_output], 1)

    return layer9_output


def test_network():
    sat_img = torch.rand(10, 3, 512, 512)
    grnd_img = torch.rand(10, 3, 224, 1232)  # if change size, change ground_padding
    sat_synth_img = torch.rand(10, 3, 512, 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vgg_model = VGG(device)
    # res = vgg_model(rand_img)

    res = FeatureExtractor(device)(grnd_img, sat_img, sat_synth_img)
    print()


def create_segmented_dataset(device):
    folder_path = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/bingmap/"
    segmented_images = a2seg(folder_path)
    return segmented_images


def create_dataset(device, folder_path, base_path):
    """ Create the dataset for the model. 
    Args:
        device: The device to run the model on.
        folder_path: The path to the dataset.
        base_path: The base path to the dataset.
    """

    batch_size = 8

    validation_dataset = CrossViewDataset(folder_path, base_path=base_path, device=device, normalize_imgs=True,
                                          dataset_content=[ImageTypes.SegmentedSat, ImageTypes.Sat,
                                                           ImageTypes.Ground])
    valid_sampler = SequentialSampler(validation_dataset)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        drop_last=True
    )
    return validation_dataloader


def synthetized_image(device):
    """ Generate synthetic images from segmented images. 
    Args: 
        device: The device to run the model on.
    """

    # folder_path_val = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/val-19zl.csv"
    folder_path_train = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/train-19zl.csv"

    base_path = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/"

    dataset = create_dataset(device, folder_path_train, base_path)

    # output_folder_val = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/my_synthetic_images_val/"
    output_folder_train = "D:/Università/CV/Ground-to-Aerial-Img-Matching/data/CVUSA_subset/my_synthetic_images_train/"

    prompt = "Aerial image of a segmented land area or cityscape"

    synt_images = seg2sa(device, dataset, output_folder_train, prompt)
    return synt_images


def main():
    # test_network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset = create_segmented_dataset(device)
    # synt_img = synthetized_image(device)
    # eight_layer_conv_multiscale()
    # three_stream_joint_feat_learning()

    train_joint_feature_learner(device)


if __name__ == '__main__':
    main()
