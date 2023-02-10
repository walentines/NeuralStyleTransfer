import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy.io
import tensorflow as tf
import pprint
import imageio

class CONFIG:
    IMAGE_WIDTH = 300
    IMAGE_HEIGHT = 225
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    VGG_MODEL = 'pretrained_model/imagenet-vgg-verydeep-19.mat'
    STYLE_IMAGE = 'images/stone_style.jpg'
    CONTENT_IMAGE = 'images/content300.jpg'
    OUTPUT_DIR = 'output/'

def load_model(path):
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    
    def _weights(layer, expected_layer_name):
        Wb = vgg_layers[0][layer][0][0][2]
        W = Wb[0][0]
        b = Wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        
        return W, b
    
    def _relu(conv2d_layer):
        
        return tf.nn.relu(conv2d_layer)
    
    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, b.size))
        
        return tf.nn.conv2d(prev_layer, filters = W, strides = [1, 1, 1, 1], padding = 'SAME') + b
    
    def _conv2d_relu(prev_layer, layer, layer_name):
        
        return _relu(_conv2d(prev_layer, layer, layer_name))
    
    def _avgpool(prev_layer):
        
        return tf.nn.avg_pool(prev_layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    graph = {}
    graph['input'] = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph

def reshape_and_normalize_image(image):
    image = np.reshape(image, ((1,) + image.shape))
    image = image - CONFIG.MEANS

    return image

def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    noise_image = np.random.uniform(-20, 20, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')
    print(noise_image.shape, content_image.shape)
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image

pp = pprint.PrettyPrinter(indent = 4)
model = load_model('pretrained-model/imagenet-vgg-verydeep-19.mat')
pp.pprint(model)

content_image = imageio.imread('images/content300.jpg') 
plt.imshow(content_image)

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(a_C, shape = [1, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape = [1, -1, n_C])
    
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content

style_image = imageio.imread('images/stone_style.jpg')
plt.imshow(style_image)

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, shape = [n_H * n_W, n_C]), perm = [1, 0])
    a_G = tf.transpose(tf.reshape(a_G, shape = [n_W * n_H, n_C]), perm = [1, 0])
    
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    J_style_layer = 1 / (4 * n_C ** 2 * (n_H * n_W) ** 2) * tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(GS, GG))))
    
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        
        a_S = out
        a_G = out
        
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        
        J_style += coeff * J_style_layer
    
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = lambda: tf.add(tf.tensordot(alpha, J_content, axes = 0), tf.tensordot(beta, J_style, axes = 0))
    
    return J

content_image = imageio.imread('images/content300.jpg')
content_image = reshape_and_normalize_image(content_image)

style_image = imageio.imread('images/stone_style.jpg')
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
plt.imshow(generated_image[0])

model['input'].assign(content_image)
out = model['conv4_2']
a_C = model['conv4_2'].numpy()
a_G = out
J_content = compute_content_cost(a_C, a_G)
J_content = tf.Variable(J_content, trainable = True)   

model['input'].assign(style_image)
J_style = compute_style_cost(model, STYLE_LAYERS)
J_style = tf.Variable(J_style, trainable = True)
 
J = total_cost(J_content, J_style, alpha = 10.0, beta = 40.0)

optimizer = tf.keras.optimizers.Adam(2.0)
train_step = optimizer.minimize(J, tf.compat.v1.trainable_variables())

def model_nn(input_image, num_iterations = 200):

    for i in range(num_iterations):
        
        generated_image = model['input']
        
        Jt, Jc, Js = J, J_content, J_style
        print(Jt)
        print(Jc)
    
    return generated_image

generated_image = model_nn(generated_image)