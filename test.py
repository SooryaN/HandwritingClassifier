import data_loader as d
import network
from PIL import Image
training_data, validation_data, test_data = d.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 22, 10, 3.0, test_data=test_data)

def image_to_number(location):
    x=Image.open(location,'r').convert('L')
    x.size
    y=np.asarray(x.resize((28,28)).getdata(),dtype=np.float64).reshape((784,1))
    y = 255-y
    y = y/256
    return np.argmax(net.feedforward(y))
