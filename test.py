import data_loader as d
import network
training_data, validation_data, test_data = d.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 2.3, test_data=test_data)
