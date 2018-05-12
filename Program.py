import numpy as np
import cv2 as cv


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


def open_image(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    array = img

    for i in range(32):
        for k in range(32):
            if array[i, k] > 55:
                array[i, k] = 1
            else:
                array[i, k] = 0
    array = np.ndarray.astype(array, dtype=np.float64)
    array.shape = (1024, 1)
    for i in range(1024):
        array[i, 0] = (float(array[i, 0]) / 200)

    return array


def make_images_array():
    array = np.ndarray(shape=(3, total_images), dtype=np.ndarray)
    for i in range(total_images):
        path = "brands/renault/" + str(i) + ".jpg"
        array[0, i] = open_image(path)
    for i in range(total_images):
        path = "brands/chevrolet/" + str(i) + ".jpg"
        array[1, i] = open_image(path)
    for i in range(total_images):
        path = "brands/subaru/" + str(i) + ".jpg"
        array[2, i] = open_image(path)
    return array


class Network(object):

    def __init__(self, total_images):
        self.ai = np.ndarray(shape=(1024, 1))
        self.ah = np.ndarray(shape=(3, 1))
        self.ao = np.ndarray(shape=(3, 1))
        self.wi = np.random.randn(3, 1024)
        for brand in range(3):
            for i in range(1024):
                if self.wi[brand, i] < 0:
                    self.wi[brand, i] *= -1.0
        self.wo = np.random.randn(1, 3)
        for i in range(3):
            if self.wo[0, i] < 0:
                self.wo[0, i] *= -1
        self.target = [[60.0], [40.0], [20.0]]
        self.total_images = total_images
        self.success = 0

    def feed_forward(self, b, image):
        self.ai = image
        sum = np.dot(self.wi[b, :], self.ai)[0]
        self.ah[b, 0] = sigmoid(sum)
        self.ao[b, 0] = self.ah[b, 0] * self.wo[0, b]
        return self.ao

    def back_propagate(self, b, N):
        delta_out = np.ndarray(shape=(3, 1))
        # delta out calculating(derivative error function by "hidden to out" weights)
        delta_out[b, 0] = (self.target[b] - self.ao[b, 0]) * self.ah[b, 0]

        # derivative function with respect to "input-hidden" weights
        delta_hidden = np.ndarray(shape=(1024, 1))
        for i in range(1024):
            delta_hidden[i, 0] = (delta_out[b, 0]) * dsigmoid(self.ah[b, 0]) * self.ai[i, 0]

        # changing "hidden to out"
        self.wo[0, b] += N * delta_out[b, 0]
        # changing "input to hidden" weights
        for i in range(1024):
            self.wi[b, i] += N * delta_hidden[i, 0]

        # calculating error
        error = 0.5 * (self.target[b] - self.ao[b, 0]) ** 2
        return error

    def train(self, images, N=0.1):
        for brand in range(3):
            if brand == 0: print("===RENAULT TRAINING===")
            if brand == 1: print("===CHEVROLET TRAINING===")
            if brand == 2: print("===SUBARU TRAINING===")
            for i in range(1000):
                error = 0.0
                for image in range(total_images):
                    self.feed_forward(self, brand, images[brand, image])
                    error = self.back_propagate(self, brand, N)
                if i % 10 == 0:
                    print(str(i) + ": " + str(error))

    def export_weights_to_file(self):
        for brand in range(3):
            if brand == 0: name = "renault"
            if brand == 1: name = "chevrolet"
            if brand == 2: name = "subaru"
            file_weights_out = open(name + "_w_out.txt", 'w')
            file_weights_out.write(str(self.wo[0, brand]) + '\n')
            file_weights_out.close()

        for brand in range(3):
            if brand == 0: name = "renault"
            if brand == 1: name = "chevrolet"
            if brand == 2: name = "subaru"
            file_weights_hidden = open(name + "_w_hidden.txt", 'w')
            for i in range(1024):
                file_weights_hidden.write(str(self.wi[brand, i]) + '\n')
            file_weights_hidden.close()

    def import_weights_from_file(self):
        for brand in range(3):
            if brand == 0: name = "renault"
            if brand == 1: name = "chevrolet"
            if brand == 2: name = "subaru"
            file_weights_out = open(name + "_w_out.txt", 'r')
            self.wo[0, brand] = float(file_weights_out.readline())
            file_weights_out.close()
        for brand in range(3):
            if brand == 0: name = "renault"
            if brand == 1: name = "chevrolet"
            if brand == 2: name = "subaru"
            file_weights_hidden = open(name + "_w_hidden.txt", 'r')
            for i in range(1024):
                self.wi[brand, i] = float(file_weights_hidden.readline())
            file_weights_hidden.close()

    def recognize_one(self, image):
        self.ai = image
        for brand in range(3):
            sum = np.dot(self.wi[brand, :], self.ai)[0]
            self.ah[brand, 0] = sigmoid(sum)

        for brand in range(3):
            self.ao[brand, 0] = self.ah[brand, 0] * self.wo[0, brand]
        result = [[1.0], [1.0], [1.0]]

        for brand in range(3):
            result[brand] = np.abs(self.target[brand] - self.ao[brand, 0])
        if np.argmin(result) == 0: print("RENAULT")
        if np.argmin(result) == 1: print("CHEVROLET")
        if np.argmin(result) == 2: print("SUBARU")
        return np.argmin(result)

    def recognize_all(self):
        for brand in range(3):
            if brand == 0: print("===RECOGNIZING RENAULT===")
            if brand == 1: print("===RECOGNIZING CHEVROLET===")
            if brand == 2: print("===RECOGNIZING SUBARU===")
            for image in range(total_images):
                print("IMAGE#", image)
                if self.recognize_one(self, images[brand, image]) == brand:
                    self.success += 1
                print()
        print("Recognizing was successful in ",
              (self.success / (self.total_images * 3)) * 100, "%")


total_images = 100
images = make_images_array()
network = Network
network.__init__(network, total_images)
command = input("Do you want to train network?(y/n)")
if command == "y":
    network.train(network, images)
    network.export_weights_to_file(network)
else:
    network.import_weights_from_file(network)
network.recognize_all(network)
