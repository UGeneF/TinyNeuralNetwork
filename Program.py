import numpy as np
import cv2 as cv


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


def open_image(path, brand):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    array = img

    array = cv.bilateralFilter(array, 7, 14, 14)
    array = brightnessing(array, brand, 160)
    array = align_brightness(array, brand)

    array = np.ndarray.astype(array, dtype=np.float64)
    if brand == 0:
        array = cut_leftright(array, 5)
    else:
        array = cut_updown(array, 8)
    array.shape = (1024, 1)
    for i in range(1024):
        array[i, 0] = (float(array[i, 0]) / 50000)
    return array


def cut_leftright(image, frame_px):
    for k in range(frame_px):
        for i in range(32):
            image[i, k] = 0
            image[i, 31 - k] = 0
    return image


def cut_updown(image, frame_px):
    for i in range(frame_px):
        for k in range(32):
            image[i, k] = 0
            image[31 - i, k] = 0
    return image


def gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    image = image / 255.0
    image = cv.pow(image, invGamma)
    return np.uint8(image * 255)


def get_average_brightness(image, brand):
    sum = 0.0
    if brand == 0:
        for i in range(32):
            for k in range(22):
                sum += image[i, k + 5]
        return float(sum / (32 * 22))
    else:
        for k in range(32):
            for i in range(16):
                sum += image[i + 8, k]
        return (float(sum / (16 * 22)))


def get_max_brightness(image, brand):
    max = 0.0
    if brand == 0:
        for i in range(32):
            for k in range(22):
                if image[i, k + 5] > max:
                    max = image[i, k + 5]
        return float(max)
    else:
        for k in range(32):
            for i in range(16):
                if image[i + 8, k] > max:
                    max = image[i + 8, k]
        return float(max)


def get_brightness(image, brand):
    return get_average_brightness(image, brand) / get_max_brightness(image, brand)


def align_brightness(image, brand):
    brightness = get_brightness(image, brand)
    if brightness > 0.65: return image
    while (np.abs(0.5 - brightness) > 0.1):
        if brightness < 0.5:
            image = gamma_correction(image, 1.1)
        else:
            image = gamma_correction(image, 0.9)
        brightness = (get_brightness(image, brand))
    return image


def brightnessing(image, brand, bright):
    if get_average_brightness(image, brand) > bright: return image
    while (get_average_brightness(image, brand) < bright):
        image = gamma_correction(image, 1.1)
    return image


def make_images_array():
    array = np.ndarray(shape=(3, total_images), dtype=np.ndarray)
    for i in range(total_images):
        path = "brands/renault/" + str(i) + ".jpg"
        array[0, i] = open_image(path, 0)
    for i in range(total_images):
        path = "brands/chevrolet/" + str(i) + ".jpg"
        array[1, i] = open_image(path, 1)
    for i in range(total_images):
        path = "brands/subaru/" + str(i) + ".jpg"
        array[2, i] = open_image(path, 2)
    return array


def result_analyse(result, right_answer):
    print("RIGHT ANSWER: ")
    if right_answer == 0: print("RENAULT")
    if right_answer == 1: print("CHEVROLET")
    if right_answer == 2: print("SUBARU")
    print("NETWORK ANSWER: ")
    if np.argmin(result) == 0: print("RENAULT")
    if np.argmin(result) == 1: print("CHEVROLET")
    if np.argmin(result) == 2: print("SUBARU")
    if right_answer == np.argmin(result):
        return 1
    else:
        return 0


def read_answers(amount_of_images):
    right_answers_file = open("answers.txt", 'r')
    right_answers = np.ndarray(shape=(amount_of_images, 1), dtype=np.int32)
    for k in range(amount_of_images):
        name = str(right_answers_file.readline())
        if name == "renault\n": right_answers[k, 0] = 0
        if name == "chevrolet\n": right_answers[k, 0] = 1
        if name == "subaru\n": right_answers[k, 0] = 2
    right_answers_file.close()
    return right_answers


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
        self.target = [[300.0], [200.0], [100.0]]
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
        delta_out[b, 0] = (self.target[b] - self.ao[b, 0])

        delta_hidden = np.ndarray(shape=(1024, 1))
        for i in range(1024):
            delta_hidden[i, 0] = (delta_out[b, 0]) * dsigmoid(self.ah[b, 0]) * self.ai[i, 0]

        self.wo[0, b] += N * delta_out[b, 0] * self.ah[b, 0]

        for i in range(1024):
            self.wi[b, i] += N * delta_hidden[i, 0]

        error = 0.5 * (self.target[b] - self.ao[b, 0]) ** 2
        return error

    def train(self, images, N=0.1):
        for brand in range(3):
            if brand == 0: print("===RENAULT TRAINING===")
            if brand == 1: print("===CHEVROLET TRAINING===")
            if brand == 2: print("===SUBARU TRAINING===")

            for i in range(1400):
                for image in range(total_images):
                    self.feed_forward(self, brand, images[brand, image])
                    error = self.back_propagate(self, brand, N)
                if i % 10 == 0:
                    print(str(i) + ": " + str(error))

    def export_weights_to_file(self):
        for brand in range(3):
            if brand == 0: name = "weights/renault"
            if brand == 1: name = "weights/chevrolet"
            if brand == 2: name = "weights/subaru"
            file_weights_out = open(name + "_w_out.txt", 'w')
            file_weights_out.write(str(self.wo[0, brand]) + '\n')
            file_weights_out.close()

        for brand in range(3):
            if brand == 0: name = "weights/renault"
            if brand == 1: name = "weights/chevrolet"
            if brand == 2: name = "weights/subaru"
            file_weights_hidden = open(name + "_w_hidden.txt", 'w')
            for i in range(1024):
                file_weights_hidden.write(str(self.wi[brand, i]) + '\n')
            file_weights_hidden.close()

    def import_weights_from_file(self):
        for brand in range(3):
            if brand == 0: name = "weights/renault"
            if brand == 1: name = "weights/chevrolet"
            if brand == 2: name = "weights/subaru"
            file_weights_out = open(name + "_w_out.txt", 'r')
            self.wo[0, brand] = float(file_weights_out.readline())
            file_weights_out.close()
        for brand in range(3):
            if brand == 0: name = "weights/renault"
            if brand == 1: name = "weights/chevrolet"
            if brand == 2: name = "weights/subaru"
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
        return np.argmin(result)

    def recognize_all(self):
        for brand in range(3):
            if brand == 0: print("===RECOGNIZING RENAULT===")
            if brand == 1: print("===RECOGNIZING CHEVROLET===")
            if brand == 2: print("===RECOGNIZING SUBARU===")
            for image in range(total_images):
                if self.recognize_one(self, images[brand, image]) == brand:
                    self.success += 1
                print()
        print("Recognizing the training base \nwas successful in ",
              (self.success / (self.total_images * 3)) * 100, "%")

    def make_test(self, amount_of_images):
        result = [[1.0], [1.0], [1.0]]
        self.success = 0
        right_answers = read_answers(amount_of_images)
        for i in range(amount_of_images):
            path = "test/" + str(i) + ".jpg"
            for brand in range(3):
                self.ai = open_image(path, brand)
                sum = np.dot(self.wi[brand, :], self.ai)[0]
                self.ah[brand, 0] = sigmoid(sum)
                self.ao[brand, 0] = self.ah[brand, 0] * self.wo[0, brand]
                result[brand] = np.abs(self.target[brand] - self.ao[brand, 0])
            print('\n', "IMAGE#", str(i))
            self.success += result_analyse(result, right_answers[i, 0])
        print("Recognizing was successful in ",
              (self.success / amount_of_images) * 100, "%")


total_images = 31
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
network.make_test(network, 30)
