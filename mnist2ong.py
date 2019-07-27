import os
import cv2
import numpy as np
f = open("mnistTest.txt", "w")
for filename in os.listdir("mnist_png/testing"):
	print(filename)
	for img in os.listdir("mnist_png/testing/" + filename):

		f.write("mnist_png/testing/" + filename + "/"+img)
		f.write("\n")

f.close()


