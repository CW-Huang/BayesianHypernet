import numpy
numpy.savetxt("data.txt", numpy.loadtxt(open("data.txt", "rb"), delimiter=",", skiprows=1))
