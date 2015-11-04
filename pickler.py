import pickle


def getPickleObject(filename):
    file = open(filename, "rb")
    instance =  pickle.load(file)
    file.close()
    return instance

def dumpPickle(filename, instance):
    file = open(filename, "wb")
    pickle.dump(instance, file)
    file.close()