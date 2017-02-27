import pickle

def saveModel(X,filename='model.pkl'):
    fw = open(filename,'w')
    pickle.dump(X,fw)
    fw.close()

def loadModel(filename):
    fr = open(filename)
    return pickle.load(fr)
