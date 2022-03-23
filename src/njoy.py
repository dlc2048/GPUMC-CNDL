import numpy as np


def _joinCard(*args):
    return " " + " ".join(list(map(str, args))) + " /\n"


def _joinGroup(arr):
    out = ""
    i = 0
    for data in arr:
        out += " "
        out += "{:6e}".format(data)
        if i == 3:
            out += "\n"
            i = 0
            continue
        i += 1
    out += " /\n"
    return out


class NjoyInput:
    def __init__(self, file_name):
        self._file = open(file_name, mode="w")
        self.mat = None
        self.temperature = None
        self._custom_group = False

    def setEnv(self, mat, temperature):
        self.mat = mat
        self.temperature = temperature

    def setGroup(self, egn):
        self._egn = egn
        self._custom_group = True

    def moder(self, nin, nout):
        # header
        self._file.write("moder\n")
        # cards
        self._file.write(_joinCard(nin, nout))

    def reconr(self, nin, nout, err):
        # header
        self._file.write("reconr\n")
        # cards
        self._file.write(_joinCard(nin, nout))
        self._file.write(_joinCard("'pendf tape for" + str(self.mat) + "'"))
        self._file.write(_joinCard(self.mat))
        self._file.write(_joinCard(err))
        self._file.write(_joinCard(0))

    def broadr(self, nendf, nin, nout, err):
        # header
        self._file.write("broadr\n")
        # cards
        self._file.write(_joinCard(nendf, nin, nout))
        self._file.write(_joinCard(self.mat))
        self._file.write(_joinCard(err))
        self._file.write(_joinCard(self.temperature))
        self._file.write(_joinCard(0))

    def thermr(self, nendf, nin, nout, kernel_mat, iin, icoh, tol, emax):
        # header
        self._file.write("thermr\n")
        # cards
        self._file.write(_joinCard(nendf, nin, nout))
        self._file.write(_joinCard(kernel_mat, self.mat, 10, 1,
                                   iin, icoh, 0, 1, 221, 1))
        self._file.write(_joinCard(self.temperature))
        self._file.write(_joinCard(tol, emax))

    def groupr(self, nendf, npend, ngout2, ign, igg, iwt, lord, sigz):
        # header
        self._file.write("groupr\n")
        # cards
        self._file.write(_joinCard(nendf, npend, 0, ngout2))
        self._file.write(_joinCard(self.mat, ign, igg, iwt, lord))
        self._file.write(_joinCard("'group structure of " + str(self.mat) + "'"))
        self._file.write(_joinCard(self.temperature))
        self._file.write(_joinCard(sigz))
        if self._custom_group:
            self._file.write(_joinCard(len(self._egn) - 1))
            self._file.write(_joinGroup(self._egn))
        # target reactions
        self._file.write(_joinCard(3))
        self._file.write(_joinCard(3, 221))
        self._file.write(_joinCard(6))
        self._file.write(_joinCard(6, 221))
        for i in range(21, 27):
            self._file.write(_joinCard(i))
        self._file.write(_joinCard(16))
        for i in range(2):
            self._file.write(_joinCard(0))

    def moder(self, nin, nout):
        # header
        self._file.write("moder\n")
        # cards
        self._file.write(_joinCard(nin, nout))

    def stop(self):
        # header
        self._file.write("stop")
        
    def write(self):
        self._file.close()




if __name__ == '__main__':
    njoy = NjoyInput("input")
    njoy.setEnv(125, 293.6)
    njoy.moder(20, -21)
    njoy.reconr(-21, -22, 0.005)
    njoy.broadr(-21, -22, -23, 0.005)
    njoy.thermr(0, -23, -24, 1, 0, 0.05, 5)
    njoy.groupr(-21, -24, -30, 10, 6, 3, 6, 1e7)
    njoy.moder(-30, 31)
    njoy.stop()
    njoy.write()
