from mainDenseDeepFusion import runFusion
import sys

def main ():
    runtypecode = sys.argv[1]
    database = sys.argv[2]
    gpudevice = sys.argv[3]

    print(runtypecode)
    if runtypecode == "RunFCFull":
        print("Run the whole FC system with no skips")
        startRun_FC_Full(database, gpudevice)
    if runtypecode == "RunFCBasics":
        print("Run only the basic FC combinations")
        startRun_FC_Basics(database, gpudevice)
    if runtypecode == "RunFCCombos":
        print("Run the best predicted FC combos")
        startRun_FC_Combos(database, gpudevice)
    if runtypecode == "RunAttnCombos":
        print("Run Attention Layer + best FC results")
        startRun_ATTN_Combos(database, gpudevice)
    if runtypecode == "RunConvCombos":
        print("Run Convolutional Layer + best FC results")
        startRun_CONV_Combos(database, gpudevice)

    print("END OF PROGRAM")

def startRun_CONV_Combos(database, gpudevice):
    if database == "0":
        bn_active = 1
        no_layers = 20
        no_neurons = 25

        for no_filters in [1, 5, 10]:
            resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice, 0, no_filters)
            resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice, 0, no_filters)
            fname = "/home/workspace/FinalResults/resultsConv" + database + ".csv"
            f = open(fname, "a+")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25,0," + str(no_filters) + "," + str(resultsRSKF75) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25,0," + str(no_filters) + "," + str(resultsRSKF75) + "\n")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50,0," + str(no_filters) + "," + str(resultsRSKF50) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50,0," + str(no_filters) + "," + str(resultsRSKF50) + "\n")
            f.close()
    if database == "1":
        bn_active = 0
        no_layers = 25
        no_neurons = 100

        for no_filters in [1, 5, 10]:
            resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice, 0, no_filters)
            resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice, 0, no_filters)
            fname = "/home/workspace/FinalResults/resultsConv" + database + ".csv"
            f = open(fname, "a+")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25,0," + str(no_filters) + "," + str(resultsRSKF75) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25,0," + str(no_filters) + "," + str(resultsRSKF75) + "\n")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50,0," + str(no_filters) + "," + str(resultsRSKF50) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50,0," + str(no_filters) + "," + str(resultsRSKF50) + "\n")
            f.close()
    if database == "2":
        bn_active = 0
        no_layers = 5
        no_neurons = 500

        for no_filters in [1, 5, 10]:
            resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice, 0, no_filters)
            resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice, 0, no_filters)
            fname = "/home/workspace/FinalResults/resultsConv" + database + ".csv"
            f = open(fname, "a+")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25,0," + str(no_filters) + "," + str(resultsRSKF75) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25,0," + str(no_filters) + "," + str(resultsRSKF75) + "\n")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50,0," + str(no_filters) + "," + str(resultsRSKF50) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50,0," + str(no_filters) + "," + str(resultsRSKF50) + "\n")
            f.close()

def startRun_ATTN_Combos(database, gpudevice):

    if database == "0":
        bn_active = 1
        no_layers = 25
        no_neurons = 2000

        resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice, 1)
        resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice, 1)
        fname = "/home/workspace/FinalResults/resultsAttn" + database + ".csv"
        f = open(fname, "a+")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.close()
    if database == "1":
        bn_active = 0
        no_layers = 10
        no_neurons = 1000

        resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice, 1)
        resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice, 1)
        fname = "/home/workspace/FinalResults/resultsAttn" + database + ".csv"
        f = open(fname, "a+")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.close()
    if database == "2":
        bn_active = 0
        no_layers = 5
        no_neurons = 500

        resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice, 1)
        resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice, 1)
        fname = "/home/workspace/FinalResults/resultsAttn" + database + ".csv"
        f = open(fname, "a+")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.close()

def startRun_FC_Combos(database, gpudevice):

    if database == "0":
        bn_active = 1
        no_layers = 25
        no_neurons = 2000

        resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice)
        resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice)
        fname = "/home/workspace/FinalResults/resultsDense" + database + ".csv"
        f = open(fname, "a+")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.close()

    if database == "1":
        bn_active = 0
        no_layers = 10
        no_neurons = 1000

        resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice)
        resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice)
        fname = "/home/workspace/FinalResults/resultsDense" + database + ".csv"
        f = open(fname, "a+")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.close()

    if database == "2":
        bn_active = 0
        no_layers = 5
        no_neurons = 500

        resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice)
        resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice)
        fname = "/home/workspace/FinalResults/resultsDense" + database + ".csv"
        f = open(fname, "a+")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
        print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
        f.close()

def startRun_FC_Basics(database, gpudevice) :
    no_layers = 5
    for no_neurons in [25, 50, 500, 1000, 2000, 5000]:
        for bn_active in [0, 1]:
            resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice)
            resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice)

            fname = "/home/workspace/FinalResults/resultsDense" + database + ".csv"
            f = open(fname, "a+")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
            f.close()

    no_neurons = 25
    for no_layers in [5, 10, 15, 20, 25]:
        for bn_active in [0, 1]:
            resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice)
            resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice)

            fname = "/home/workspace/FinalResults/resultsDense" + database + ".csv"
            f = open(fname, "a+")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
            print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
            f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
            f.close()

def startRun_FC_Full(database, gpudevice) :
    for no_layers in [5, 10, 15, 20, 25]:
        for no_neurons in [25, 50, 500, 1000, 2000, 5000]:
            for bn_active in [0, 1]:
                resultsRSKF75 = runFusion(no_layers, no_neurons, int(database), bn_active, 4, 25, gpudevice)
                resultsRSKF50 = runFusion(no_layers, no_neurons, int(database), bn_active, 2, 50, gpudevice)


                fname = "/home/workspace/FinalResults/resultsDense" + database + ".csv"
                f = open(fname, "a+")
                print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
                f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",4,25," + str(resultsRSKF75) + "\n")
                print(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
                f.write(database + "," + str(no_layers) + "," + str(no_neurons) + "," + str(bn_active) + ",2,50," + str(resultsRSKF50) + "\n")
                f.close()

if __name__ == '__main__':
    main()