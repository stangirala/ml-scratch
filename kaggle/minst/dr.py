train_file = 'data/train.csv'
test_file = 'data/test.csv'
weights = 'weight.txt'
results_file = 'results.txt'
k = 10
features = 784

def read_data_and_train():
    kmeans = [[0 for _ in xrange(features)] for _ in xrange(k)]
    points = [1 for _ in xrange(k)]

    with open(train_file, 'r') as ifile:
        next(ifile)
        for line in ifile:
            data = [int(i.strip()) for i in line.split(',')]
            points[data[0]] += 1
            for i in xrange(1, features+1):
                kmeans[data[0]][i-1] += data[i]

        for j in xrange(k):
            for i in xrange(features):
                kmeans[j][i] /= float(points[j])

        with open(weights, 'w') as ofile:
            for i in xrange(k):
                for j in xrange(features):
                    ofile.write(str(kmeans[i][j]) + ' ')
                ofile.write('\n')

def classify_test_data():
    kmeans = [[0 for _ in xrange(features)] for _ in xrange(k)]
    c = 0
    with open(weights, 'r') as ifile:
        for line in ifile:
            weight = [float(i.strip()) for i in line.split()]
            for j in xrange(features):
                kmeans[c][j] = weight[j]
            c += 1

    c = 1
    with open(test_file, 'r') as ifile, open(results_file, 'w') as ofile:
        ofile.write('ImageId,Label\n')
        next(ifile)
        for line in ifile:
            data = [int(i.strip()) for i in line.split(',')]
            dist = float('inf')
            index = -1
            for i in xrange(k):
                tdist = 0
                for j in xrange(features):
                    tdist += (kmeans[i][j] - data[j]) ** 2
                tdist = tdist ** 0.5
                if tdist < dist:
                    dist = tdist
                    index = i
            ofile.write(str(c)+','+str(index)+'\n')
            c += 1
