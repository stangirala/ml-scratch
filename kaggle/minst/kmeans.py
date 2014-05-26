import matplotlib.pyplot as plt
import random

def train_kmeans(data, k=2):
    ''' k is the number of clusters.
        data is a list of lists.
        TODO: Use numpy. '''

    curr_means = [[0 for _ in xrange(0, len(data[0]))] for _ in xrange(0, k)]

    iterations = 0

    while True:
        iterations += 1
        #plt.plot([i for i in xrange(0, k)], curr_means, 'ro', [i for i in xrange(0, len(data))], data, 'bo')
        #plt.show()
        #raw_input('hit any key to continue')
        point_count  = [1 for _ in xrange(0, k)]
        new_means = [[0 for _ in xrange(0, len(data[0]))] for _ in xrange(0, k)]

        for i in xrange(0, len(data)):
            dist = float('inf')
            index = -1
            for l in xrange(0, k):
                tdist = 0
                for j in xrange(0, len(data[0])):
                    tdist += (data[i][j] - curr_means[l][j]) ** 2
                tdist = tdist ** 0.5
                if dist > tdist:
                    dist = tdist
                    index = l

            point_count[index] += 1
            for l in xrange(0, len(data[0])):
                new_means[index][l] += data[i][l]

        for i in xrange(0, k):
            for l in xrange(0, len(data[0])):
                new_means[i][l] /= point_count[i]

        state = True
        for i in xrange(0, len(data[0])):
            if (curr_means[i] != new_means[i]):
                state = False
                break
        if state == True:
            break
        curr_means = new_means

    return iterations, curr_means

def classify_kmeans(kmeans, data):
    ''' Returns list of cluster classifications.
        kmeans is a list of lists representing the
        cluster feature values and data is a list
        of lists. '''
    clusters = []
    for i in xrange(0, len(data)):
        index = -1
        dist = float('inf')
        for j in xrange(0, len(kmeans)):
            tdist = 0
            for k in xrange(len(kmeans[0])):
                tdist += (kmeans[j][k] - data[i][k]) ** 2
            tdist = tdist ** 0.5
            if tdist < dist:
                dist = tdist
                index = j
        clusters.append(index)

    return clusters

if __name__ == '__main__':
    data = []
    data_count = 500
    feature_size = 2
    k = 5
    partition = 4
    end = random.uniform(0, 100)
    for i in xrange(0, partition):
        end = random.uniform(0, end)
        for i in xrange(0, data_count/partition):
            data.append([random.uniform(0, end) for _ in xrange(0, feature_size)])

    iterations, kmeans = train_kmeans(data, k)
    print iterations, '\n', kmeans
    x_data = [data[i][0] for i in xrange(0, data_count)]
    y_data = [data[i][1] for i in xrange(0, data_count)]
    x_k = [kmeans[i][0] for i in xrange(0, k)]
    y_k = [kmeans[i][1] for i in xrange(0, k)]


    plt.plot(x_data, y_data,'ro', x_k, y_k, 'bo')
    plt.show()
