from math import sqrt
import random  as rand
import numpy as np

def get_nearest_neighbors(x,X):
    distances = []

    for j in X:
        distances.append([x,j,euclidean_distance(x,j)])
    distances.sort(key=lambda x: x[2])
    distances.remove(distances[0])

    return distances



# (12,13) (18,10)

# (x,y,z,h,g)
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2

    return sqrt(distance)


def find_linear_equation(x1,x2):
    direction = np.subtract(x2, x1)
    return x1, direction

def smote(T, classes, N=3, K=3):
    minority = T[classes == 1]

    majority = T[classes == 0]

    oversample_minority=[]
    while N!=0:

        random_minority_index = np.random.choice(len(minority))

        random_minority_selected=T[random_minority_index];
        neighbors = get_nearest_neighbors(minority[random_minority_index],minority)

        nearest_neighbor=neighbors[rand.randint(0,K-1)]

        random_gap= np.random.random()
        while random_gap==0:
            random_gap=np.random.random()

        point, direction = find_linear_equation(random_minority_selected, nearest_neighbor[1])
        temp= direction*random_gap
        new_point = point +temp
        oversample_minority.append(np.append(new_point, 1))
        N = N - 1
    return oversample_minority

x = np.array([[4, 21,3], [5, 19,4], [10, 24,5], [4, 17,9], [3, 16,1], [11, 25,12], [14, 24,5], [8, 22,0], [10, 21,1],
              [12, 21,19], [13, 18,11], [15, 19,220], [17, 11,110]])
classes = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0])
oversample_minority=smote(x, classes)
print(oversample_minority)