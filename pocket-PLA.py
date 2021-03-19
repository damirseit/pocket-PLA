import numpy as np
import matplotlib.pyplot as plt
import random

N = 100

l1 = [N/2, random.randint(0, N/2)]
l2 = [0, random.randint(0, N/2)]
t = 1000


wbest = np.array([[1, 1, 1]])
werrin = N

def perceptron(x, y, w, T, wbest, werr):
  upd = 0
  j = 0
  while j < t:
    cnt = 0
    for i in range(x.shape[0]):
        j += 1
        if (y[i][0] > 0) != (np.matmul(np.reshape(w, (1, 3)),np.reshape(x[i], (3, 1)))[0] > 0):
            cnt = cnt + 1
            w = np.add(w, np.matmul(np.reshape(y[i], (1, 1)), np.reshape(x[i],(1, 3))))
            if error(x, y, w) < werr:
                wbest = w
                werr = error(x, y, w)
        upd = upd + 1
    if cnt == 0:
      break
  print("There were " + str(upd) + " updates before converging")
  return wbest

def error(x, y, w):
    if w[0][2] != 0:
        wlineA = [0, -(w[0][0] + w[0][1] * 0) / w[0][2]]
        wlineB = [N / 2, -(w[0][0] + w[0][1] * N / 2) / w[0][2]]
    else:
        wlineA = [-w[0][0]/w[0][1], 0]
        wlineB = [-w[0][0]/w[0][1], N/2]
    cnt = 0
    for i in range(x.shape[0]):
        diff = (wlineB[0] - wlineA[0])*(x[i][2]-wlineA[1]) - (wlineB[1] - wlineA[1])*(x[i][1] - wlineA[0])
        if diff > 0 and y[i][0] == -1:
            cnt += 1
        elif diff < 0 and y[i][0] == 1:
            cnt += 1

    return cnt





data = np.array([[1, 1, 3, 1]])


for i in range(N):
    a = random.randint(1, N/2)
    b = random.randint(1, N/2)
    d = (l2[0] - l1[0])*(b-l1[1]) - (l2[1] - l1[1])*(a - l1[0])
    if d == 0:
        i -= 1
        continue
    elif d > 0:
        if i%10 == 0:
            c = np.array([[1, a, b, 1]])
        else:
            c = np.array([[1, a, b, -1]])
    elif d < 0:
        if i%10 == 0:
            c = np.array([[1, a, b, -1]])
        else:
            c = np.array([[1, a, b, 1]])
    data = np.concatenate((data, c))


# for zero, x, y, label in data:
#   if label == -1:
#       plt.scatter(x, y, color = "#0000FF")
#   else:
#       plt.scatter(x, y, color = "#00FF00")
# plt.plot([l1[0], l2[0]], [l1[1], l2[1]],color="red", label = 'target function')
# plt.legend()
# plt.xlabel("x1")
# plt.ylabel("x2")

def listaverage(li):
    m = 0
    for i in range(len(li)):
        m += li[i]
    m = m / len(li)
    return m

x_data, y_data = np.split(data,[-1],axis=1)
w = np.array([[1, 1, 1]])

datatest = np.array([[1, 1, 3, 1]])


for i in range(N*10):
    a = random.randint(1, N/2)
    b = random.randint(1, N/2)
    d = (l2[0] - l1[0])*(b-l1[1]) - (l2[1] - l1[1])*(a - l1[0])
    if d == 0:
        i -= 1
        continue
    elif d > 0:
        if i%10 == 0:
            c = np.array([[1, a, b, 1]])
        else:
            c = np.array([[1, a, b, -1]])
    elif d < 0:
        if i%10 == 0:
            c = np.array([[1, a, b, -1]])
        else:
            c = np.array([[1, a, b, 1]])
    datatest = np.concatenate((data, c))

x_test, y_test = np.split(datatest,[-1],axis=1)

errinw = []
errinwave = []
errinbest = []
errinbestave = []
erroutw = []
erroutbest = []
time = []

upd = 0
j = 0
while j <= t:
    cnt = 0
    for i in range(x_data.shape[0]):
        if j > t:
            break
        j += 1
        if (y_data[i][0] > 0) != (np.matmul(np.reshape(w, (1, 3)),np.reshape(x_data[i], (3, 1)))[0] > 0):
            cnt = cnt + 1
            w = np.add(w, np.matmul(np.reshape(y_data[i], (1, 1)), np.reshape(x_data[i],(1, 3))))

        errinw.append(error(x_data, y_data, w))
        errinbest.append(error(x_data, y_data, wbest))
        erroutw.append(error(x_test, y_test, w))
        erroutbest.append(error(x_test, y_test, wbest))
        errinwave.append(listaverage(errinw))
        errinbestave.append(listaverage(errinbest))
        time.append(j)
        if error(x_data, y_data, w) < error(x_data, y_data, wbest):
            wbest = w
        upd = upd + 1
    if cnt == 0:
      break



plt.plot(time, errinwave, label = 'w(t) error')
plt.plot(time, errinbestave, label = 'w* error')
plt.legend()
plt.xlabel('t')
plt.ylabel('error')

plt.show()

