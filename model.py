# Name : model.py
# Time : 2021/8/2 9:48
import numpy as np
epsilon = 1e-7


def initialize(layers_dims, L):
    """
    初始化
    :param layers_dims: [test_X.shape[0], 20, 3, 1] each layers's node's num, from input to output
    :param L: len(layer_dims)
    :return: para--[W1, b1, W2, b2, WL-1, bL-1]
    """

    para = []
    for i in range(1, L):  # 从第一层到第L-1层
        W = np.random.randn(layers_dims[i], layers_dims[i - 1]) * (2 / np.sqrt(layers_dims[i - 1]))
        b = np.zeros((layers_dims[i], 1))
        para.append(W)
        para.append(b)

    return para  # W1,b1,W2,b2,...,WL-1,bL-1


def linear_forward(A, para_part):
    """
    线性正向传播
    :param A: A_prev
    :param para_part: [Wi,Ai]
    :return: Zi
    """
    # Z1 = W1A0 + b1
    Z = np.dot(para_part[0], A) + para_part[1]
    return Z


def sigmoid(Z):
    """
    sigmoid函数
    :param Z: Zi
    :return: Ai
    """
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    """
    relu函数
    :param Z: Zi
    :return: Ai
    """
    # A = np.where(Z > 0, Z, 0)
    A = np.maximum(0, Z)
    return A


def forward_prop(X, para, L):
    """
    正向传播
    :param X: input
    :param para: [W1,b1,...,WL-1,bL-1]
    :param L: len(layers_dims)
    :return: cache -- [Z1, A1, ..., ZL-1, AL-1]
    """
    cache = []
    A = np.copy(X)
    for i in range(L - 2):  # 0 ~ L-3
        Z = linear_forward(A, para[2 * i:2 * (i + 1)])
        A = relu(Z)
        cache.append(Z)
        cache.append(A)
    Z = linear_forward(A, para[2 * (L - 2):2 * (L - 1)])
    A = sigmoid(Z)
    cache.append(Z)
    cache.append(A)

    return cache  # Z1,A1,...,Z4,A4


def forward_prop_drop(X, para, L, keep_prob):
    """
    随机失活的正向传播
    :param X: input
    :param para: [W1,b1,...,WL-1,bL-1]
    :param L: len(layers_dims)
    :param keep_prob: 随机失活的概率
    :return: cache -- [Z1,A1,D1, Z2,A2,D2, ..., ZL-1,AL-1,0]
                        最后一个0用来占位，使得计算更方便
    """

    cache = []
    A = np.copy(X)
    for i in range(L - 1):
        if i != 0:
            D = np.random.rand(A.shape[0], A.shape[1])
            assert (D.shape == A.shape)
            D = np.where(D > keep_prob, np.int64(0), np.int64(1))
            # cache.append(D)
            A = D * A / keep_prob
            cache.append(A)
            cache.append(D)
            if i == L - 2:
                break

        Z = linear_forward(A, para[2 * i:2 * (i + 1)])
        A = relu(Z)

        cache.append(Z)
        # cache.append(A)

    Z = linear_forward(A, para[2 * (L - 2):2 * (L - 1)])
    A = sigmoid(Z)
    cache.append(Z)
    cache.append(A)
    assert (len(cache) == 3 * L - 4)
    cache.append(0)
    return cache  # Z1,A1,D1, Z2,A2,D2, ..., ZL-1,AL-1,0


def relu_back(dA, Z):
    """
    relu反向传播求导
    :param dA: dAi
    :param Z: Zi
    :return: dZi
    """
    dZ = np.where(Z > 0, dA, np.int64(0))
    return dZ


def backward_prop(cache, para, X, Y, L):
    """
    反向传播
    :param cache: [Z1, A1, ..., ZL-1, AL-1]
    :param para: [W1,b1,...,WL-1,bL-1]
    :param X: input
    :param Y: label
    :param L: len(layers_dims)
    :return: grad -- [dW1,db1,...,dWL-1,dbL-1]
    """
    m = Y.shape[1]
    A = cache[-1]
    dA = 0
    grad_orig = []
    for i in range(L - 1, 0, -1):  # L-1 ~ 1
        if i == L - 1:
            dZ = (1 / m) * (A - Y)

        else:
            dZ = relu_back(dA, cache[2 * (i - 1)])  # Zi
        if 2 * i - 3 > 0:
            dW = np.dot(dZ, cache[2 * i - 3].T)  # Ai-1
        else:
            dW = np.dot(dZ, X.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(para[2 * (i - 1)].T, dZ)
        grad_orig.append(db)
        grad_orig.append(dW)

    grad = list(reversed(grad_orig))
    return grad


def backward_prop_re(cache, para, X, Y, L, lambd):
    """
    L2正则的反向传播
    :param cache: [Z1, A1, ..., ZL-1, AL-1]
    :param para: [W1,b1,...,WL-1,bL-1]
    :param X: input
    :param Y: label
    :param L: len(layers_dims)
    :param lambd: lambd参数
    :return: grad -- [dW1,db1,...,dWL-1,dbL-1]
    """
    m = Y.shape[1]
    A = cache[-1]
    dA = 0
    grad_orig = []
    for i in range(L - 1, 0, -1):  # L-1 ~ 1
        if i == L - 1:
            dZ = (1 / m) * (A - Y)

        else:
            dZ = relu_back(dA, cache[2 * (i - 1)])  # Zi
        if 2 * i - 3 > 0:
            dW = np.dot(dZ, cache[2 * i - 3].T) + (lambd / m) * para[2 * (i - 1)]  # Ai-1
        else:
            dW = np.dot(dZ, X.T) + (lambd / m) * para[2 * (i - 1)]
        db = np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(para[2 * (i - 1)].T, dZ)
        grad_orig.append(db)
        grad_orig.append(dW)

    grad = list(reversed(grad_orig))
    return grad


def backward_prop_drop(cache, para, X, Y, L, keep_prob):
    """
    随机失活反向传播
    :param cache: [Z1,A1,D1, Z2,A2,D2, ..., ZL-1,AL-1,0]
    :param para: [W1,b1,...,WL-1,bL-1]
    :param X: input
    :param Y: label
    :param L: len(layers_dims)
    :param keep_prob: 随机失活的概率
    :return: grad -- [dW1,db1,...,dWL-1,dbL-1]
    """
    m = Y.shape[1]
    A = cache[-2]
    dA = 0
    grad_orig = []
    for i in range(L - 1, 0, -1):  # L-1 ~ 1
        if i == L - 1:
            dZ = (1 / m) * (A - Y)

        else:
            dZ = relu_back(dA, cache[3 * (i - 1)])  # Zi
        if 2 * i - 3 > 0:
            dW = np.dot(dZ, cache[3 * i - 5].T)  # Ai-1
        else:
            dW = np.dot(dZ, X.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            dA = np.dot(para[2 * (i - 1)].T, dZ) * cache[3 * i - 4] / keep_prob # Di-1
        grad_orig.append(db)
        grad_orig.append(dW)

    grad = list(reversed(grad_orig))
    return grad


def update_para(para, grad, learning_rate):
    """
    更新参数
    :param para: [W1,b1,...,WL-1,bL-1]
    :param grad: [dW1,db1,...,dWL-1,dbL-1]
    :param learning_rate: 学习率
    :return: para -- new[W1,b1,...,WL-1,bL-1]
    """
    length = len(para)
    assert (length == len(grad))
    for i in range(length):
        assert (para[i].shape == grad[i].shape)
        para[i] = para[i] - learning_rate * grad[i]

    return para


def compute_cost(A, Y):
    """
    计算损失
    :param A: 最后一层输出
    :param Y:
    :return: J
    """
    m = Y.shape[1]
    A = np.where(A == 0, A + epsilon, A)
    A_new = np.where(A == 1, A - epsilon, A)
    J = (-1 / m) * (np.dot(Y, np.log(A_new).T) + np.dot(1 - Y, np.log(1 - A_new).T))
    return np.squeeze(J)


def compute_cost_re(A, Y, para, lambd):
    """
    L2正则的损失函数
    :param A: 最后一层输出
    :param Y: label
    :param para: [W1,b1,...,WL-1,bL-1]
    :param lambd: lambd参数
    :return: J
    """
    m = Y.shape[1]
    A = np.where(A == 0, A + epsilon, A)
    A_new = np.where(A == 1, A - epsilon, A)
    extra = 0
    for i in range((len(para) + 1) // 2):
        # extra += np.linalg.norm(para[2 * i]) ** 2
        extra += np.sum(np.square(para[2 * i]))
    J = (-1 / m) * (np.dot(Y, np.log(A_new).T) + np.dot(1 - Y, np.log(1 - A_new).T)) + (lambd / (2 * m)) * extra
    return np.squeeze(J)


def list_to_vec(para):
    """
    将矩阵列表形式的参数转换成一维向量形式的参数
    :param para: [W1,b1,...,WL-1,bL-1]
    :return: vec -- [w11,w12,...]
             shapes -- [W1.shape, b1.shape,...]
    """
    length = len(para)
    shapes = []
    for i in range(length):
        tmp = np.ravel(para[i])
        shapes.append(para[i].shape)
        if i == 0:
            vec = tmp
        else:
            vec = np.concatenate((vec, tmp), axis=0)

    vec = vec.reshape(vec.shape[0], 1)
    return vec, shapes


def vec_to_lst(vec, shapes):
    """
    将一维向量形式的参数转换为矩阵列表形式的参数
    :param vec: [w11,w12,...]
    :param shapes: [W1.shape, b1.shape,...]
    :return: para -- [W1,b1,...,WL-1,bL-1]
    """
    lst = []
    length = len(shapes)
    last, nxt = 0, 0
    for i in range(length):
        if len(shapes[i]) == 1:
            nxt = last + shapes[i][0]
        else:
            nxt = last + shapes[i][0] * shapes[i][1]
        tmp = vec[last:nxt].reshape(shapes[i])
        lst.append(tmp)
        last = nxt

    return lst


def gradient_check(grad, para, X, Y, L, epsilon=1e-7):
    """
    梯度测试
    :param grad: [dW1,db1,...,dWL-1,dbL-1]
    :param para: [W1,b1,...,WL-1,bL-1]
    :param X: input
    :param Y: label
    :param L: len(layers_dims)
    :param epsilon: 无穷小
    :return: diff - 原梯度和估测梯度之间的差距 应该小于epsilon
    """
    para_vec, shapes = list_to_vec(para)
    total_num = para_vec.shape[0]
    grad_approx = np.zeros((total_num, 1))
    grad_vec, _ = list_to_vec(grad)
    assert (para_vec.shape == grad_vec.shape)

    for i in range(total_num):
        theta_plus_vec = np.copy(para_vec)
        theta_plus_vec[i] += epsilon
        theta_plus_lst = vec_to_lst(theta_plus_vec, shapes)
        cache = forward_prop(X, theta_plus_lst, L)
        J_plus = compute_cost(cache[-1], Y)

        theta_minus_vec = np.copy(para_vec)
        theta_minus_vec[i] -= epsilon
        theta_minus_lst = vec_to_lst(theta_minus_vec, shapes)
        cache = forward_prop(X, theta_minus_lst, L)
        J_minus = compute_cost(cache[-1], Y)

        grad_approx[i] = (J_plus - J_minus) / (2 * epsilon)

    diff = np.linalg.norm(grad_vec - grad_approx) / (np.linalg.norm(grad_vec) + np.linalg.norm(grad_approx))

    return diff


def gradient_check_re(grad, para, X, Y, L, lambd, epsilon=1e-7):
    """
    L2正则的梯度测试
    :param grad: [dW1,db1,...,dWL-1,dbL-1]
    :param para: [W1,b1,...,WL-1,bL-1]
    :param X: input
    :param Y: label
    :param L: len(layers_dims)
    :param lambd: 参数lambd
    :param epsilon: 无穷小
    :return: diff - 原梯度和估测梯度之间的差距 应该小于epsilon
    """
    para_vec, shapes = list_to_vec(para)
    total_num = para_vec.shape[0]
    grad_approx = np.zeros((total_num, 1))
    grad_vec, _ = list_to_vec(grad)
    assert (para_vec.shape == grad_vec.shape)

    for i in range(total_num):
        theta_plus_vec = np.copy(para_vec)
        theta_plus_vec[i] += epsilon
        theta_plus_lst = vec_to_lst(theta_plus_vec, shapes)
        cache = forward_prop(X, theta_plus_lst, L)
        J_plus = compute_cost_re(cache[-1], Y, theta_plus_lst, lambd)

        theta_minus_vec = np.copy(para_vec)
        theta_minus_vec[i] -= epsilon
        theta_minus_lst = vec_to_lst(theta_minus_vec, shapes)
        cache = forward_prop(X, theta_minus_lst, L)
        J_minus = compute_cost_re(cache[-1], Y, theta_minus_lst, lambd)

        grad_approx[i] = (J_plus - J_minus) / (2 * epsilon)

    diff = np.linalg.norm(grad_vec - grad_approx) / (np.linalg.norm(grad_vec) + np.linalg.norm(grad_approx))

    return diff


def l_layer_model(X, Y, layers_dims, num_iterations=3000, learning_rate=0.0075, break_time=1000,
                  print_cost=True, grad_check=False):
    """
    普通l层神经网络模型
    :param X: input
    :param Y: label
    :param layers_dims: [test_X.shape[0], 20, 3, 1] each layers's node's num, from input to output
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param break_time: 中断进行梯度检测的时间
    :param print_cost: 是否输出损失
    :param grad_check: 是否进行梯度检测
    :return: para -- new[W1,b1,...,WL-1,bL-1]
    """
    L = len(layers_dims)  # 从输入层到输出层一共有L层
    para = initialize(layers_dims, L)
    costs = []
    for i in range(num_iterations):
        cache = forward_prop(X, para, L)
        grad = backward_prop(cache, para, X, Y, L)

        # 测试bp算法是否正确
        if i == break_time and grad_check:
            diff = gradient_check(grad, para, X, Y, L)
            if diff < 1e-7:
                print("BP success")
                print(diff)
            else:
                print("BP unsuccess")
                print(diff)
                raise ValueError("BP 算法执行错误，请检查")

        # 计算损失函数
        cost = compute_cost(cache[-1], Y)
        costs.append(cost)
        para = update_para(para, grad, learning_rate)
        if i % 1000 == 0 and print_cost:
            print(f'cost after {i} iterations:', cost)

    return para, costs


def l_layer_model_with_re(X, Y, layers_dims, num_iterations=3000, learning_rate=0.0075, break_time=1000,
                          lambd=0.7, print_cost=True, grad_check=False):
    """
    L2正则神经网络
    :param X: input
    :param Y: label
    :param layers_dims: [test_X.shape[0], 20, 3, 1] each layers's node's num, from input to output
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param break_time: 中断进行梯度检测的时间
    :param lambd: 参数lambd
    :param print_cost: 是否输出损失
    :param grad_check: 是否进行梯度检测
    :return: para -- new[W1,b1,...,WL-1,bL-1]
    """
    L = len(layers_dims)  # 从输入层到输出层一共有L层
    para = initialize(layers_dims, L)
    costs = []
    for i in range(num_iterations):
        cache = forward_prop(X, para, L)
        grad = backward_prop_re(cache, para, X, Y, L, lambd)

        # 测试bp算法是否正确
        if i == break_time and grad_check:
            diff = gradient_check_re(grad, para, X, Y, L, lambd)
            if diff < 1e-7:
                print("BP success")
                print(diff)
            else:
                print("BP unsuccess")
                print(diff)
                raise ValueError("BP 算法执行错误，请检查")

        # 计算损失函数
        cost = compute_cost_re(cache[-1], Y, para, lambd)
        costs.append(cost)
        para = update_para(para, grad, learning_rate)
        if i % 1000 == 0 and print_cost:
            print(f'cost after {i} iterations:', cost)

    return para, costs


def l_layer_model_dropout(X, Y, layers_dims, num_iterations=3000, learning_rate=0.0075,
                          print_cost=True, keep_prob=0.86):
    """
    随机失活神经网络
    :param X: input
    :param Y: label
    :param layers_dims: [test_X.shape[0], 20, 3, 1] each layers's node's num, from input to output
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param print_cost: 是否输出损失
    :param keep_prob: 随机失活概率
    :return: new[W1,b1,...,WL-1,bL-1]
    """
    L = len(layers_dims)  # 从输入层到输出层一共有L层
    para = initialize(layers_dims, L)
    costs = []
    for i in range(num_iterations):
        cache = forward_prop_drop(X, para, L, keep_prob)
        grad = backward_prop_drop(cache, para, X, Y, L, keep_prob)

        # 计算损失函数
        cost = compute_cost(cache[-2], Y)
        costs.append(cost)
        para = update_para(para, grad, learning_rate)
        if i % 1000 == 0 and print_cost:
            print(f'cost after {i} iterations:', cost)

    return para, costs


def predict(X_test, para, L):
    """
    预测
    :param X_test: 测试集
    :param para: [W1,b1,...,WL-1,bL-1]
    :param L: len(layers_dims)
    :return: y_hat 预测
    """
    cache = forward_prop(X_test, para, L)
    AL = cache[-1]
    Y_hat = np.where(AL > 0.5, 1, 0)
    return Y_hat


def compute_accuracy(Y_hat, Y):
    """
    计算准确度
    :param Y_hat:
    :param Y: label
    :return: accuracy 百分数
    """
    m = Y.shape[1]
    part1 = np.squeeze(np.dot(Y_hat, Y.T))
    part2 = np.squeeze(np.dot(1 - Y_hat, (1 - Y).T))
    acc = (part1 + part2) / m
    return str(acc * 100) + '%'
