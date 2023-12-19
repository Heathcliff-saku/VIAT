import numpy as np
from evaluate import comput_fitness
from rendering_image import render_image
from classifier.predict import test_baseline
from tqdm import tqdm
from datasets.opts import get_opts
import joblib
import torch
import time
np.set_printoptions(precision=4,  linewidth=100, suppress=True)
np.random.seed(0)


def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def compute_normalize(x):
  mean = np.mean(x)
  var = np.var(x)
  y = x-mean/var
  return y



def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
  def __init__(self, pi, epsilon=1e-08):
    self.pi = pi
    self.dim = pi.num_params
    self.epsilon = epsilon
    self.t = 0

  def update(self, globalg):
    self.t += 1
    step = self._compute_step(globalg)
    theta = self.pi.mu
    ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
    self.pi.mu = theta + step
    return ratio

  def _compute_step(self, globalg):
    raise NotImplementedError


class BasicSGD(Optimizer):
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  def _compute_step(self, globalg):
    step = -self.stepsize * globalg
    return step

class SGD(Optimizer):
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v = np.zeros(self.dim, dtype=np.float32)
    self.stepsize, self.momentum = stepsize, momentum

  def _compute_step(self, globalg):
    self.v = self.momentum * self.v + (1. - self.momentum) * globalg
    step = -self.stepsize * self.v
    return step


class Adam(Optimizer):
  def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

  def _compute_step(self, globalg):
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step


class PEPG:
  '''Extension of PEPG with bells and whistles.'''

  def __init__(self, num_params,  # number of model parameters
               num_k=5,  # 混合高斯的分量数
               sigma_init=0.10,  # initial standard deviation
               sigma_alpha=0.20,  # learning rate for standard deviation
               sigma_decay=0.999,  # anneal standard deviation
               sigma_limit=0.01,  # stop annealing if less than this
               sigma_max_change=0.2,  # clips adaptive sigma to 20%
               sigma_min=0.05,  # 允许的最小sigma
               sigma_update=True,
               omiga_max_change=0.2,
               omiga_decay=0.999,
               omiga_alpha=0.020,
               learning_rate=0.01,  # learning rate for standard deviation
               learning_rate_decay=0.9999,  # annealing the learning rate
               learning_rate_limit=0.01,  # stop annealing learning rate
               elite_ratio=0,  # if > 0, then ignore learning_rate
               popsize=256,  # population size
               average_baseline=True,  # set baseline to average of batch
               weight_decay=0.01,  # weight decay coefficient
               rank_fitness=True,  # use rank rather than fitness numbers
               forget_best=True,
               mu_lambda=0.0001,
               sigma_lambda=0.0001,
               omiga_lamba=0.0001, # don't keep the historical best solution
               update_omiga=False,
               random_begin=True):

    self.num_params = num_params
    self.num_k = num_k
    self.sigma_init = sigma_init
    self.sigma_alpha = sigma_alpha
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.sigma_max_change = sigma_max_change

    self.omiga_decay = omiga_decay
    self.omiga_alpha = omiga_alpha
    self.omiga_max_change = omiga_max_change

    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    self.average_baseline = average_baseline
    self.sigma_update = sigma_update
    self.sigma_min = sigma_min
    self.mu_lamba = mu_lambda
    self.sigma_lamba = sigma_lambda
    self.omiga_lamba = omiga_lamba


    if self.average_baseline:
      assert (self.popsize % 2 == 0), "Population size must be even"
      self.batch_size = int(self.popsize / 2)
    else:
      assert (self.popsize & 1), "Population size must be odd"
      self.batch_size = int((self.popsize - 1) / 2)

    # option to use greedy es method to select next mu, rather than using drift param
    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.use_elite = False
    if self.elite_popsize > 0:
      self.use_elite = True

    self.forget_best = forget_best
    self.batch_reward = np.zeros(self.batch_size * 2)

# -----------------------------------------------------------------------------------
    if not random_begin:
      self.mu = np.zeros([self.num_k, self.num_params])
    else:
      self.mu = 2.0*np.random.random_sample([self.num_k, self.num_params])-1.0

    self.sigma = np.ones([self.num_k, self.num_params]) * self.sigma_init
    self.omiga = 1/self.num_k * np.ones([self.num_k, self.num_params])

    self.curr_best_mu = np.zeros([self.num_k, self.num_params])
    self.best_mu = np.zeros([self.num_k, self.num_params])
# -----------------------------------------------------------------------------------

    self.best_reward = 0
    self.first_interation = True
    self.weight_decay = weight_decay
    self.rank_fitness = rank_fitness
    if self.rank_fitness:
      self.forget_best = True  # always forget the best one if we rank
    # choose optimizer
    self.optimizer = SGD(self, learning_rate)

    self.update_omiga = update_omiga

  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma * sigma))

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    # self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
    # self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
    # if self.average_baseline:
    #   epsilon = self.epsilon_full
    # else:
    #   # first population is mu, then positive epsilon, then negative epsilon
    #   epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
    #   self.r = epsilon/self.sigma.reshape(1, self.num_params)
    # solutions = self.mu.reshape(1, self.num_params) + epsilon
    # self.solutions = solutions
    # return solutions

    self.r = np.random.randn(self.batch_size, self.num_params)  # N*6 的 r
    self.r_full = np.concatenate([self.r, - self.r])
    if self.average_baseline:
        r = self.r_full
    else:
     # first population is every mu, then positive epsilon, then negative epsilon
        r = np.concatenate([np.zeros((self.num_k, self.num_params)), self.r_full])
        self.r_full = r

    self.F = np.zeros([self.batch_size, self.num_params])
    for i in range(self.num_params):
        F = np.random.choice(a=np.arange(self.num_k), size=self.batch_size, replace=True, p=self.omiga[:, i])
        self.F[:, i] = F
    self.F_full = np.concatenate([self.F, self.F])

    if self.average_baseline:
        F = self.F_full
    else:
     # first population is every mu, then positive epsilon, then negative epsilon
        l = np.linspace(0, self.num_k-1, num=self.num_k).repeat(self.num_params, axis=0).reshape((self.num_k, -1))
        F = np.concatenate([l, self.F_full])
        self.F_full = F

    # 依照各个分量得到solution
    self.solutions = np.zeros([self.batch_size*2+self.num_k, self.num_params])
    self.epsilon_full = np.zeros([self.batch_size*2+self.num_k, self.num_params])
    for i in range(self.num_params):
        for j in range(self.batch_size*2+self.num_k):
            k = int(self.F_full[j, i])
            self.epsilon_full[j, i] = r[j, i] * self.sigma[k, i]
            self.solutions[j, i] = self.mu[k, i] + self.epsilon_full[j, i]
    solutions = self.solutions
    self.epsilon = self.epsilon_full[self.num_k:self.num_k+self.batch_size, :]
    return solutions

  def comput_entropy(self):
    # 计算每个批次的高斯分布的熵
    r = torch.Tensor(self.r_full)  # N(0,1)中的采样点
    sigma = torch.Tensor(self.sigma)
    sigma.requires_grad = True
    mu = torch.Tensor(self.mu)
    mu.requires_grad = True
    omiga = torch.Tensor(self.omiga)
    omiga.requires_grad = True
    a = torch.Tensor([60, 180, 60, 1, 0.7, 0.7])    # 各个参数的前系数
    F_full = torch.Tensor(self.F_full)  # 隐变量

    # 计算当前批次的每个搜索维度下 各个高斯分量分布的熵 NUM_K*NUM_params
    Entropy = torch.zeros([self.num_k, self.num_params])
    for i in range(self.num_params):
        for j in range(self.batch_size*2+self.num_k):
            k = int(F_full[j, i])
            inside = 1-torch.pow(torch.tanh(mu[k, i] + sigma[k, i] * r[j, i]), 2) + 1e-8
            neg_logp = -torch.log(omiga[k, i]+1e-8) + torch.log(sigma[k, i]+1e-8) + 1/2*torch.pow(r[j, i], 2) + torch.log(inside)
            Entropy[k, i] += neg_logp

    print('每个搜索维度下各个高斯分量分布的熵 entropy:\n', Entropy)

    Entropy = Entropy / torch.Tensor([self.popsize]).repeat(self.num_k, self.num_params)

    # inside = 1-torch.pow(torch.tanh(mu+sigma*r), 2) +1e-8
    # neg_logp = torch.log(sigma+1e-8) + 1/2*torch.pow(r, 2) + torch.log(inside)
    # Entropy = torch.mean(neg_logp, 0)
    # print('entropy:\n', Entropy)
    Entropy = torch.sum(Entropy)

    # 得到的Entropy是当前种群所有参数熵的总和

    # 梯度反向传播
    Entropy.backward()

    mu_entropy_grad = mu.grad.clone()
    sigma_entropy_grad = sigma.grad.clone()
    omiga_entropy_grad = omiga.grad.clone()
    # 梯度清零
    mu.grad.data.zero_()
    sigma.grad.data.zero_()
    omiga.grad.data.zero_()
    print("总的 Entropy：\n", Entropy)
    self.entropy = Entropy

    return mu_entropy_grad.cpu().detach().numpy(), sigma_entropy_grad.cpu().detach().numpy(), omiga_entropy_grad.cpu().detach().numpy()


  def tell(self, reward_table_result, mu_entropy_grad, sigma_entropy_grad, omiga_entropy_grad):
    # input must be a numpy float array
    assert (len(reward_table_result) == self.batch_size*2+self.num_k), "Inconsistent reward_table size reported."

    reward_table = np.array(reward_table_result)

    if self.rank_fitness:
      reward_table = compute_centered_ranks(reward_table)
      # reward_table = compute_normalize(reward_table)

    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay

    reward_offset = self.num_k
    if self.average_baseline:
      b = np.mean(reward_table)
      reward_offset = 0
    else:
      b = reward_table[0:reward_offset]  # baseline

    reward = reward_table[reward_offset:]
    F = self.F_full[reward_offset:]


    best_mu = np.zeros([self.num_k, self.num_params])
    best_rewards = np.zeros([self.num_k, self.num_params])

    for j in range(self.num_params):

      for k in range(self.num_k):
        current_reward = np.array([])
        current_epsilon = np.array([])
        for i in range(len(F)):
          if int(F[i, j]) == k:
            current_reward = np.append(current_reward, reward[i])
            current_epsilon = np.append(current_epsilon, self.epsilon_full[self.num_k+i, j])
        # current_epsilon = current_epsilon.reshape((-1, self.num_params))

        if len(current_reward) != 0:
          if self.use_elite:
            idx = np.argsort(current_reward)[::-1][0:self.elite_popsize]
          else:
            idx = np.argsort(current_reward)[::-1]
          best_reward = current_reward[idx[0]]

          if (best_reward > b[k] or self.average_baseline):
            best_mu[k, j] = self.mu[k, j] + current_epsilon[idx[0]]
            best_rewards[k, j] = reward[idx[0]]
          else:
            best_mu[k, j] = self.mu[k, j]
            best_rewards[k, j] = b[k]
        else:
          best_mu[k, j] = self.mu[k, j]
          best_rewards[k, j] = b[k]

    self.curr_best_reward = best_rewards
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.sigma = np.ones([self.num_k, self.num_params]) * self.sigma_init
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward

    # short hand
    epsilon = self.epsilon
    sigma = self.sigma

    # update the mean

    # move mean to the average of the best idx means
    if self.use_elite:
      self.mu += self.epsilon_full[idx].mean(axis=0)
    else:
      change_mu = np.zeros([self.num_k, self.num_params])
      for j in range(self.num_params):

        for k in range(self.num_k):
          current_reward = np.array([])
          current_epsilon = np.array([])
          for i in range(len(F)):
            if int(F[i, j]) == k:
              current_reward = np.append(current_reward, reward[i])
              current_epsilon = np.append(current_epsilon, self.epsilon_full[self.num_k+i, j])
          if len(current_reward) != 0:
              rT = (current_reward[:int(len(current_reward)/2)] - current_reward[int(len(current_reward)/2):])
              change_mu[k, j] = np.dot(rT, current_epsilon[:int(len(current_epsilon)/2)])/self.omiga[k, j]
          else:
              change_mu[k, j] = 0

      change_mu_all = change_mu + self.mu_lamba*mu_entropy_grad
      print('mu-loss1:\n', change_mu)
      print('mu-loss2:\n', self.mu_lamba*mu_entropy_grad)

      self.optimizer.stepsize = self.learning_rate
      update_ratio = self.optimizer.update(-change_mu_all)  # adam, rmsprop, momentum, etc.
      # self.mu += (change_mu * self.learning_rate) # normal SGD method

    # adaptive sigma
    # normalization
    if self.sigma.all() > self.sigma_min:
    #if (self.sigma[a] > self.sigma_min for a in range(self.num_params)):
      if (self.sigma_alpha > 0 and self.sigma_update):
        stdev_reward = 1.0
        if not self.rank_fitness:
          stdev_reward = reward.std()

        delta_sigma = np.zeros([self.num_k, self.num_params])
        for j in range(self.num_params):
          for k in range(self.num_k):
            current_reward = np.array([])
            current_epsilon = np.array([])
            for i in range(len(F)):
              if int(F[i, j]) == k:
                current_reward = np.append(current_reward, reward[i])
                current_epsilon = np.append(current_epsilon, self.epsilon_full[self.num_k + i, j])

            if len(current_reward) != 0:
              S = ((current_epsilon[:int(len(current_epsilon)/2)] * current_epsilon[:int(len(current_epsilon)/2)] - sigma[k, j] * sigma[k, j]) / sigma[k, j]*self.omiga[k, j])
              reward_avg = (current_reward[:int(len(current_reward)/2)] + current_reward[int(len(current_reward)/2):]) / 2.0
              rS = reward_avg - b[k]

              delta_sigma[k, j] = (np.dot(rS, S)) / (2 * len(current_reward)/2 * stdev_reward)
            else:
              delta_sigma[k, j] = 0
          # adjust sigma according to the adaptive sigma calculation
        # for stability, don't let sigma move more than 10% of orig value
        change_sigma = self.sigma_alpha * (delta_sigma + self.sigma_lamba*sigma_entropy_grad)

        print('sigma-loss1:\n', self.sigma_alpha * delta_sigma)
        print('sigma-loss2:\n', self.sigma_lamba*sigma_entropy_grad)

        change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
        change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
        self.sigma += change_sigma
        self.sigma = np.clip(self.sigma, 0.0, 0.15)

        if (self.sigma_decay < 1):
          self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

    if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

    # adaptive omiga
    if self.update_omiga:
      change_omiga = np.zeros([self.num_k, self.num_params])
      for j in range(self.num_params):
        for k in range(self.num_k):
          current_reward = np.array([])
          for i in range(len(F)):
            if int(F[i, j]) == k:
              current_reward = np.append(current_reward, reward[i])
          if len(current_reward) != 0:
            rT = (current_reward[:int(len(current_reward) / 2)] - current_reward[int(len(current_reward) / 2):])
            change_omiga[k, j] = sum(rT) / self.omiga[k, j]*len(rT)
          else:
            change_omiga[k, j] = 0

      change_omiga_all = self.omiga_alpha*(change_omiga + self.omiga_lamba * omiga_entropy_grad)
      print('omiga-loss1:\n', change_omiga)
      print('omiga-loss2:\n', self.omiga_lamba*omiga_entropy_grad)

      change_omiga_all = np.minimum(change_omiga_all, self.omiga_max_change * self.omiga)
      change_omiga_all = np.maximum(change_omiga_all, - self.omiga_max_change * self.omiga)
      self.omiga += change_omiga_all
      self.omiga = np.clip(self.omiga, 10e-8, 1.0-10e-8)
      # 按列归一化
      for i in range(self.num_params):
        # self.omiga[:, i] = (self.omiga[:, i] - min(self.omiga[:, i])) / (max(self.omiga[:, i])-min(self.omiga[:, i]))
        self.omiga[:, i] = self.omiga[:, i]/sum(self.omiga[:, i])


  def current_param(self):
    return self.curr_best_mu

  def set_mu(self, mu):
    self.mu = np.array(mu)

  def best_param(self):
    return self.best_mu

  def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma, self.omiga)



def NES_GMM_search():
  args = get_opts()
  search_num = args.search_num

  # 搜索三维空间，th phi r
  if search_num == 3:

    MAX_ITERATION = args.iteration
    POPSIZE = args.popsize
    NUM_PARAMS = 3
    solver = PEPG(num_params=NUM_PARAMS,                         # number of model parameters
              sigma_init=0.15,                  # initial standard deviation
              learning_rate=0.1,               # learning rate for standard deviation
              learning_rate_decay=0.999,       # don't anneal the learning rate
              popsize=POPSIZE,             # population size
              average_baseline=False,          # set baseline to average of batch
              weight_decay=0.00,             # weight decay coefficient
              rank_fitness=False,           # use rank rather than fitness numbers
              forget_best=False)

    history = []
    history_best_solution = []
    for j in range(MAX_ITERATION):
      solutions = solver.ask()
      # th (-180,180)
      solutions[:, 0] = 180 * np.tanh(solutions[:, 0])
      # phi (180, 0)
      solutions[:, 1] = 45 * np.tanh(solutions[:, 2]) - 90
      # r (4, 6)
      solutions[:, 2] = np.tanh(solutions[:, 2]) + 4

      fitness_list = np.zeros(solver.popsize)
      for i in tqdm(range(solver.popsize)):
        fitness_list[i] = comput_fitness(solutions[i])
      solver.tell(fitness_list)
      result = solver.result()  # first element is the best solution, second element is the best fitness

      history.append(result[1])
      max_idx = np.argmax(fitness_list)
      history_best_solution.append(solutions[max_idx])
      if (j + 1) % 1 == 0:
        print("fitness at iteration", (j + 1), result[1])
      # print('fitness_list', fitness_list)

    max_idx_ = 0
    '问题：下次迭代的不一定是最好的，如果最好值没变化，要记录之前的'
    for i in range(len(history)-1):
      if history[i+1] > history[i]:
        max_idx_ = i+1
      else:
        continue

    best_solutions = history_best_solution[max_idx_]

    result[0][0] = 180 * np.tanh(result[0][0])
    result[0][1] = 45 * np.tanh(result[0][1]) - 90
    result[0][2] = np.tanh(result[0][2]) + 4

    print('local optimum discovered by solver(best mu):\n th: {:.16f},th: {:.16f},th: {:.16f}'.format(result[0][0], result[0][1], result[0][2]))
    print('local optimum discovered by solver(best solution):\n th: {:.16f},th: {:.16f},th: {:.16f}'.format(best_solutions[0], best_solutions[1], best_solutions[2]))
    print("fitness score at this local optimum: ", result[1])


    "验证"
    x = render_image(th=best_solutions[0], phi=best_solutions[1], r=best_solutions[2])
    test_baseline(path="C:/Users/Silvester/PycharmProjects/NeRFAttack/NeRF/results/blender_for_attack/'hotdog'/",
                  label='hotdog, hot dog, red hot')

  if search_num == 6:

    MAX_ITERATION = args.iteration
    POPSIZE = args.popsize
    NUM_PARAMS = 6
    NUM_K = args.num_k
    N_JOBS = 10
    max_stop_fitness = 6.0
    # 搜索六维空间，th phi gamma r x y
    solver = PEPG(num_params=NUM_PARAMS,  # number of model parameters
                  num_k = NUM_K,
                  sigma_init=0.1,  # initial standard deviation
                  sigma_update=True,  # 不大幅更新sigma
                  learning_rate=0.1,  # learning rate for standard deviation
                  learning_rate_decay=0.99, # don't anneal the learning rate
                  learning_rate_limit=0,
                  popsize=POPSIZE,  # population size
                  average_baseline=False,  # set baseline to average of batch
                  weight_decay=0.00,  # weight decay coefficient
                  rank_fitness=True,  # use rank rather than fitness numbers
                  forget_best=False,
                  mu_lambda=args.mu_lamba,
                  sigma_lambda=args.sigma_lamba,
                  omiga_lamba=args.omiga_lamba,
                  random_begin=args.random_begin,
                  omiga_alpha=0.02
                  )

    logging = {'mu': [], 'sigma': [], 'fitness': [], 'entropy':[]}
    history = []
    fitness_origin = []
    history_best_solution = []
    for j in tqdm(range(MAX_ITERATION)):
      solutions = solver.ask()
      mu_entropy_grad, sigma_entropy_grad, omiga_entropy_grad = solver.comput_entropy()

      # gamma (-60,60)
      solutions[:, 0] = 30 * np.tanh(solutions[:, 0])
      # th (-180,180)
      solutions[:, 1] = 180 * np.tanh(solutions[:, 1])
      # phi (-60, 60)
      solutions[:, 2] = 70 * np.tanh(solutions[:, 2])
      # r (4, 6)
      solutions[:, 3] = np.tanh(solutions[:, 3]) + 4
      # x (-1, 1)
      solutions[:, 4] = 0.5 * np.tanh(solutions[:, 4])
      # x (-1, 1)
      solutions[:, 5] = 0.5 * np.tanh(solutions[:, 5])

      fitness_list = np.zeros(solver.popsize)

      #  多进程工作
      # with joblib.Parallel(n_jobs=N_JOBS) as parallel:
      #   #for i in tqdm(range(solver.popsize)):
      #     #fitness_list[i] = comput_fitness(solutions[i])

      #   fitness_list = parallel(joblib.delayed(comput_fitness)(solutions[i], solver.sigma) for i in tqdm(range(solver.batch_size*2+solver.num_k)))

      fitness_list = comput_fitness(solutions)
      
      solver.tell(fitness_list, mu_entropy_grad, sigma_entropy_grad, omiga_entropy_grad)
      result = solver.result()  # first element is the best solution, second element is the best fitness

      fitness_origin.append(np.max(fitness_list))
      history.append(result[1])
      average_fitness = np.mean(fitness_list)
      max_idx = np.argmax(fitness_list)
      history_best_solution.append(solutions[max_idx])
      if (j + 1) % 1 == 0:
        print("fitness at iteration\n", (j + 1), max(fitness_origin))
        print("average fitness at iteration\n", (j + 1), average_fitness)
        print("sigma at iteration\n", (j + 1), result[3])
        print("mu at iteration\n", (j + 1), result[0])
        print("omiga at iteration\n", (j + 1), result[4])

        # 写入日志数据
        logging['fitness'].append(result[1])
        logging['sigma'].append(result[3])
        logging['mu'].append(result[0])
        logging['entropy'].append(solver.entropy)
      # print('fitness_list', fitness_list)

      # 若达到搜索要求，则停止迭代
      #if average_fitness > max_stop_fitness:
        #break

    # max_idx_ = 0
    # '问题：下次迭代的不一定是最好的，如果最好值没变化，要记录之前的'
    # for i in range(len(history) - 1):
    #   if history[i + 1] > history[i]:
    #     max_idx_ = i + 1
    #   else:
    #     continue

    # best_solutions = history_best_solution[max_idx_]

    # 输出sigma和mu的在tanh后的混合高斯采样值
    # 根据omiga矩阵对每个维度采样分量标号
    random = np.zeros([args.num_sample + solver.num_k, 6])
    mu = result[0]
    sigma = result[3]

    F_all = np.zeros([args.num_sample, solver.num_params])
    for i in range(solver.num_params):
      F = np.random.choice(a=np.arange(solver.num_k), size=args.num_sample, replace=True, p=solver.omiga[:, i])
      F_all[:, i] = F

    def get_GMM_sample(F, mu, sigma):
      sample_all = np.zeros(args.num_sample)
      for i in range(len(F)):
        k = int(F[i])
        sample = np.random.normal(loc=mu[k], scale=sigma[k], size=1)
        sample_all[i] = sample
      sample_all = np.concatenate([sample_all, mu])
      return sample_all

    a = [30, 180, 70, 1.0, 0.5, 0.5]
    b = [0, 0, 0, 4.0, 0, 0]
    for i in range(solver.num_params):
      random[:, i] = a[i]*np.tanh(get_GMM_sample(F_all[:, i], mu[:, i], sigma[:, i]))+b[i]

    # 计算各高斯分量的方差 均值
    # mu_ = np.zeros([solver.num_k, solver.num_params])
    # var = np.zeros([solver.num_k, solver.num_params])
    # for j in range(solver.num_params):
    #   for i in range(solver.num_k):
    #     sample = np.array([])
    #     for l in range(len(F)):
    #       if int(F[l]) == i:
    #         sample = np.append(sample, random[l, j])
    #     if len(sample) != 0:
    #       mu_[i, j] = np.mean(sample)
    #       var_ = (sample - mu_[i, j]).T @ (sample - mu_[i, j]) / sample.shape[0]
    #       var[i, j] = np.sqrt(var_)
    #     else:
    #       mu_[i, j] = 0
    #       var[i, j] = None
    
    for i in range(solver.num_k):
      max_value = a * np.tanh(mu[i, :]+sigma[i, :]) + b
      min_value = a * np.tanh(mu[i, :]-sigma[i, :]) + b
      value_range = max_value-min_value
      sigma[i,:] = value_range

    print('final sigma after tanh（角度方差）\n', sigma)

    for i in range(solver.num_k):
      mu[i, :] = a * np.tanh(mu[i, :]) + b

    print('final mu after tanh（角度中心值）\n', mu)
    print('final omiga after tanh（高斯分量权重）\n', result[4])

    "渲染100张该分布下的图像"
    print('begin render 100 images in current adv-distribution')
    print('--------------------------------------------------')
    render_image(random, is_over=True)

    "验证准确率"
    print('begin test the accuracy')
    print('--------------------------------------------------')
    path = '/data/home/scv7303/run/rsw_/NeRFAttack/ngp_pl/run_GMM/results/nerf_for_attack/' + args.scene_name + '/'
    acc = test_baseline(path=path, label=args.label, model='resnet')
    print("acc:", acc)

    # print('no.100 the mean img')
    # print('--------------------------------------------------')
    # path = '/HOME/scz1972/run/rsw_/NeRFAttack/run_NES_rebuttal_A/results/blender_for_attack/' + args.scene_name + '/'
    # test_baseline(path=path, label=args.label_name, model='resnet', is_mean=True)

