import numpy as np
import os
from evaluate_forAT import comput_fitness
from rendering_image import render_image
from classifier.predict import test_baseline
from tqdm import tqdm
from datasets.opts import get_opts
import joblib
import torch
import time
np.set_printoptions(precision=4,  linewidth=100, suppress=True)
# np.random.seed(0)


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
               random_begin=True,
               mood='init',
               mu_start=None,
               sigma_start=None
               ):

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
    if mood == 'init' or mood == 'eval_test':
      if not random_begin:
        self.mu = np.zeros([self.num_k, self.num_params])
      else:
        self.mu = 2.0*np.random.random_sample([self.num_k, self.num_params])-1.0

      self.sigma = np.ones([self.num_k, self.num_params]) * self.sigma_init
    else:
      self.mu = mu_start
      self.sigma = sigma_start


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

    # print('每个搜索维度下各个高斯分量分布的熵 entropy:\n', Entropy)

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
    # print("总的 Entropy：\n", Entropy)
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
      # print('mu-loss1:\n', change_mu)
      # print('mu-loss2:\n', self.mu_lamba*mu_entropy_grad)

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

        # print('sigma-loss1:\n', self.sigma_alpha * delta_sigma)
        # print('sigma-loss2:\n', self.sigma_lamba*sigma_entropy_grad)

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
      # print('omiga-loss1:\n', change_omiga)
      # print('omiga-loss2:\n', self.omiga_lamba*omiga_entropy_grad)

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



def NES_GMM_search_step(model, label, ckpt_path, mood, mu_start=None, sigma_start=None):
  args = get_opts()
  search_num = args.search_num
  
  if search_num == 6:

    if mood == 'init' or mood == 'eval_test':
      MAX_ITERATION = args.iteration
    else:
      MAX_ITERATION = args.iteration_warmstart
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
                  omiga_alpha=0.02,
                  mood = mood,
                  mu_start=mu_start,
                  sigma_start=sigma_start
                  )
    history = []
    fitness_origin = []
    for j in range(MAX_ITERATION):
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

      fitness_list = comput_fitness(model, label, ckpt_path, solutions, is_viewfool=False)
      
      solver.tell(fitness_list, mu_entropy_grad, sigma_entropy_grad, omiga_entropy_grad)
      result = solver.result()  # first element is the best solution, second element is the best fitness

      fitness_origin.append(np.max(fitness_list))
      history.append(result[1])
      average_fitness = np.mean(fitness_list)

      # if (j + 1) % 10 == 0:
        # print("================================================================")
        # print("fitness at iteration\n", (j + 1), max(fitness_origin))
        # print("average fitness at iteration\n", (j + 1), average_fitness)
      
 
    mu = result[0]
    sigma = result[3]
    Entropy = solver.entropy

    return mu, sigma, Entropy



class GMFool:

  def __init__(self, dist_pool_mu=None, dist_pool_sigma=None, mood='init'):

    args = get_opts()
    if mood == 'init' or mood == 'warm_start':
      all_class = os.listdir(f'{args.ckpt_attack_path}/train/')
    else: # eval_test
      all_class = os.listdir(f'{args.ckpt_attack_path}/test/')
    all_class.sort()
    class_num = len(all_class)
    self.class_num = class_num
    self.all_class = all_class
    # object_num = len(os.listdir(f'{args.ckpt_attack_path}/' + all_class[0] + '/'))
    object_num = 10

    self.dist_pool_mu = dist_pool_mu
    self.dist_pool_sigma = dist_pool_sigma


  def step(self, class_id, all_class, model, mood):
      args = get_opts()
      object_num = 10
      mu_result = np.zeros([object_num, args.num_k, 6])
      sigma_result = np.zeros([object_num, args.num_k, 6])
      Entropy_class = 0

      if mood == 'init' or mood == 'warm_start':
        path = f'{args.ckpt_attack_path}/train/'
      else:
        path = f'{args.ckpt_attack_path}/test/'

      all_object = os.listdir(path + all_class[class_id] + '/')
      all_object.sort()

      if args.fast_AVDT:  #轮流优化策略，每次随机优化一个物体的参数
        if mood == 'init' or mood == 'eval_test':
          for object_id in range(len(all_object)):
            ckpt_path = path + all_class[class_id] + '/' + all_object[object_id]
            label = int(all_class[class_id])
            mu, sigma, Entropy = NES_GMM_search_step(model, label, ckpt_path, mood)
            Entropy_class += Entropy

            mu_result[object_id, :, :] = mu
            sigma_result[object_id, :, :] = sigma
          rand = None

        elif mood == 'warm_start':
          rand = int(np.random.random_integers(0, len(all_object)-1, size=1))
          ckpt_path = path + all_class[class_id] + '/' + all_object[rand]
          label = int(all_class[class_id])

          mu_start = self.dist_pool_mu[class_id, rand, :, :]
          sigma_start = self.dist_pool_sigma[class_id, rand, :, :]
          mu, sigma, Entropy = NES_GMM_search_step(model, label, ckpt_path, mood, mu_start, sigma_start)
          Entropy_class += Entropy

          mu_result[rand, :, :] = mu
          sigma_result[rand, :, :] = sigma

      else:
        for object_id in range(len(all_object)):
          ckpt_path = path + all_class[class_id] + '/' + all_object[object_id]
          label = int(all_class[class_id])

          if mood == 'init' or mood == 'eval_test':
            mu, sigma, Entropy = NES_GMM_search_step(model, label, ckpt_path, mood)
            Entropy_class += Entropy

          elif mood == 'warm_start':
            mu_start = self.dist_pool_mu[class_id, object_id, :, :]
            sigma_start = self.dist_pool_sigma[class_id, object_id, :, :]
            mu, sigma, Entropy = NES_GMM_search_step(model, label, ckpt_path, mood, mu_start, sigma_start)
            Entropy_class += Entropy

          mu_result[object_id, :, :] = mu
          sigma_result[object_id, :, :] = sigma

        rand = None

      return mu_result, sigma_result, rand, Entropy_class
    

def NES_GMM_search(model, dist_pool_mu, dist_pool_sigma, mood='init'):

  args = get_opts()
  method = 'para'
  
  GMFool_solver = GMFool(dist_pool_mu=dist_pool_mu, dist_pool_sigma=dist_pool_sigma, mood=mood)
  
  if mood == 'init' or mood == 'eval_test':
    dist_pool_mu_return = np.zeros([GMFool_solver.class_num, 10, args.num_k, 6])
    dist_pool_sigma_return = np.zeros([GMFool_solver.class_num, 10, args.num_k, 6])
  elif mood == 'warm_start':
    dist_pool_mu_return = dist_pool_mu
    dist_pool_sigma_return = dist_pool_sigma

  #-----------------------------------并行-------------------------------------------------------------
  if method == 'para':
    with joblib.Parallel(n_jobs=5) as parallel:
      res = parallel(joblib.delayed(GMFool_solver.step)(class_id, GMFool_solver.all_class, model, mood) for class_id in tqdm(range(GMFool_solver.class_num)))

    mu_result = [item[0] for item in res]
    sigma_result = [item[1] for item in res]
    rand = [item[2] for item in res]
    Entropy_class = [item[3] for item in res]

    average_entropy = sum(Entropy_class)/len(Entropy_class)

    if mood == 'init' or mood == 'eval_test':
      for i in range(len(mu_result)):
        dist_pool_mu_return[i, :, :, :] = mu_result[i]
        dist_pool_sigma_return[i, :, :, :] = sigma_result[i]

    elif mood == 'warm_start' and not args.fast_AVDT:
      for i in range(len(mu_result)):
        dist_pool_mu_return[i, :, :, :] = mu_result[i]
        dist_pool_sigma_return[i, :, :, :] = sigma_result[i]

    elif mood == 'warm_start' and args.fast_AVDT:
      for i in range(len(mu_result)):
        mu = mu_result[i]
        sigma = sigma_result[i]
        dist_pool_mu_return[i, rand[i], :, :] = mu[rand[i], :, :]
        dist_pool_sigma_return[i, rand[i], :, :] = sigma[rand[i], :, :]

  #-----------------------------------串行-------------------------------------------------------------
  else:
    for class_id in tqdm(range(GMFool_solver.class_num)):
      mu_result, sigma_result, rand, Entropy_class = GMFool_solver.step(class_id, GMFool_solver.all_class, model, mood)
      if mood == 'init' or mood == 'eval_test':
        dist_pool_mu_return[class_id, :, :, :] = mu_result
        dist_pool_sigma_return[class_id, :, :, :] = sigma_result

      elif mood == 'warm_start' and not args.fast_AVDT:
        dist_pool_mu_return[class_id, :, :, :] = mu_result
        dist_pool_sigma_return[class_id, :, :, :] = sigma_result

      elif mood == 'warm_start' and args.fast_AVDT:
        dist_pool_mu_return[class_id, rand, :, :] = mu_result
        dist_pool_sigma_return[class_id, rand, :, :] = sigma_result
#-----------------------------------------------------------------------------------------------------------
  
  # print("mu_result:", dist_pool_mu_return)
  # print("sigma_result:", dist_pool_sigma_return)

  print("average_entropy:", average_entropy)

  if mood == 'init' or mood == 'warm_start':
    np.save(f'./dist_pool/dist_pool_mu_{args.AT_exp_name}.npy', dist_pool_mu_return)
    np.save(f'./dist_pool/dist_pool_sigma_{args.AT_exp_name}.npy', dist_pool_sigma_return)
  else:
    np.save(f'./dist_pool/dist_pool_mu_{args.AT_exp_name}_{mood}.npy', dist_pool_mu_return)
    np.save(f'./dist_pool/dist_pool_sigma_{args.AT_exp_name}_{mood}.npy', dist_pool_sigma_return)

