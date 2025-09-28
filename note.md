## 一、强化学习的基本框架

强化学习的目标是让智能体（Agent）通过与环境（Environment）的交互，不断学习最优策略（Policy），以最大化累计奖励（Reward）。

核心元素：

* **状态（State, s）** ：环境的描述，比如在 CartPole 中，小车位置和速度。
* **动作（Action, a）** ：智能体的选择，如“左移”或“右移”。
* **奖励（Reward, r）** ：环境反馈，指导智能体学习。
* **策略（Policy, π）** ：从状态到动作的映射。
* **价值函数（Value Function）** ：评估状态/状态-动作的好坏程度。
* **环境模型（Model）** ：可选，用于预测状态转移。

数学上通常建模为  **马尔可夫决策过程 (MDP)** ：

M=(S,A,P,R,γ)\mathcal{M} = (S, A, P, R, \gamma)

* SS：状态空间
* AA：动作空间
* P(s′∣s,a)P(s'|s,a)：状态转移概率
* R(s,a)R(s,a)：奖励函数
* γ∈[0,1)\gamma \in [0,1)：折扣因子

---

## 二、价值函数与贝尔曼方程

强化学习的核心在于学习价值函数。

1. **状态价值函数 (State Value Function)**

Vπ(s)=Eπ[∑t=0∞γtrt∣s0=s]V^\pi(s) = \mathbb{E}_\pi \Big[ \sum_{t=0}^\infty \gamma^t r_{t} \mid s_0 = s \Big]
2. **动作价值函数 (Action Value Function)**

Qπ(s,a)=Eπ[∑t=0∞γtrt∣s0=s,a0=a]Q^\pi(s,a) = \mathbb{E}_\pi \Big[ \sum_{t=0}^\infty \gamma^t r_{t} \mid s_0 = s, a_0=a \Big]
3. **贝尔曼方程 (Bellman Equation)**

* 对于 VπV^\pi：

Vπ(s)=∑aπ(a∣s)∑s′P(s′∣s,a)[R(s,a)+γVπ(s′)]V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \big[ R(s,a) + \gamma V^\pi(s') \big]

* 对于最优 Q∗Q^*：

Q∗(s,a)=E[r+γmax⁡a′Q∗(s′,a′)]Q^*(s,a) = \mathbb \Big[ r + \gamma \max_ Q^*(s',a') \Big]
--------------------------------------------------------------------------------------------

## 三、强化学习方法分类

1. **基于价值（Value-based）**
   * 目标：学习 Q∗(s,a)Q^*(s,a)
   * 代表：Q-learning、DQN（Deep Q-Network）
2. **基于策略（Policy-based）**
   * 目标：直接学习策略 πθ(a∣s)\pi_\theta(a|s)
   * 代表：REINFORCE、Actor-Critic
3. **Actor-Critic 结合**
   * 既学策略又学价值
   * 代表：A2C, A3C, PPO, SAC
4. **基于模型（Model-based）**
   * 学习环境转移模型
   * 代表：Dyna-Q, MuZero

---

## 四、经典算法学习路线

1. **入门阶段（理解基本概念）**
   * 多臂老虎机 (Multi-Armed Bandit)
   * 策略评估 & 策略迭代 (Policy Evaluation / Iteration)
   * 值迭代 (Value Iteration)
2. **表格型方法（小状态空间）**
   * Q-learning
   * SARSA
3. **函数逼近（大状态空间）**
   * 深度 Q 网络 (DQN)
   * Double DQN, Dueling DQN, Prioritized Replay
4. **策略梯度方法**
   * REINFORCE
   * Actor-Critic
5. **进阶与SOTA**
   * A3C / A2C
   * PPO (最常用)
   * DDPG, TD3, SAC (连续动作空间)

---

## 五、强化学习常用工具

* **Gymnasium / OpenAI Gym** ：环境接口
* **Stable Baselines3** ：强化学习算法库
* **RLlib** ：分布式RL
* **PettingZoo** ：多智能体RL环境

---

## 六、学习路径建议

1. **数学基础** ：概率论、线性代数、优化
2. **代码实践** ：用 Python + Gym 实现 Q-learning、DQN
3. **深度学习结合** ：理解 DNN 在价值函数逼近中的作用
4. **论文阅读** ：从 DQN (2015) → A3C (2016) → PPO (2017) → SAC (2018)
5. **实战** ：CartPole、MountainCar → Atari → MuJoCo → 自定义任务
