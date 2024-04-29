---
layout: single
title: "Counterfactual Multi-Agent Policy Gradients"
categories: "MARL"
tag: "Paper"
typora-root-url: ../
sidebar:
  nav: "docs"
use_math: true
---

## Proximal Policy Optimization Algorithms

v1: 2017.07.20 (OpenAI)

citations: 16244(2024-03-09)

------

**surrogate objective function**:  양의 정부호 행렬을 가지는 선형방정식을 효과적으로 풀기 위한 알고리즘이다. 이 방법은 크기가 크고 값이 대부분 0인 행렬을 다룰 때 유용하게 사용된다. 선형 방정식의 해를 찾기 위해 반복적인 과정을 수행한다. 에너지 최소화와 같은 제약 조건이 없는 최적화 문제의 해결에도 사용 가능하다. 

**sample complexity**: target function을 성공적으로 학습하기 위해 요구되는 training sample의 수

**first-order optimization**: 전역 최소값을 구하는 과정을 optimization으로 정의할 수 있다. 이때 적어도 하나의 1차 도함수/gradient를 요구하는 모든 알고리즘을 말한다. gradient descent가 여기에 속한다. 

**conjugate gradient**: 양의 정부호 행렬

**linear approximation**: 어떤 복잡한 함수를 선형 함수, 즉 1차함수로 근사하는 것을 의미한다. 

**quadratic approximation**: 어떤 복잡한 함수를 2차함수로 근사하는 것을 의미한다. 

-----



### Abstract 

environment와의 상호작용을 통한 데이터 샘플링과 stochastic gradient ascent를 이용한 surrogate(대체) objective function 최적화를 번갈아가며 수행하는 새로운 policy gradient 계열 알고리즘을 소개한다. 기존 policy gradient method가 각 데이터당 하나의 gradient update를 수행한 것과 다르게 새로운 objective function은 minibatch update에서의 multiple epoch를 보장한다. PPO는 TRPO에 비해 몇 가지 장점이 있으면서 더 일반적이고 실행하기 쉬우며 더 좋은 sample complexity를 가진다. 저자는 PPO가 각종 benchmark에서 다른 policy gradient methods를 능가하는 성능을 보이며 sample complexity, simplicity, wall-time 사이에서 전체적으로 더 좋은 밸런스를 보이는 것을 확인했다. 

### 1. Introduction

기존의 알고리즘들은 scalable(모델의 크기 확장과 병렬 실행), data efficient, robust(다양한 task에 대해 하이퍼파라미터 튜닝 없이 적용 가능한가?)에 대해서 개선의 여지가 있었다. Q-Learning의 경우 continuous control benchmark에 취약하고 Vanilla policy gradient method의 경우 낮은 data efficiency, robustness가 발목을 잡았다. TRPO(trust region policy optimization)의 경우 상대적으로 복잡하며 parameter sharing, dropout과 같은 architecture와 호환되지 않았다. 

논문의 알고리즘은 first-order optimization만을 사용해 TRPO의 data efficiency와 성능을 달성한다. 저자는 새로운 objective: clipped probability ratio를 제안하며 이는 policy의 성능에 대해 하한을 형성한다. 정책을 최적화하기 위해 정책에서 데이터를 샘플링하고, 샘플링된 데이터에 대해 여러 epoch를 통해 최적화를 수행하는 과정을 반복한다. continuous control task에서는 비교 대상 알고리즘보다 더 좋은 성능을 보인다. Atari에서는 A2C보다 더 뛰어난 sample complexity를 보이며 ACER과 유사한 성능을 보이지만 훨씬 간단하다. 

### 2. Background: Policy Optimization

##### 2.1 Policy Gradient Methods

$$
\begin{aligned}
L^{PG}(\theta) &= \hat{\mathbb{E}}_t[\log\pi_{\theta}(a_{t} | s_{t})\hat{A}_t] \\
\hat{g} &= \hat{\mathbb{E}}_{t} [∇_{\theta}\log\pi_{\theta}(a_{t} | s_{t})\hat{A}_t]
\end{aligned}
$$

gradient 추정치 g_hat은 위와 같은 값을 가지며 목적 함수를 미분하여 얻어진다. 동일한 trajectory를 활용해 이 Loss function을 여러 번 최적화하는 것은 매력적인 반면에,  well-justified(증명되거나 검증)되지 않으며, 경험적으로 볼 때 종종 파괴적으로 큰 policy update로 이끈다. 

##### 2.2 Trust Region Methods

TRPO는 objective function(surrogate objective)를 최대화하면서 policy update의 크기에 대해 제약조건을 부여하는 방법이다. 
$$
\begin{aligned}
maximize_{\theta}\;&\hat{\mathbb{E}}\bigg[\frac{\pi_\theta(a_t|s_t)}{\pi_{{\theta}_{old}}(a_t|s_t)}\hat{A}_t\bigg]  \\ 
subject\;to\;&\hat{\mathbb{E}}[KL[\pi_{\theta_{old}}(·|s_t),\,\pi_{\theta}(·|s_t)]] \leq \delta
\end{aligned}
$$
θold는 업데이트되기 전의 policy parameter다. 이 문제는 conjugate gradient algorithm을 사용해 근사적으로 해결할 수 있다. objective에 대해서 linear approximation을 한 뒤 constraint에  대해 quadratic approximation을 수행한다. 

TRPO를 정당화하는 이론은 사실 constraint 대신에 penalty를 사용할 것, 즉 unconstrained optimization problem을 해결하는 것을 제안한다. 
$$
\begin{aligned}
maximize_{\theta}\;&\hat{\mathbb{E}}\bigg[\frac{\pi_\theta(a_t|s_t)}{\pi_{{\theta}_{old}}(a_t|s_t)}\hat{A}_t-\beta KL[\pi_{\theta_{old}}(·|s_t),\,\pi_{\theta}(·|s_t)]\bigg]
\end{aligned}
$$
beta는 coefficient를 나타낸다. 특정한 surrogate objective(평균 대신 각 state 별 최대 KL을 계산하는 함수)가 정책의 성능에 대해 하한을 형성한다는 사실로부터 추론된다. TRPO는 penalty를 사용하기 보다 hard constraint를 사용한다. 다양한 문제에서 잘 동작하는 단일 beta값을 선택하기 어렵기 때문이다. 심지어 단일 문제 내에서도 학습 과정에서 특성이 변하기 때문에 어려움이 있다. 

따라서 TRPO의 단조적 향상을 모방하는 first-order algorithm을 성취하기 위해, 실험 결과는 단순히 고정된 penalty coefficient beta를 선택하고 penalized objective function을 SGD로 최적화하는 것만으로는 충분하지 않다는 것을 보여준다.

### 3. Clipped Surrogate Objective 

$$
\begin{aligned}
r_t(\theta) &= \frac{\pi_\theta(a_t|s_t)}{\pi_{{\theta}_{old}}(a_t|s_t)} \\
L^{CPI}(\theta) &= \hat{\mathbb{E}}_t\bigg[\frac{\pi_\theta(a_t|s_t)}{\pi_{{\theta}_{old}}(a_t|s_t)}\hat{A}_t\bigg] = \hat{\mathbb{E}}_t\Big[r_t(\theta)\hat{A}_t\Big]
\end{aligned}
$$

r_t(θ)를 위 수식과 같이 표기하므로 r_t(θold)일 경우 r_t(θ)는 1이다. TRPO는 surrogate objective를 maximize한다. superscript(위첨자) CPI는 conservative policy iteration이다. constraint 없이 L^CPI의 maximization은 정책 업데이트를 지나치게 빠르게 만들 수 있다. 따라서 r_t(θ)가 1에서 벗어나는 정책의 업데이트에 대해서 패널티로 적용하기 위해서 objective를 수정해야 한다. 논문이 제안하는 objective는 아래와 같다. 
$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}\Big[\min(r_t(\theta)\hat{A}_t,\,clip(r_t(\theta),\,1-\epsilon, \, 1+\epsilon)\hat{A}_t)
$$
epsilon은 하이퍼파라미터로 예를 들어 0.2라고 가정한다. 이 objective의 동기는 아래와 같다. min 내부의 첫 번째 항은 L^CPI이다. 두 번째 항은 확률 비율을 clipping하여 surrogate objective를 수정하며 이로써 r_t가 [1-eps, 1+eps]구간을 벗어나려는 의욕을 제거한다. 마지막으로 clipped objective와 unclipped objective를 비교해 작은 값을 선택하여 최종적인 objective는 unclipped objective에 대한 하한이 된다. 이 방식을 이용하면 object가 개선될 때에만 probability ratio의 변화를 무시하고, object가 악화될 때에만 포함시킨다. L^CLIP(θ)는 θold 주변에서  first-order(1차 근사)로는 같지만 θ가 θold에서 멀어질수록 다르게 된다. 

################## Figure 1

Figure 1은 L^CLIP의 single term(single t)를 그래프로 나타내며, 이때 probability ratio r은 advantage가 양수인지 음수인지에 따라 [1-eps, 1+eps]에서 클리핑된다는 것을 알 수 있다. 

################## Figure 2

Figure 2는 surrogate objective(L^CLIP)에 대한 직관을 제시한다. continuous control problem에서 PPO를 통해 policy update 방향을 보간(중간에 삽입)함에 따라 여러 objective function이 어떻게 변하는지 보여준다. L^CLIP가 L^CPI의 하한임을 볼 수 있으며, 정책 업데이트가 너무 급격한 경우에 대해 penalty가 있음을 확인가능하다.  

