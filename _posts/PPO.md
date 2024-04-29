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

## <center>근위 정책 최적화(PPO)</center>



기존 Policy Gradient 강화학습이 활용하는 방법

![Screenshot from 2024-04-29 11-56-56](/바탕화면/singforai.github.io/images/PPO/Screenshot from 2024-04-29 11-56-56.png)

보상을 최대화하는 보상 함수를 설계하고 그 값을 극대화하는 방향으로 학습, 즉 Gradient  Descent가 아닌 Gradient Ascent를 통해 학습하므로 값이 클수록 좋은 거임.



기존 Policy gradient의 한계

1. **performance collapse**  !!!중요!!!
2. **low sample efficiency**

PPO: 이 두 가지의 한계를 극복하기 위한 일종의 최적화 알고리즘으로 주요 아이디어는

1. **monotonic policy improvement(단조 정책 향상)** 
2. **surrogate objective(대리 목적)** 

PPO는 원래의 목적 함수 J를 수정된 PPO 목적 함수로 대체하여 REINFORCE 또는 Actor-Critic을 확장하는데 사용할 수 있다. 



#### performance collapse

목적 함수 J가 policy space에 속하는 한 policy에 의해 생성되는 trajectory를 이용하여 계산될지라도, 실제 optimal policy에 대한 탐색은 parameter space에서 적합한 파라미터를 찾는 방식으로 진행된다. policy space와 parameter space가 항상 정확히 매칭되지는 않기 때문에 두 파라미터집합이 서로 거리가 있더라도 둘 다 좋은 정책일 수 있고 가까움에도 두 정책의 점수가 극명하게 다를수도 있다. 
$$
\begin{aligned}
d_\theta(\theta_1, \theta_2) = d_\theta(\theta_2, \theta_3) \not\Leftrightarrow	 d_\pi(\pi_{\theta_1}, \pi_{\theta_2}) = d_\pi(\pi_{\theta_2}, \pi_{\theta_3})
\end{aligned}
$$
각 쌍을 이루는 파라미터 사이의 거리가 같다고 해도 파라미터에 대응되는 정책이 꼭 같은 거리를 유지하는 것은 아니다. 이것은 파라미터 업데이트를 위한 이상적인 LR을 결정하기 어렵다는 점에서 문제가 된다. parameter space에 대응되는 policy space에 속하는 policy 사이의 간격이 얼마인지를 사전에 아는 것을 불가능하다. LR은 정책이 아닌 "파라미터의 업데이트 크기에 영향"을 미치기 때문에 LR이 너무 크면 policy space 내부의 업데이트 간격이 너무 커서 좋은 policy를 건너뛰게 되어 performance collapse가 발생한다. 기존 policy보다 안좋은 policy가 더 나쁜 trajectory data를 생성하고 그 data를 이용해 다음 업데이트가 진행되기 때문에 policy iteration을 망치게 될 것이다. 일반적으로 LR을 고정하면 이 이슈를 해결할 수 없다. parameter space에서의 업데이트가 policy space에 어떤 영향을 미치는지를 고려하여 그에 맞게 LR을 결정하기 위해서는 두 policy 사이의 성능 차이를 측정하는 방법을 알아야 한다. 즉, LR Decay와 같은 경험적 하이퍼 파라미터 업데이트 방식에서 벗어나 이론적 기반을 가진 업데이트 방식을 활용했다는 것에 의의가 있다. 

#### Objective function

직관적으로 보면, 이 문제는 업데이트 간격으로 인한 문제이기 때문에 performance collapse를 막기 위해서 "정책의 업데이트 간격"을 안전 영역 이내로 유지하게 하는 제약 조건을 도입하는 것을 생각해볼 수 있다. 적합한 제약 조건을 부여함으로써 **Monotonic Improvement Theory(단조 향상 이론)**을 도출할 수 있다. 

아래의 수식과 같이 두 목적 함수 사이의 차이로 정의된 **relative policy performance identity(상대적 정책 성능 식별자: RPPI)**를 통해 두 정책 사이의 성능차이를 측정한다. π'은 업데이트된 정책이고 π는 업데이트되기 전의 정책이다. 
$$
\begin{aligned}
J(\pi') - J(\pi) = \mathbb{E}_{\tau \sim \pi'}\Big[\sum^{T}_{t=0}\gamma^tA^\pi(s_t,a_t)\Big]
\end{aligned}
$$
RPPI는 정책이 향상된 정도를 측정하는 지표의 역할을 한다. 이 차이가 양수이면 새 정책이 기존 정책보다 좋은 것이다. policy iteration 과정에서 새로운 정책 π'을 선택할 때 이 차이가 최대가 되도록 하는 것이 이상적이다. 그러므로 목적 함수 J(π')을 최대화하는 것과 RPPI를 최대화하는 것은 동일하며, 이 둘은 모두 Gradient ascent를 통해 이루어진다. 
$$
\begin{aligned}
\max_{π'} J(π') \Leftrightarrow \max_{\pi'}(J(\pi') - J(\pi))
\end{aligned}
$$
목적 함수를 이러한 방식으로 구조화하는 것은 모든 policy iteration이 monotonic한 향상을 보장해야 하기에 RPPI의 값이 항상 0보다 같거나 커야 한다. 왜냐하면 최악의 경우에도 아무런 policy iteration없이 π' = π를 보장하기 때문이다. 이렇게 하면 훈련 과정에서 performance collapse가 일어나지 않을 것이다. 

하지만 RPPI를 새로운 objective function으로써 사용함에 있어 두 가지의 제약이 있다. 



1. 업데이트를 위해 새로운 정책 π'로부터 추출된 trajectory로부터 기댓값이 계산되어야 하지만 π'은 업데이트 전에는 얻을 수가 없다. 이 역설을 해결하기 위해 당장 이용 가능한 기존 정책 π를 사용하도록 변경해야 한다. 이를 위해 연속적인 정책 π', π가 상대적으로 가까워서 정책이 샘플링한 상태 분포가 유사하다고 가정할 수 있다(KL-divergence가 작은 것으로부터 확인할 수 있다). 그러면 위의 RPPI의 식에 있는 π'를 π로 변경하고 importance sampling weight로 조정해 근사적으로 표현 가능하다. 이것은 π를 이용해 생성된 reward를 연속된 두 정책 π', π 사이의 행동 확률의 비율만큼 조정한다. 새로운 정책 π'에 더 잘 부합할 것 같은 행동과 관련된 reward의 가중치는 증가할 것이고, 상대적으로 π'하에서 발생할 것 같지 않은 행동과 관련된 보상의 가중치는 감소할 것이다. 
   $$
   \begin{aligned}
   J(\pi') - J(\pi) &= \mathbb{E}_{\tau \sim \pi'}\Big[\sum_{t\geq0}A^\pi(s_t,a_t)\Big] \\
   &\approx \mathbb{E}_{\tau \sim \pi}\Big[\sum_{t\geq0}A^\pi(s_t,a_t)\frac{\pi'(a_t, s_t)}{\pi(a_t, s_t)}\Big] = J^{CPI}_\pi(\pi')
   \end{aligned}
   $$
   새롭게 등장한 J(CPI) 목적 함수는 π'와 π 사이의 비율을 포함하므로 surrogate objective라고 한다. 위첨자 CPI는 보수적 정책 반복(Conservative policy iteration)을 뜻한다. 새로운 목적 함수를 정책 경사 알고리즘에 적용하기 위해 이 목적 함수를 이용한 최적화가 여전히 정책 경사상승을 수행하는지 확인할 필요가 있다.

$$
\begin{aligned}
\nabla_\theta J^{CPI}_{\theta_{old}}(\theta) | \theta_{old} = \nabla_{\theta}J(\pi_\theta) | \theta_{old}
\end{aligned}
$$

​	위 수식을 통해 surrogate objective의 경사가 정책 경사와 같음을 보일 수 있다. 

2. J(CPI)는 importance sampling을 통해 근사를 수행했기 때문에 근사 오차가 있을 것이다. 이 근사 오차는 monotonic한 향상을 보장해야 하는 기존 RPPI의 조건을 만족하지 못하게 할 수 있다. 따라서 근사 과정에서 발생하는 오차에 대한 이해가 필요하다. 

   연속적인 정책 π', π가 KL-Divergence를 기준으로 서로 충분히 비슷하다면 RPPI를 표현할 수 있다. 이를 위해 새로운 목적함수 J(π')와 그에 대한 추정값 J(π) + J(CPI)의 차이에 대한 절댓값 오차를 표현한다. 그러면 이 오차는 π', π 사이의 KL-Divergence를 통해 제한될 수 있다.  
   $$
   \begin{aligned}
   |(J(\pi') - J(\pi)) - J^{CPI}_\pi(\pi')| \leq C \sqrt{\mathbb{E}_t[KL(\pi'(a_t, s_t)||\pi(a_t, s_t))]} \;\;\;\;\;\;7.26
   \end{aligned}
   $$
   C는 선택해야 하는 상수이다. 연속적인 정책 π', π를 나타내는 확률분포가 서로 유사해서 식의 우변에 있는 KL 발산이 작다면 좌변의 오차가 작다는 것을 의미한다. 오차가 작으면 J(CPI)는 J(π') - J(π)을 잘 근사한다. 이것을 이용하면 우리가 원하는 결과, 즉 (J(π') - J(π)) >= 0 을 유도하는 것은 매우 간단한다. 이 이론에 따르면 목적함수는 결코 감소하지 않고 최소한 모든 policy iteration 단계에서 원래 값을 유지하거나 증가하기 때문에 이를 Monotonic Improvement Theory(단조 향상 이론)이라고 한다. 이러한 결과를 얻기 위해 우선 위의 식을 전개하고 하한값을 생각해보자.
   $$
   \begin{aligned}
   J(\pi') - J(\pi) \geq J^{CPI}_\pi(\pi') - C\sqrt{\mathbb{E}_t[KL(\pi'(a_t, s_t)||\pi(a_t, s_t))]} \;\;\;\;\;\; 7.26-1
   \end{aligned}
   $$
   policy iteration의 한 단계 동안 발생 가능한 최악의 상황을 생각해볼 때  다른 정책에 비해 성능이 더 좋은 정책 후보가 없다면 단순히  π' =  π로 설정하고 해당 반복 단계에서는 업데이트를 수행하지 않는다. 때문에 아래 수식과 같이 J(CPI)가 0이 된다.
   $$
   \begin{aligned}
   J(π'=π) - J(π) = J^{CPI}_\pi(\pi' = \pi) = 0
   \end{aligned}
   $$
   그리고 두 정책이 같기 때문에 KL-Divergence도 0이 된다. 식 7.26-1에 따라 정책 변화가 수용되기 위해서는 정책 향상의 추정값 J(CPI)가 최대 오차보다 커야 한다. 지금 다루고 있는 최적화 문제에서 오차의 한계를 일종의 penalty로 추가하면 단조 Monotonic Improvement를 보장할 수 있다. 이제 최적화 문제는 다음과 같아진다. 
   $$
   \text{argmax}_{\pi'}\bigg(J^{CPI}_\pi(\pi') - C\sqrt{\mathbb{E}_t[KL(\pi'(a_t, s_t)||\pi(a_t, s_t))]}\bigg) \\
   \Rightarrow J(\pi') - J(\pi) \geq 0 \;\;\;\;\;\; 7.27
   $$
   이 결과는 의도했던 최종 요구조건을 만족한다. 이로써 원래의 목적함수 J를 사용했을 때 발생할 수 있었던 performance collapse를 피할 수 있게 되었다. 하지만 Monotonic Improvement가 optimal poicy로 수렴함을 보장할 수는 없다는 것이다. 이것은 여전히 학계가 풀어야 할 숙제로 남아있다. 
