"""
Multi-Armed Bandit — Reinforcement Learning Project
====================================================
The Multi-Armed Bandit (MAB) is a classic RL problem that
models the explore-exploit tradeoff in its purest form.

Real-world applications:
  - A/B testing & website optimization
  - Clinical drug trials
  - Online ad bidding
  - Recommendation systems
  - Hyperparameter tuning

Algorithms implemented:
  1. Pure Greedy           — always pick best known arm
  2. Epsilon-Greedy        — explore randomly with prob ε
  3. Epsilon-Greedy Decay  — ε shrinks over time
  4. UCB1 (Upper Confidence Bound) — explore based on uncertainty
  5. Thompson Sampling     — Bayesian, sample from posterior

Usage:
  python multi_armed_bandit.py
"""

import numpy as np
import random
from abc import ABC, abstractmethod


# ── Bandit Environment ─────────────────────────────────────────────────────────

class BernoulliBandit:
    """
    A k-armed bandit where each arm gives reward 1 with some
    fixed (unknown) probability, and 0 otherwise.
    """

    def __init__(self, k=10, seed=42):
        rng = np.random.default_rng(seed)
        self.k = k
        self.true_probs = rng.uniform(0.1, 0.9, k)
        self.best_arm   = int(np.argmax(self.true_probs))
        self.best_prob  = self.true_probs[self.best_arm]

    def pull(self, arm: int) -> float:
        """Pull an arm; returns 1.0 (win) or 0.0 (loss)."""
        return float(np.random.random() < self.true_probs[arm])

    def regret(self, arm: int) -> float:
        """Opportunity cost vs always playing the best arm."""
        return self.best_prob - self.true_probs[arm]

    def __repr__(self):
        probs = ", ".join(f"{p:.2f}" for p in self.true_probs)
        return f"BernoulliBandit(k={self.k}, probs=[{probs}], best={self.best_arm})"


# ── Base Agent ─────────────────────────────────────────────────────────────────

class BanditAgent(ABC):
    def __init__(self, k: int):
        self.k      = k
        self.counts = np.zeros(k, dtype=int)   # pulls per arm
        self.values = np.zeros(k)              # estimated Q(a)
        self.t      = 0                         # total steps

    def update(self, arm: int, reward: float):
        """Incremental mean update: Q(a) ← Q(a) + (r - Q(a)) / N(a)"""
        self.counts[arm] += 1
        self.t += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    @abstractmethod
    def select_arm(self) -> int:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── Agents ─────────────────────────────────────────────────────────────────────

class GreedyAgent(BanditAgent):
    """Always exploits; never explores. Gets stuck on first lucky arm."""

    def select_arm(self) -> int:
        # Explore all arms once before going greedy
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited):
            return unvisited[0]
        return int(np.argmax(self.values))

    @property
    def name(self): return "Greedy"


class EpsilonGreedyAgent(BanditAgent):
    """With prob ε explore randomly; with prob 1-ε exploit best known arm."""

    def __init__(self, k: int, epsilon: float = 0.1):
        super().__init__(k)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.k)
        return int(np.argmax(self.values))

    @property
    def name(self): return f"Epsilon-Greedy(ε={self.epsilon})"


class EpsilonGreedyDecayAgent(BanditAgent):
    """Epsilon decays over time: more exploration early, exploitation later."""

    def __init__(self, k: int, initial_epsilon: float = 1.0, decay: float = 0.995):
        super().__init__(k)
        self.epsilon = initial_epsilon
        self.decay   = decay
        self.min_eps = 0.01

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.k)
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        super().update(arm, reward)
        self.epsilon = max(self.min_eps, self.epsilon * self.decay)

    @property
    def name(self): return f"Epsilon-Greedy-Decay(ε0=1.0, decay={self.decay})"


class UCB1Agent(BanditAgent):
    """
    Upper Confidence Bound — selects arm with highest UCB score:
        UCB(a) = Q(a) + sqrt(2 * ln(t) / N(a))

    The bonus term decreases as an arm is pulled more,
    balancing exploration and exploitation automatically.
    """

    def __init__(self, k: int, c: float = 2.0):
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited):
            return unvisited[0]
        ucb_scores = self.values + self.c * np.sqrt(
            np.log(self.t + 1) / (self.counts + 1e-9)
        )
        return int(np.argmax(ucb_scores))

    @property
    def name(self): return f"UCB1(c={self.c})"


class ThompsonSamplingAgent(BanditAgent):
    """
    Bayesian approach using Beta distribution as posterior over arm probs.
    Beta(α, β) where α = successes + 1, β = failures + 1.
    Sample θ̂ from each arm's posterior; pull the arm with highest θ̂.
    """

    def __init__(self, k: int):
        super().__init__(k)
        self.successes = np.zeros(k)
        self.failures  = np.zeros(k)

    def select_arm(self) -> int:
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        super().update(arm, reward)
        if reward > 0:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

    @property
    def name(self): return "Thompson Sampling"


# ── Simulation Runner ──────────────────────────────────────────────────────────

def run_simulation(agent: BanditAgent, bandit: BernoulliBandit, n_steps: int):
    """
    Run one agent on a bandit for n_steps.
    Returns arrays of rewards, cumulative regret, and best-arm selection.
    """
    rewards      = np.zeros(n_steps)
    cum_regret   = np.zeros(n_steps)
    best_arm_sel = np.zeros(n_steps, dtype=bool)
    total_regret = 0.0

    for t in range(n_steps):
        arm    = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)

        rewards[t]      = reward
        total_regret   += bandit.regret(arm)
        cum_regret[t]   = total_regret
        best_arm_sel[t] = (arm == bandit.best_arm)

    return rewards, cum_regret, best_arm_sel


# ── Results Printer ────────────────────────────────────────────────────────────

def print_results(agent_name, rewards, cum_regret, best_arm_sel, n_steps):
    total_reward   = rewards.sum()
    final_regret   = cum_regret[-1]
    best_arm_pct   = best_arm_sel.mean() * 100
    # Moving average over last 100 steps
    late_avg       = rewards[-100:].mean()

    print(f"\n{'─'*55}")
    print(f"  {agent_name}")
    print(f"{'─'*55}")
    print(f"  Total reward      : {total_reward:.1f}  ({total_reward/n_steps:.3f} per step)")
    print(f"  Cumulative regret : {final_regret:.2f}")
    print(f"  Best arm chosen   : {best_arm_pct:.1f}% of the time")
    print(f"  Late avg reward   : {late_avg:.3f}  (last 100 steps)")


# ── Q-value Inspector ──────────────────────────────────────────────────────────

def print_q_table(agent: BanditAgent, bandit: BernoulliBandit):
    print(f"\n  Learned Q-values vs. True probabilities:")
    print(f"  {'Arm':>4} | {'True p':>7} | {'Q(a)':>7} | {'Pulls':>6} | {'Error':>7}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")
    for i in range(bandit.k):
        marker = " <-- BEST" if i == bandit.best_arm else ""
        err = abs(agent.values[i] - bandit.true_probs[i])
        print(f"  {i+1:>4} | {bandit.true_probs[i]:>7.3f} | {agent.values[i]:>7.3f} | "
              f"{agent.counts[i]:>6} | {err:>7.4f}{marker}")


# ── Comparative Benchmark ──────────────────────────────────────────────────────

def benchmark(k=10, n_steps=1000, n_runs=200, seed=42):
    """
    Average results over multiple random seeds to get stable estimates.
    Each run uses a *different* bandit instance (different true probs).
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark: k={k} arms, {n_steps} steps, {n_runs} runs each")
    print(f"{'='*60}")

    agents_cfg = [
        lambda k: GreedyAgent(k),
        lambda k: EpsilonGreedyAgent(k, epsilon=0.01),
        lambda k: EpsilonGreedyAgent(k, epsilon=0.10),
        lambda k: EpsilonGreedyDecayAgent(k),
        lambda k: UCB1Agent(k),
        lambda k: ThompsonSamplingAgent(k),
    ]

    results = []
    for cfg in agents_cfg:
        all_regrets, all_best = [], []
        for run in range(n_runs):
            bandit = BernoulliBandit(k=k, seed=seed + run)
            agent  = cfg(k)
            _, cum_regret, best_arm_sel = run_simulation(agent, bandit, n_steps)
            all_regrets.append(cum_regret[-1])
            all_best.append(best_arm_sel.mean() * 100)
        name = cfg(k).name
        mean_reg  = np.mean(all_regrets)
        mean_best = np.mean(all_best)
        results.append((name, mean_reg, mean_best))

    print(f"\n  {'Algorithm':<35} | {'Avg Regret':>10} | {'Best Arm %':>10}")
    print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}")
    for name, reg, best in sorted(results, key=lambda x: x[1]):
        print(f"  {name:<35} | {reg:>10.2f} | {best:>9.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Armed Bandit — Reinforcement Learning Demo")
    print("=" * 60)

    N_ARMS  = 10
    N_STEPS = 1000
    SEED    = 7

    bandit = BernoulliBandit(k=N_ARMS, seed=SEED)
    print(f"\n{bandit}")
    print(f"Best arm: {bandit.best_arm + 1}  (true prob = {bandit.best_prob:.2f})")

    agents = [
        GreedyAgent(N_ARMS),
        EpsilonGreedyAgent(N_ARMS, epsilon=0.01),
        EpsilonGreedyAgent(N_ARMS, epsilon=0.10),
        EpsilonGreedyDecayAgent(N_ARMS),
        UCB1Agent(N_ARMS),
        ThompsonSamplingAgent(N_ARMS),
    ]

    print(f"\nRunning {len(agents)} agents for {N_STEPS} steps each...\n")
    for agent in agents:
        np.random.seed(SEED)
        rewards, cum_regret, best_arm_sel = run_simulation(agent, bandit, N_STEPS)
        print_results(agent.name, rewards, cum_regret, best_arm_sel, N_STEPS)

    print(f"\n{'─'*55}")
    print("  Learned Q-table (Thompson Sampling — last agent run):")
    print_q_table(agents[-1], bandit)

    benchmark(k=N_ARMS, n_steps=N_STEPS, n_runs=300, seed=SEED)

    print("\n\nKey concepts illustrated:")
    print("  - Explore-exploit tradeoff: all strategies balance these differently")
    print("  - Regret: gap between your reward and the optimal (omniscient) agent")
    print("  - UCB1 & Thompson Sampling achieve O(log T) regret (near-optimal)")
    print("  - Pure greedy achieves O(T) regret — linear, never stops paying")
    print("  - Thompson Sampling is Bayesian: maintains a distribution over beliefs")
