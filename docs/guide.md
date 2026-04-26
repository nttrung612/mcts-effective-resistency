# CORE IDEAS & ALGORITHMS: EFFECTIVE RESISTANCE AUGMENTED MCTS (MCTS-ER)

This document outlines the core concepts and mathematical formulations of Effective Resistance (ER) augmented Monte-Carlo Tree Search (MCTS) algorithms[cite: 9], specifically designed to guide the implementation of these RL algorithms. Note: The algorithms BTS and DENTS are not covered in this scope as they are not part of the source paper [cite: 8-664]. 

## 1. The Core Concept: Effective Resistance (ER)
* **The Problem:** Traditional MCTS exploration mechanisms depend only on local visit counts, ignoring the global tree structure[cite: 20, 21]. A node might be heavily visited locally but reached via a weakly-explored edge higher in the tree, making the trajectory unreliable[cite: 21].
* **The ER Solution:** ER provides a path-level uncertainty signal by penalizing trajectories containing weakly-visited edges[cite: 10]. On a tree structure, the ER contribution for selecting an action $a$ at state $s_h$ reduces simply to the local quantity $1/T_{s_h,a}(t)$, where $T_{s_h,a}(t)$ is the action visit count[cite: 61].

---

## 2. PART I: Power Mean & Polynomial MCTS (UCT / Fixed-Depth MCTS)
This section covers Stochastic-Power-UCT (often associated with Fixed-Depth MCTS using a polynomial bonus) and its ER-augmented variant[cite: 11, 18, 29].

* **Stochastic-Power-UCT (Base):** Uses a polynomial exploration bonus instead of a logarithmic one, and a power mean operator for value backups[cite: 11, 43].
* **MCTS-ER (ER-Power-UCT / ER-Fixed-Depth):**
    * **Concept:** The ER bonus is directly added to the Upper Confidence Bound (UCB) selection formula[cite: 64, 65].
    * **Selection Formula:** $a_h = \arg \max_{a \in \mathcal{A}_{s_h}} \left\{ \hat{Q}_{T_{s_h,a}(t)}(s_h,a) + C_1 \frac{T_{s_h}(t)^{b_{h+1}/\beta_{h+1}}}{T_{s_h,a}(t)^{\alpha_{h+1}/\beta_{h+1}}} + \frac{C_2}{T_{s_h,a}(t)} \right\}$[cite: 65, 76].
    * **Behavior:** The ER bonus $C_2/T_{s_h,a}(t)$ dominates in the sparse planning regime (early phase) for strong exploration but decays faster than the polynomial bonus asymptotically[cite: 129, 341]. 

---

## 3. PART II: Convex Regularized MCTS (MENTS, RENTS, TENTS)
These algorithms use strongly convex regularizers to achieve exponential convergence[cite: 30, 31, 346]. 

* **ER Integration (General for all three):** Unlike Part I, the ER bonus is NOT added as a standalone UCB term. Instead, it is injected directly into the Q-values BEFORE computing the regularized policy[cite: 379]. 
* **ER-Augmented Q-value:** $Q^{ER}_{\Omega}(s,a) := Q_{\Omega}(s,a) + \frac{c_2}{T_{s,a}(t)}$[cite: 381].

### A. MENTS & ER-MENTS (Maximum Entropy)
* **Base:** Uses Shannon entropy[cite: 365].
* **ER-MENTS Policy:** The ER term acts as a temperature-scaled optimistic shift in the logits[cite: 392].
    $\pi^{ER}(a|s) \propto \exp\left( \frac{Q(s,a) + c_2/T_{s,a}}{\tau} \right)$[cite: 391].

### B. RENTS & ER-RENTS (Relative Entropy)
* **Base:** Uses Kullback-Leibler (KL) divergence relative to the previous policy[cite: 368].
* **ER-RENTS Policy:** Prior-weighted policy with the ER bonus[cite: 393].
    $\pi^{ER}_t(a|s) \propto \pi_{t-1}(a|s) \cdot \exp\left( \frac{Q_t(s,a) + c_2/T_{s,a}}{\tau} \right)$[cite: 393].

### C. TENTS & ER-TENTS (Tsallis Entropy) - *Key Highlight*
* **Base:** Uses Tsallis entropy to generate a sparse policy (assigns exactly zero probability to poor actions)[cite: 370, 371].
* **ER-TENTS Policy:** $\pi^{ER}(a|s) = \max\left\{ \frac{Q(s,a) + c_2/T_{s,a}}{\tau} - \frac{\sum_{a' \in \mathcal{K}_{ER}} (Q(s,a') + c_2/T_{s,a'})/\tau - 1}{|\mathcal{K}_{ER}|}, 0 \right\}$[cite: 395].
* **Adaptive Sparsity Advantage:** By inflating the Q-values of weakly-visited actions with $c_2/T_{s,a}$, the ER bonus dynamically expands the sparsity set $\mathcal{K}_{ER}$ early on (broad exploration). As visits accumulate, the bonus vanishes, contracting the policy back to sparse exploitation[cite: 403]. This creates an adaptive exploration schedule without manual annealing[cite: 404].

---

## 4. Crucial Implementation Notes for Claude
1.  **Strict Separation of Selection and Backup:** The ER bonus must ONLY be used during the action selection phase to dictate the policy/exploration[cite: 387, 388]. The backup operator must update $Q$ using the original, unaugmented values[cite: 387].
2.  **Handling Zero Visits:** Because the ER term divides by $T_{s,a}$, ensure the implementation handles unvisited nodes safely (e.g., initializing $T_{s,a}=1$ or forcing exploration of all actions at least once before applying the ER formula).
3.  **TENTS-ER Solver Implementation:** To compute the TENTS-ER policy, you must dynamically compute the active sparsity set $\mathcal{K}_{ER}$[cite: 396]. This involves sorting the ER-augmented Q-values in descending order and finding the threshold index $i$ where $1 + i \cdot \frac{Q(s,a_i) + c_2/T_{s,a_i}}{\tau} > \sum_{j=1}^{i} \frac{Q(s,a_j) + c_2/T_{s,a_j}}{\tau}$[cite: 401].