#pragma once

#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include <memory>

namespace thts {
    /**
     * UCT augmented with the additive Effective-Resistance (ER) bonus: c2 / N(s,a).
     *
     * Selection at this node is:
     *      opp_coeff * Q(s,a) + bias * sqrt(log N(s) / N(s,a)) + c2 / N(s,a)
     *
     * The ER bonus sits *outside* the bias factor (and the prior factor, when present), as
     * specified by the MCTS-ER paper -- C1 (= bias here) and C2 (= er_c2) are independent additive
     * coefficients. The base log exploration term is inherited from UctDNode unchanged.
     *
     * Note: the MCTS-ER paper formally augments Stochastic-Power-UCT (polynomial bonus) rather than
     * UCT with the log bonus; this class is kept as a "log-UCB + ER" baseline.
     */
    class ERUctDNode : public UctDNode {
        public:
            ERUctDNode(
                std::shared_ptr<UctManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const UctCNode> parent=nullptr);

            virtual ~ERUctDNode() = default;

        protected:
            virtual double compute_ucb_bonus_term(int num_visits, int child_visits) const override;
    };
}
