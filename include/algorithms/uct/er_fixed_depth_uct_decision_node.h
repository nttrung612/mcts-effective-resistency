#pragma once

#include "algorithms/uct/fixed_depth_uct_decision_node.h"

#include <memory>

namespace thts {
    /**
     * Stochastic-Power-UCT augmented with the additive Effective-Resistance (ER) bonus c2 / N(s,a).
     *
     * Selection at this node is (matching MCTS-ER Eq. (combined_bonus)):
     *      Q + bias * (N(s)^{1/4} / N(s,a)^{1/2})  +  c2 / N(s,a)
     *
     * The polynomial exploration term is inherited from FixedDepthUctDNode and remains scaled by
     * `bias` (= C1). The ER bonus c2 / N(s,a) is added *outside* the bias and prior factors so that
     * C1 and C2 act as independent additive coefficients, as in the paper. The power-mean backup
     * (governed by manager.power_mean_p) lives in UctDNode::backup_average_return and is unchanged.
     */
    class ERFixedDepthUctDNode : public FixedDepthUctDNode {
        public:
            ERFixedDepthUctDNode(
                std::shared_ptr<UctManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const UctCNode> parent=nullptr);

            virtual ~ERFixedDepthUctDNode() = default;

        protected:
            virtual double compute_ucb_bonus_term(int num_visits, int child_visits) const override;
    };
}
