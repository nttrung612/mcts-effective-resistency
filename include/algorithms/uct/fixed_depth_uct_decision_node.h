#pragma once

#include "algorithms/uct/uct_decision_node.h"

#include <memory>

namespace thts {
    /**
     * UCT variant with polynomial exploration bonus:
     *      N(s)^(1/4) / N(s,a)^(1/2)
     */
    class FixedDepthUctDNode : public UctDNode {
        public:
            FixedDepthUctDNode(
                std::shared_ptr<UctManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const UctCNode> parent=nullptr);

            virtual ~FixedDepthUctDNode() = default;

        protected:
            virtual double compute_ucb_term(int num_visits, int child_visits) const;
    };
}