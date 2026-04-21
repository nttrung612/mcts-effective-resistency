#pragma once

#include "algorithms/uct/fixed_depth_uct_decision_node.h"

#include <memory>

namespace thts {
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
            virtual double compute_ucb_term(int num_visits, int child_visits) const;
    };
}