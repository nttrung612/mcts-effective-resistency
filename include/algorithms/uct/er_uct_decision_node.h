#pragma once

#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include <memory>

namespace thts {
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
            virtual double compute_ucb_term(int num_visits, int child_visits) const;
    };
}