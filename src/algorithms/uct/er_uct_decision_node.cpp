#include "algorithms/uct/er_uct_decision_node.h"

namespace thts {
    ERUctDNode::ERUctDNode(
        std::shared_ptr<UctManager> thts_manager,
        std::shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        std::shared_ptr<const UctCNode> parent) :
            UctDNode(thts_manager, state, decision_depth, decision_timestep, parent)
    {
    }

    double ERUctDNode::compute_ucb_bonus_term(int /*num_visits*/, int child_visits) const {
        std::shared_ptr<UctManager> manager = std::static_pointer_cast<UctManager>(thts_manager);
        if (manager->er_c2 == 0.0) return 0.0;
        double child_visits_d = (child_visits > 0) ? static_cast<double>(child_visits) : 1.0;
        return manager->er_c2 / child_visits_d;
    }
}
