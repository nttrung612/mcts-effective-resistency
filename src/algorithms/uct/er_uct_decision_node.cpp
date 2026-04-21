#include "algorithms/uct/er_uct_decision_node.h"

#include <cmath>

using namespace std;

namespace thts {
    ERUctDNode::ERUctDNode(
        shared_ptr<UctManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const UctCNode> parent) :
            UctDNode(thts_manager, state, decision_depth, decision_timestep, parent)
    {
    }

    double ERUctDNode::compute_ucb_term(int num_visits, int child_visits) const {
        shared_ptr<UctManager> manager = static_pointer_cast<UctManager>(thts_manager);
        double num_visits_d = (num_visits > 0) ? (double) num_visits : 1.0;
        double child_visits_d = (child_visits > 0) ? (double) child_visits : 1.0;
        double ucb_term = sqrt(log(num_visits_d) / child_visits_d);
        return ucb_term + (manager->er_c2 / child_visits_d);
    }
}