#include "algorithms/uct/fixed_depth_uct_decision_node.h"

#include <cmath>

using namespace std;

namespace thts {
    FixedDepthUctDNode::FixedDepthUctDNode(
        shared_ptr<UctManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const UctCNode> parent) :
            UctDNode(thts_manager, state, decision_depth, decision_timestep, parent)
    {
    }

    double FixedDepthUctDNode::compute_ucb_term(int num_visits, int child_visits) const {
        double num_visits_d = (num_visits > 0) ? (double) num_visits : 1.0;
        double child_visits_d = (child_visits > 0) ? (double) child_visits : 1.0;
        return pow(num_visits_d, 0.25) / sqrt(child_visits_d);
    }
}