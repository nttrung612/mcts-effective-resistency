#include "algorithms/ments/er_rents_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <limits>

using namespace std;

namespace thts {
    ERRentsDNode::ERRentsDNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ERRentsCNode> parent) :
            RentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const RentsCNode>(parent))
    {
        ThtsEnvContext spoof_ctx;
        cached_action_distr = select_action_alias_tables_get_mixed_distr(spoof_ctx)->get_distr_map();
    }

    void ERRentsDNode::compute_action_weights(
        ActionDistr& action_weights,
        double& sum_action_weights,
        double& normalisation_term,
        ThtsEnvContext& context) const
    {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();

        normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : *actions) {
            double q_value_over_temp = (get_soft_q_value(action, opp_coeff) + get_er_bonus(action)) / temp;
            if (normalisation_term < q_value_over_temp) {
                normalisation_term = q_value_over_temp;
            }
        }

        shared_ptr<ActionDistr> parent_distr = get_parent_distr_from_context(context);

        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : *actions) {
            double soft_q_value = get_soft_q_value(action, opp_coeff) + get_er_bonus(action);
            double action_weight = exp((soft_q_value / temp) - normalisation_term);
            action_weight *= get_parent_action_prob(parent_distr, action);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }

        if (sum_action_weights < 1e-16) {
            double uniform_weight = 1.0 / actions->size();
            for (shared_ptr<const Action> action : *actions) {
                action_weights[action] = uniform_weight;
            }
            sum_action_weights = 1.0;
        }
    }

    shared_ptr<ERRentsCNode> ERRentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<ERRentsCNode>(
            static_pointer_cast<MentsManager>(thts_manager),
            state,
            action,
            decision_depth,
            decision_timestep,
            static_pointer_cast<const ERRentsDNode>(shared_from_this()));
    }

    shared_ptr<ThtsCNode> ERRentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ERRentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }

    ERRentsCNode::ERRentsCNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ERRentsDNode> parent) :
            RentsCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const RentsDNode>(parent))
    {
    }

    shared_ptr<ERRentsDNode> ERRentsCNode::create_child_node_helper(
        shared_ptr<const State> observation,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<ERRentsDNode>(
            static_pointer_cast<MentsManager>(thts_manager),
            mdp_next_state,
            decision_depth + 1,
            decision_timestep + 1,
            static_pointer_cast<const ERRentsCNode>(shared_from_this()));
    }

    shared_ptr<ThtsDNode> ERRentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ERRentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}