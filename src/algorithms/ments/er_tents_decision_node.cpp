#include "algorithms/ments/er_tents_decision_node.h"

#include "helper_templates.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

namespace thts {
    ERTentsDNode::ERTentsDNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ERTentsCNode> parent) :
            TentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const TentsCNode>(parent))
    {
    }

    void ERTentsDNode::compute_action_weights(
        ActionDistr& action_weights,
        double& sum_action_weights,
        double& normalisation_term,
        ThtsEnvContext& context) const
    {
        sum_action_weights = 0.0;
        normalisation_term = 0.0;

        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();

        vector<pair<shared_ptr<const Action>, double>> augmented_q_values;
        augmented_q_values.reserve(actions->size());
        for (shared_ptr<const Action> action : *actions) {
            double augmented_q_value = (get_soft_q_value(action, opp_coeff) + get_er_bonus(action)) / temp;
            augmented_q_values.push_back(make_pair(action, augmented_q_value));
        }

        sort(augmented_q_values.begin(), augmented_q_values.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second > rhs.second;
        });

        vector<shared_ptr<const Action>> sparse_action_set;
        double sum_values = 0.0;
        for (size_t index = 0; index < augmented_q_values.size(); ++index) {
            double value = augmented_q_values[index].second;
            sum_values += value;
            if (1.0 + (static_cast<double>(index) + 1.0) * value > sum_values) {
                sparse_action_set.push_back(augmented_q_values[index].first);
            }
        }

        double sum_sparse_values = 0.0;
        for (shared_ptr<const Action> action : sparse_action_set) {
            for (const auto& pr : augmented_q_values) {
                if (pr.first == action) {
                    sum_sparse_values += pr.second;
                    break;
                }
            }
        }

        double common_term = 0.0;
        if (sparse_action_set.size() > 0) {
            common_term = (sum_sparse_values - 1.0) / static_cast<double>(sparse_action_set.size());
        }

        for (const auto& pr : augmented_q_values) {
            double weight = pr.second - common_term;
            if (weight < 0.0) weight = 0.0;
            action_weights[pr.first] = weight;
            sum_action_weights += weight;
        }

        if (sum_action_weights < 1e-16) {
            double uniform_weight = 1.0 / actions->size();
            for (shared_ptr<const Action> action : *actions) {
                action_weights[action] = uniform_weight;
            }
            sum_action_weights = 1.0;
        }
    }

    shared_ptr<ERTentsCNode> ERTentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<ERTentsCNode>(
            static_pointer_cast<MentsManager>(thts_manager),
            state,
            action,
            decision_depth,
            decision_timestep,
            static_pointer_cast<const ERTentsDNode>(shared_from_this()));
    }

    shared_ptr<ThtsCNode> ERTentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ERTentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }

    ERTentsCNode::ERTentsCNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ERTentsDNode> parent) :
            TentsCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const TentsDNode>(parent))
    {
    }

    shared_ptr<ERTentsDNode> ERTentsCNode::create_child_node_helper(
        shared_ptr<const State> observation,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<ERTentsDNode>(
            static_pointer_cast<MentsManager>(thts_manager),
            mdp_next_state,
            decision_depth + 1,
            decision_timestep + 1,
            static_pointer_cast<const ERTentsCNode>(shared_from_this()));
    }

    shared_ptr<ThtsDNode> ERTentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ERTentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}