#pragma once

#include <cmath>

namespace thts {
    namespace helper {
        inline double power_mean_transform(double value, double power_mean_p) {
            if (std::fabs(power_mean_p - 1.0) < 1e-12) {
                return value;
            }
            return (value >= 0.0) ? std::pow(value, power_mean_p) : -std::pow(-value, power_mean_p);
        }

        inline double power_mean_inverse(double value, double power_mean_p) {
            if (std::fabs(power_mean_p - 1.0) < 1e-12) {
                return value;
            }
            return (value >= 0.0) ? std::pow(value, 1.0 / power_mean_p) : -std::pow(-value, 1.0 / power_mean_p);
        }
    }
}