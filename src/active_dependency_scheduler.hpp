#ifndef FNCS_ACTIVE_DEPENDENCY_SCHEDULER_HPP
#define FNCS_ACTIVE_DEPENDENCY_SCHEDULER_HPP

#include <algorithm>
#include <climits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "fncs.hpp"

namespace fncs {

struct ActiveTimeState {
    std::string name;
    fncs::time requested;
    fncs::time pending_time;
    bool messages_pending;
    bool processing;
};

struct ActiveGrant {
    size_t index;
    fncs::time time;
};

/** Select a causally safe set of federates to grant concurrently. */
inline std::vector<ActiveGrant> select_active_grants(
        const std::vector<ActiveTimeState>& states,
        const std::map<std::string, std::set<std::string> >& dependencies) {
    const fncs::time infinity = ULLONG_MAX;
    std::map<std::string, size_t> indexes;
    std::vector<fncs::time> actionable(states.size(), infinity);
    std::vector<fncs::time> lower_bound(states.size(), infinity);

    for (size_t i = 0; i < states.size(); ++i) {
        indexes[states[i].name] = i;
        if (!states[i].processing) {
            actionable[i] = states[i].messages_pending
                ? states[i].pending_time : states[i].requested;
            lower_bound[i] = actionable[i];
        }
    }

    // Propagate wake-up bounds transitively.  For example, compute may wake
    // the orchestrator, which may then dispatch network work at the same time.
    for (size_t pass = 0; pass < states.size(); ++pass) {
        bool changed = false;
        for (size_t i = 0; i < states.size(); ++i) {
            if (states[i].processing) {
                continue;
            }
            std::map<std::string, std::set<std::string> >::const_iterator dep =
                dependencies.find(states[i].name);
            if (dep == dependencies.end()) {
                continue;
            }
            fncs::time candidate = lower_bound[i];
            for (std::set<std::string>::const_iterator producer = dep->second.begin();
                    producer != dep->second.end(); ++producer) {
                std::map<std::string, size_t>::const_iterator found = indexes.find(*producer);
                if (found != indexes.end()) {
                    candidate = std::min(candidate, lower_bound[found->second]);
                }
            }
            if (candidate < lower_bound[i]) {
                lower_bound[i] = candidate;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }

    std::vector<ActiveGrant> grants;
    for (size_t i = 0; i < states.size(); ++i) {
        if (!states[i].processing && actionable[i] == lower_bound[i]) {
            grants.push_back(ActiveGrant{i, actionable[i]});
        }
    }

    // Invalid or cyclic equal-time dependencies must not deadlock the broker.
    // The original global-minimum rule is always a conservative fallback.
    if (grants.empty()) {
        fncs::time minimum = infinity;
        for (size_t i = 0; i < actionable.size(); ++i) {
            minimum = std::min(minimum, actionable[i]);
        }
        for (size_t i = 0; i < actionable.size(); ++i) {
            if (!states[i].processing && actionable[i] == minimum) {
                grants.push_back(ActiveGrant{i, minimum});
            }
        }
    }
    return grants;
}

} // namespace fncs

#endif
