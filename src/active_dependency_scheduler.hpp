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

struct ActiveDependencyGraph {
    std::vector<std::vector<size_t> > direct;
    std::vector<std::vector<size_t> > closure;
    // A consumer frontier certifies that it cannot receive a new message
    // before this logical time. Zero means that no explicit certificate was
    // advertised. Frontiers only affect the certified consumer; they are not
    // propagated when that consumer acts as a producer for another federate.
    std::vector<fncs::time> frontier;

    void reset(size_t size) {
        direct.assign(size, std::vector<size_t>());
        closure.assign(size, std::vector<size_t>());
        frontier.assign(size, 0);
    }

    void rebuild_closure() {
        const size_t size = direct.size();
        std::vector<std::vector<bool> > reachable(
            size, std::vector<bool>(size, false));
        for (size_t consumer = 0; consumer < size; ++consumer) {
            for (size_t edge = 0; edge < direct[consumer].size(); ++edge) {
                size_t producer = direct[consumer][edge];
                if (producer < size && producer != consumer) {
                    reachable[consumer][producer] = true;
                }
            }
        }
        for (size_t via = 0; via < size; ++via) {
            for (size_t consumer = 0; consumer < size; ++consumer) {
                if (!reachable[consumer][via]) {
                    continue;
                }
                for (size_t producer = 0; producer < size; ++producer) {
                    if (reachable[via][producer]) {
                        reachable[consumer][producer] = true;
                    }
                }
            }
        }
        closure.assign(size, std::vector<size_t>());
        for (size_t consumer = 0; consumer < size; ++consumer) {
            for (size_t producer = 0; producer < size; ++producer) {
                if (producer != consumer && reachable[consumer][producer]) {
                    closure[consumer].push_back(producer);
                }
            }
        }
    }
};

struct ActiveSchedulerScratch {
    std::vector<fncs::time> actionable;
    std::vector<fncs::time> lower_bound;
    std::vector<ActiveGrant> grants;

    void resize(size_t size) {
        actionable.resize(size);
        lower_bound.resize(size);
        grants.reserve(size);
    }
};

inline ActiveDependencyGraph index_active_dependencies(
        const std::vector<ActiveTimeState>& states,
        const std::map<std::string, std::set<std::string> >& dependencies) {
    ActiveDependencyGraph indexed;
    indexed.reset(states.size());
    std::map<std::string, size_t> indexes;
    for (size_t i = 0; i < states.size(); ++i) {
        indexes[states[i].name] = i;
    }
    for (size_t i = 0; i < states.size(); ++i) {
        std::map<std::string, std::set<std::string> >::const_iterator dep =
            dependencies.find(states[i].name);
        if (dep == dependencies.end()) {
            continue;
        }
        for (std::set<std::string>::const_iterator producer = dep->second.begin();
                producer != dep->second.end(); ++producer) {
            std::map<std::string, size_t>::const_iterator found =
                indexes.find(*producer);
            if (found != indexes.end() && found->second != i) {
                indexed.direct[i].push_back(found->second);
            }
        }
    }
    indexed.rebuild_closure();
    return indexed;
}

/** Select a causally safe set using a pre-indexed dependency graph. */
inline const std::vector<ActiveGrant>& select_active_grants(
        const std::vector<ActiveTimeState>& states,
        const ActiveDependencyGraph& dependencies,
        ActiveSchedulerScratch *scratch) {
    const fncs::time infinity = ULLONG_MAX;
    const size_t size = states.size();
    if (scratch->actionable.size() != size) {
        scratch->resize(size);
    }
    std::fill(scratch->actionable.begin(), scratch->actionable.end(), infinity);
    std::fill(scratch->lower_bound.begin(), scratch->lower_bound.end(), infinity);
    scratch->grants.clear();
    bool has_processing = false;

    for (size_t i = 0; i < size; ++i) {
        if (!states[i].processing) {
            scratch->actionable[i] = states[i].messages_pending
                ? states[i].pending_time : states[i].requested;
            scratch->lower_bound[i] = scratch->actionable[i];
        }
        else {
            has_processing = true;
        }
    }

    if (!has_processing && dependencies.closure.size() == size) {
        // The transitive closure is rebuilt only when a new dependency epoch
        // arrives.  Normal scheduling rounds become a compact indexed scan.
        for (size_t i = 0; i < size; ++i) {
            fncs::time candidate = scratch->actionable[i];
            for (size_t edge = 0; edge < dependencies.closure[i].size(); ++edge) {
                candidate = std::min(
                    candidate,
                    scratch->actionable[dependencies.closure[i][edge]]);
            }
            scratch->lower_bound[i] = candidate;
        }
    }
    else {
        // Preserve the original behavior while disconnected/processing
        // federates are present: dependencies do not propagate through them.
        for (size_t pass = 0; pass < size; ++pass) {
            bool changed = false;
            for (size_t i = 0; i < size; ++i) {
                if (states[i].processing || i >= dependencies.direct.size()) {
                    continue;
                }
                fncs::time candidate = scratch->lower_bound[i];
                for (size_t edge = 0; edge < dependencies.direct[i].size(); ++edge) {
                    size_t producer = dependencies.direct[i][edge];
                    if (producer < size) {
                        candidate = std::min(
                            candidate, scratch->lower_bound[producer]);
                    }
                }
                if (candidate < scratch->lower_bound[i]) {
                    scratch->lower_bound[i] = candidate;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }
    }

    for (size_t i = 0; i < size; ++i) {
        const bool dependency_safe =
            scratch->actionable[i] == scratch->lower_bound[i];
        const bool frontier_safe = dependencies.frontier.size() == size
            && dependencies.frontier[i] > 0
            && scratch->actionable[i] <= dependencies.frontier[i];
        if (!states[i].processing && (dependency_safe || frontier_safe)) {
            ActiveGrant grant = {i, scratch->actionable[i]};
            scratch->grants.push_back(grant);
        }
    }

    // Invalid or cyclic equal-time dependencies must not deadlock the broker.
    // The original global-minimum rule is always a conservative fallback.
    if (scratch->grants.empty()) {
        fncs::time minimum = infinity;
        for (size_t i = 0; i < size; ++i) {
            minimum = std::min(minimum, scratch->actionable[i]);
        }
        for (size_t i = 0; i < size; ++i) {
            if (!states[i].processing && scratch->actionable[i] == minimum) {
                ActiveGrant grant = {i, minimum};
                scratch->grants.push_back(grant);
            }
        }
    }
    return scratch->grants;
}

/** Compatibility wrapper for callers that keep name-based dependencies. */
inline std::vector<ActiveGrant> select_active_grants(
        const std::vector<ActiveTimeState>& states,
        const std::map<std::string, std::set<std::string> >& dependencies) {
    ActiveDependencyGraph indexed = index_active_dependencies(states, dependencies);
    ActiveSchedulerScratch scratch;
    const std::vector<ActiveGrant>& grants =
        select_active_grants(states, indexed, &scratch);
    return grants;
}

} // namespace fncs

#endif
