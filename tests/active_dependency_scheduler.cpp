#include <cassert>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "active_dependency_scheduler.hpp"

using fncs::ActiveGrant;
using fncs::ActiveTimeState;

static ActiveTimeState state(const std::string& name, fncs::time requested) {
    ActiveTimeState value;
    value.name = name;
    value.requested = requested;
    value.pending_time = ULLONG_MAX;
    value.messages_pending = false;
    value.processing = false;
    return value;
}

int main() {
    typedef std::map<std::string, std::set<std::string> > Dependencies;

    std::vector<ActiveTimeState> independent;
    independent.push_back(state("a", 10));
    independent.push_back(state("b", 100));
    std::vector<ActiveGrant> grants =
        fncs::select_active_grants(independent, Dependencies());
    assert(grants.size() == 2);
    assert(grants[0].time == 10);
    assert(grants[1].time == 100);

    Dependencies chain;
    chain["b"].insert("a");
    chain["c"].insert("b");
    std::vector<ActiveTimeState> chained;
    chained.push_back(state("a", 10));
    chained.push_back(state("b", 100));
    chained.push_back(state("c", 1000));
    grants = fncs::select_active_grants(chained, chain);
    assert(grants.size() == 1);
    assert(grants[0].index == 0);
    assert(grants[0].time == 10);

    // A controller-issued frontier certifies that b cannot receive a message
    // before 100, so b may reach its local event without waiting at a's
    // earlier request. The certificate must not alter b's producer bound for
    // downstream consumers.
    fncs::ActiveDependencyGraph bounded =
        fncs::index_active_dependencies(chained, chain);
    bounded.frontier[1] = 100;
    fncs::ActiveSchedulerScratch bounded_scratch;
    const std::vector<ActiveGrant>& bounded_grants =
        fncs::select_active_grants(chained, bounded, &bounded_scratch);
    assert(bounded_grants.size() == 2);
    assert(bounded_grants[0].index == 0);
    assert(bounded_grants[1].index == 1);
    assert(bounded_grants[1].time == 100);

    // A consumer may safely run before its producer's advertised lower bound.
    chained[1].requested = 5;
    grants = fncs::select_active_grants(chained, chain);
    assert(grants.size() == 2);
    assert(grants[0].index == 0);
    assert(grants[1].index == 1);

    // An already queued message is actionable at its sender's grant time.
    chained[1] = state("b", 100);
    chained[1].messages_pending = true;
    chained[1].pending_time = 7;
    grants = fncs::select_active_grants(chained, chain);
    assert(grants.size() == 2);
    assert(grants[1].time == 7);

    Dependencies cycle;
    cycle["a"].insert("b");
    cycle["b"].insert("a");
    std::vector<ActiveTimeState> cyclic;
    cyclic.push_back(state("a", 10));
    cyclic.push_back(state("b", 20));
    grants = fncs::select_active_grants(cyclic, cycle);
    assert(grants.size() == 1);
    assert(grants[0].index == 0);

    // The broker uses an indexed graph and reuses its scratch buffers across
    // scheduling rounds.  It must produce the same grants after state changes.
    fncs::ActiveDependencyGraph indexed =
        fncs::index_active_dependencies(chained, chain);
    fncs::ActiveSchedulerScratch scratch;
    const std::vector<ActiveGrant>& indexed_grants =
        fncs::select_active_grants(chained, indexed, &scratch);
    assert(indexed_grants.size() == 2);
    assert(indexed_grants[0].index == 0);
    assert(indexed_grants[1].index == 1);
    chained[1].messages_pending = false;
    chained[1].pending_time = ULLONG_MAX;
    const std::vector<ActiveGrant>& reused_grants =
        fncs::select_active_grants(chained, indexed, &scratch);
    assert(reused_grants.size() == 1);
    assert(reused_grants[0].index == 0);

    return 0;
}
