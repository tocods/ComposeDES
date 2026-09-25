/* autoconf header */
#include "config.h"

/* C++ standard headers */
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <vector>

/* 3rd party headers */
#include "czmq.h"

/* fncs headers */
#include "log.hpp"
#include "fncs.hpp"
#include "fncs_internal.hpp"
#include "active_dependency_scheduler.hpp"

using namespace ::std;

class SimulatorState {
    public:
        SimulatorState()
            : name("")
            , time_requested(0)
            , time_delta(0)
            , time_last_processed(0)
            , processing(true)
            , messages_pending(false)
            , pending_time(ULLONG_MAX)
            , current_grant(0)
            , grant_count(0)
        {}

        string name;
        fncs::time time_requested;
        fncs::time time_delta;
        fncs::time time_last_processed;
        bool processing;
        bool messages_pending;
        fncs::time pending_time;
        fncs::time current_grant;
        unsigned long long grant_count;
        set<string> subscription_values;
};

typedef map<string,size_t> SimIndex;
typedef vector<SimulatorState> SimVec;
typedef vector<size_t> IndexVec;
typedef vector<fncs::time> TimeVec;
typedef map<string,IndexVec> TopicMap;
typedef map<string,set<string> > SimKeyMap;
typedef map<string,TimeVec> SimTimeMap;

static fncs::time time_real_start;
static fncs::time time_real;
static ofstream trace; /* the trace stream, if requested */

static bool parse_dependency_update(
        const string &value,
        const SimIndex &name_to_index,
        fncs::ActiveDependencyGraph *dependencies,
        unsigned long long *epoch) {
    istringstream input(value);
    string line;
    fncs::ActiveDependencyGraph parsed;
    parsed.reset(name_to_index.size());
    unsigned long long parsed_epoch = 0;
    bool found_epoch = false;
    while (getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        size_t equals = line.find('=');
        if (equals == string::npos) {
            return false;
        }
        string consumer = line.substr(0, equals);
        string producers = line.substr(equals + 1);
        if (consumer == "epoch") {
            istringstream parser(producers);
            parser >> parsed_epoch;
            if (!parser || !parser.eof()) {
                return false;
            }
            found_epoch = true;
            continue;
        }
        const string frontier_prefix = "frontier.";
        if (consumer.compare(0, frontier_prefix.size(), frontier_prefix) == 0) {
            string simulator = consumer.substr(frontier_prefix.size());
            if (name_to_index.count(simulator) == 0
                    || producers.empty()
                    || producers.find_first_not_of("0123456789") != string::npos) {
                return false;
            }
            fncs::time parsed_frontier = 0;
            istringstream parser(producers);
            parser >> parsed_frontier;
            if (!parser || !parser.eof() || parsed_frontier == 0) {
                return false;
            }
            parsed.frontier[name_to_index.find(simulator)->second] =
                parsed_frontier;
            continue;
        }
        if (name_to_index.count(consumer) == 0) {
            return false;
        }
        size_t consumer_index = name_to_index.find(consumer)->second;
        istringstream producer_stream(producers);
        string producer;
        while (getline(producer_stream, producer, ',')) {
            if (producer.empty()) {
                continue;
            }
            if (name_to_index.count(producer) == 0 || producer == consumer) {
                return false;
            }
            size_t producer_index = name_to_index.find(producer)->second;
            if (find(parsed.direct[consumer_index].begin(),
                    parsed.direct[consumer_index].end(), producer_index)
                    == parsed.direct[consumer_index].end()) {
                parsed.direct[consumer_index].push_back(producer_index);
            }
        }
    }
    if (!found_epoch || parsed_epoch <= *epoch) {
        return false;
    }
    parsed.rebuild_closure();
    *dependencies = parsed;
    *epoch = parsed_epoch;
    return true;
}

static void write_coordination_metrics(
        const char *path,
        bool active,
        unsigned long long rounds,
        unsigned long long dependency_updates,
        const SimVec &simulators) {
    if (!path || !path[0]) {
        return;
    }
    ofstream output(path);
    if (!output.is_open()) {
        LERROR << "Could not open coordination metrics file '" << path << "'";
        return;
    }
    unsigned long long total_grants = 0;
    for (size_t i = 0; i < simulators.size(); ++i) {
        total_grants += simulators[i].grant_count;
    }
    output << "{\n"
        << "  \"mode\": \"" << (active ? "active_dependency" : "global") << "\",\n"
        << "  \"scheduler_rounds\": " << rounds << ",\n"
        << "  \"dependency_updates\": " << dependency_updates << ",\n"
        << "  \"total_grants\": " << total_grants << ",\n"
        << "  \"grants_by_simulator\": {";
    for (size_t i = 0; i < simulators.size(); ++i) {
        output << (i == 0 ? "\n" : ",\n")
            << "    \"" << simulators[i].name << "\": "
            << simulators[i].grant_count;
    }
    output << "\n  }\n}\n";
}

static void broker_die(const SimVec &simulators, zsock_t *server) {
    /* repeat the fatal die to all connected sims */
    for (size_t i=0; i<simulators.size(); ++i) {
        zstr_sendm(server, simulators[i].name.c_str());
        zstr_send(server, fncs::DIE);
    }
    zsock_destroy(&server);
    zsys_shutdown(); /* without this, Windows will assert */
    if (trace.is_open()) {
        trace.close();
    }
    exit(EXIT_FAILURE);
}

static void time_real_update(void)
{
    time_real = fncs::timer_ft() - time_real_start;
}




int main(int argc, char **argv)
{
    /* declare all variables */
    unsigned int n_sims = 0;    /* how many sims will connect */
    set<string> byes;           /* which sims have disconnected */
    int n_processing = 0;       /* how many sims are processing a time step */
    const char *endpoint = NULL;/* broker location */
    SimVec simulators;          /* vector of connected simulator state */
    SimIndex name_to_index;     /* quickly lookup sim state index */
    TopicMap topic_to_indexes;  /* quickly lookup subscribed sims */
    SimKeyMap name_to_keys;     /* summary of topics per sim name */
    SimKeyMap name_to_peers;    /* summary of peers per sim name */
    SimTimeMap name_to_peertimes; /* summary of peer deltas */
    fncs::time time_granted = 0;/* global clock */
    zsock_t *server = NULL;     /* the broker socket */
    bool do_trace = false;      /* whether to dump all received messages */
    fncs::time realtime_interval = 0;
    bool active_dependency_mode = false;
    fncs::ActiveDependencyGraph active_dependencies;
    vector<fncs::ActiveTimeState> active_states;
    fncs::ActiveSchedulerScratch active_scheduler_scratch;
    unsigned long long dependency_epoch = 0;
    unsigned long long dependency_updates = 0;
    unsigned long long scheduler_rounds = 0;
    const char *coordination_metrics_path = getenv("FNCS_COORDINATION_METRICS");

    {
        const char *env_active = getenv("FNCS_ACTIVE_DEPENDENCY");
        active_dependency_mode = env_active && string(env_active) != "0"
            && string(env_active) != "false" && string(env_active) != "no";
    }

    fncs::start_logging();
    fncs::replicate_logging(FNCSLog::ReportingLevel(),
            Output2Tee::Stream1(), Output2Tee::Stream2());

    /* how many simulators are connecting? */
    if (argc > 3) {
        LERROR << "too many command line args";
        exit(EXIT_FAILURE);
    }
    if (argc < 2) {
        LERROR << "missing command line arg for number of simulators";
        exit(EXIT_FAILURE);
    }
    if (argc >= 2) {
        int n_sims_signed = 0;
        istringstream iss(argv[1]);
        iss >> n_sims_signed;
        LDEBUG4 << "n_sims_signed = " << n_sims_signed;
        if (n_sims_signed <= 0) {
            LERROR << "number of simulators arg must be >= 1";
            exit(EXIT_FAILURE);
        }
        n_sims = static_cast<unsigned int>(n_sims_signed);
    }
    if (argc == 3) {
        realtime_interval = fncs::parse_time(argv[2]);
        LDEBUG4 << "realtime_interval = " << realtime_interval << " ns";
    }

    {
        const char *env_do_trace = getenv("FNCS_TRACE");
        if (env_do_trace) {
            if (env_do_trace[0] == 'Y'
                    || env_do_trace[0] == 'y'
                    || env_do_trace[0] == 'T'
                    || env_do_trace[0] == 't') {
                do_trace = true;
            }
        }
    }

    if (do_trace) {
        LDEBUG4 << "tracing of all published messages enabled";
        trace.open("broker_trace.txt");
        if (!trace) {
            LERROR << "Could not open trace file 'broker_trace.txt'";
            exit(EXIT_FAILURE);
        }
        trace << "#nanoseconds\ttopic\tvalue" << endl;
    }

    /* broker endpoint may come from env var */
    endpoint = getenv("FNCS_BROKER");
    if (!endpoint) {
        endpoint = "tcp://*:5570";
    }

    server = zsock_new_router(endpoint);
    if (!server) {
        LERROR << "socket creation failed";
        exit(EXIT_FAILURE);
    }
    if (!(zsock_resolve(server) != server)) {
        LERROR << "socket failed to resolve";
        exit(EXIT_FAILURE);
    }
    LDEBUG4 << "broker socket bound to " << endpoint;

    /* begin event loop */
    zmq_pollitem_t items[] = { { zsock_resolve(server), 0, ZMQ_POLLIN, 0 } };
    while (true) {
        int rc = 0;
        
        LDEBUG4 << "entering blocking poll";
        rc = zmq_poll(items, 1, -1);
        if (rc == -1) {
            LERROR << "broker polling error: " << strerror(errno);
            broker_die(simulators, server); /* interrupted */
        }

        if (items[0].revents & ZMQ_POLLIN) {
            zmsg_t *msg = NULL;
            zframe_t *frame = NULL;
            string sender;
            string message_type;

            LDEBUG4 << "incoming message";
            msg = zmsg_recv(server);
            if (!msg) {
                LERROR << "null message received";
                broker_die(simulators, server);
            }

            /* first frame is sender */
            frame = zmsg_first(msg);
            if (!frame) {
                LERROR << "message missing sender";
                broker_die(simulators, server);
            }
            sender = fncs::to_string(frame);

            /* next frame is message type identifier */
            frame = zmsg_next(msg);
            if (!frame) {
                LERROR << "message missing type identifier";
                broker_die(simulators, server);
            }
            message_type = fncs::to_string(frame);
            
            /* dispatcher */
            if (fncs::HELLO == message_type) {
                SimulatorState state;
                string config_string;
                fncs::Config config;
                string time_delta;
                size_t index = 0;

                LDEBUG4 << "HELLO received";

                /* check for duplicate sims */
                if (name_to_index.count(sender) != 0) {
                    LERROR << "simulator '" << sender << "' already connected";
                    broker_die(simulators, server);
                }
                index = simulators.size();
                LDEBUG4 << "registering client '" << sender << "'";

                /* next frame is config chunk */
                frame = zmsg_next(msg);
                if (!frame) {
                    LERROR << "HELLO message missing config frame";
                    broker_die(simulators, server);
                }

                /* copy config frame into chunk */
                config_string = fncs::to_string(frame);
                LDEBUG2 << "-- recv configuration as follows --" << endl << config_string;

                /* next frame is FNCS library version */
                frame = zmsg_next(msg);
                if (!frame) {
                    LWARNING << "HELLO message from '" << sender << "' missing FNCS library version";
                }
                else {
                    string version_string = fncs::to_string(frame);
                    int result[3];
                    std::istringstream parser(version_string);
                    parser >> result[0];
                    for(int idx = 1; idx < 3; idx++) {
                        parser.get(); //Skip period
                        parser >> result[idx];
                    }
                    if (FNCS_VERSION_MAJOR != result[0]
                            || FNCS_VERSION_MINOR != result[1]
                            || FNCS_VERSION_PATCH != result[2]) {
                        LWARNING << "FNCS library version mismatch, client="
                            << version_string
                            << " broker="
                            << FNCS_VERSION_MAJOR << "."
                            << FNCS_VERSION_MINOR << "."
                            << FNCS_VERSION_PATCH;
                    }
                }

                /* parse config chunk */
                config = fncs::parse_config(config_string);

                /* get time delta from config */
                time_delta = config.time_delta;
                if (time_delta.empty()) {
                    LWARNING << sender << " config does not contain 'time_delta'";
                    LWARNING << sender << " time_delta defaulting to 1s";
                    time_delta = "1s";
                }
                state.time_delta = fncs::parse_time(time_delta);

                /* parse subscription values */
                set<string> subscription_values;
                if (!config.values.empty()) {
                    vector<fncs::Subscription> &subs = config.values;
                    set<string> peers;
                    for (size_t i=0; i<subs.size(); ++i) {
                        string topic = subs[i].topic;
                        LDEBUG4 << "adding value '" << topic << "'";
                        subscription_values.insert(topic);
                        TopicMap::iterator it = topic_to_indexes.find(topic);
                        if (it != topic_to_indexes.end()) {
                            it->second.push_back(index);
                        }
                        else {
                            topic_to_indexes[topic] = IndexVec(1,index);
                        }
                        size_t loc = topic.find('/');
                        if (loc == string::npos) {
                            LWARNING << "invalid topic: " << topic;
                        }
                        else {
                            string name = topic.substr(0,loc);
                            string key = topic.substr(loc+1);
                            name_to_keys[name].insert(key);
                            LDEBUG4 << "name_to_keys[" << name << "]=" << key;
                            peers.insert(name);
                        }
                    }
                    for (set<string>::iterator it=peers.begin();
                            it!=peers.end(); ++it) {
                        SimTimeMap::iterator stm = name_to_peertimes.find(*it);
                        if (stm == name_to_peertimes.end()) {
                            name_to_peertimes[*it] = TimeVec(1, state.time_delta);
                        }
                        else {
                            name_to_peertimes[*it].push_back(state.time_delta);
                        }
                    }
                    name_to_peers[sender] = peers;
                }
                else {
                    LDEBUG4 << "no subscription values";
                }

                /* populate sim state object */
                state.name = sender;
                /*state.time_delta = fncs::parse_time(time_delta);*/ /*above*/
                state.time_requested = 0;
                state.time_last_processed = 0;
                state.processing = false;
                state.messages_pending = false;
                state.subscription_values = subscription_values;
                name_to_index[sender] = index;
                simulators.push_back(state);

                LDEBUG4 << "simulators.size() = " << simulators.size();

                /* if all sims have connected, send the go-ahead */
                if (simulators.size() == n_sims) {
                    active_dependencies.reset(n_sims);
                    active_states.resize(n_sims);
                    active_scheduler_scratch.resize(n_sims);
                    for (size_t i=0; i<n_sims; ++i) {
                        active_states[i].name = simulators[i].name;
                    }
                    time_real_start = fncs::timer_ft();
                    time_real = 0;
                    if (realtime_interval) {
#ifdef _WIN32
                        cerr << "realtime clock not yet supported on WIN32" << endl;
                        exit(EXIT_FAILURE);
#else
                        struct itimerval it_val;  /* for setting itimer */

                        /* setitimer call needs seconds and useconds */
                        if (signal(SIGALRM, (void (*)(int)) time_real_update) == SIG_ERR) {
                            perror("Unable to catch SIGALRM");
                            exit(EXIT_FAILURE);
                        }
                        it_val.it_value.tv_sec = realtime_interval/1000000000UL;
                        LDEBUG4 << "realtime_sec = " << it_val.it_value.tv_sec;
                        it_val.it_value.tv_usec = realtime_interval/1000 % 1000000;
                        LDEBUG4 << "realtime_usec = " << it_val.it_value.tv_usec;
                        it_val.it_interval = it_val.it_value;
                        if (setitimer(ITIMER_REAL, &it_val, NULL) == -1) {
                            broker_die(simulators, server);
                        }
#endif
                    }
                    /* easier to keep a counter than iterating over states */
                    n_processing = n_sims;
                    /* send ACK to all registered sims */
                    for (size_t i=0; i<n_sims; ++i) {
                        set<string> &keys = name_to_keys[simulators[i].name];
                        simulators[i].processing = true;
                        LDEBUG4 << "sending first ACK to " << simulators[i].name;
                        zstr_sendm(server, simulators[i].name.c_str());
                        zstr_sendm(server, fncs::ACK);
                        zstr_sendfm(server, "%llu", (unsigned long long)i);
                        zstr_sendfm(server, "%llu", (unsigned long long)n_sims);
                        zstr_sendfm(server, "%llu", (unsigned long long)keys.size());
                        for (set<string>::iterator it=keys.begin(); it!=keys.end(); ++it) {
                            zstr_sendm(server, it->c_str());
                        }
                        /* smallest delta of any clients */
                        {
                            fncs::time time_peer = 0;
                            TimeVec &peertimes = name_to_peertimes[simulators[i].name];
                            set<string> &peers = name_to_peers[simulators[i].name];
                            for (set<string>::iterator it=peers.begin();
                                    it!=peers.end(); ++it) {
                                SimIndex::iterator simit = name_to_index.find(*it);
                                if (simit != name_to_index.end()) {
                                    size_t index = simit->second;
                                    peertimes.push_back(simulators[index].time_delta);
                                }
                            }
                            if (!peertimes.empty()) {
                                time_peer = *min_element(
                                        peertimes.begin(),
                                        peertimes.end());
                            }
                            LDEBUG4 << "time_peer = " << time_peer;
                            LDEBUG4 << "time_delta= " << simulators[i].time_delta;
                            zstr_sendfm(server, "%llu", (unsigned long long)time_peer);
                        }
                        zstr_sendfm(server, "%d.%d.%d", FNCS_VERSION_MAJOR, FNCS_VERSION_MINOR, FNCS_VERSION_PATCH);
                        zstr_send(server, fncs::ACK);
                        LDEBUG4 << "ACK sent to '" << simulators[i].name;
                    }
                }
            }
            else if (fncs::TIME_REQUEST == message_type
                    || fncs::BYE == message_type) {
                size_t index = 0; /* index of sim state */
                fncs::time time_requested;
                fncs::time time_last;

                if (fncs::TIME_REQUEST == message_type) {
                    LDEBUG4 << "TIME_REQUEST received from " << sender;
                }
                else if (fncs::BYE == message_type) {
                    LDEBUG4 << "BYE received";
                }

                /* did we receive message from a connected sim? */
                if (name_to_index.count(sender) == 0) {
                    LERROR << "simulator '" << sender << "' not connected";
                    broker_die(simulators, server);
                }
                /* index of sim state */
                index = name_to_index[sender];

                if (fncs::BYE == message_type) {
                    /* next frame is time last processed */
                    frame = zmsg_next(msg);
                    if (!frame) {
                        LERROR << "BYE message missing time last frame";
                        broker_die(simulators, server);
                    }
                    /* convert time string */
                    {
                        istringstream iss(fncs::to_string(frame));
                        iss >> time_last;
                    }

                    /* soft error if muliple byes received */
                    if (byes.count(sender)) {
                        LWARNING << "duplicate BYE from '" << sender << "'";
                    }

                    /* add sender to list of leaving sims */
                    byes.insert(sender);

                    /* if all byes received, then exit */
                    if (byes.size() == n_sims) {
                        /* let all sims know that globally we are finished */
                        for (size_t i=0; i<n_sims; ++i) {
                            zstr_sendm(server, simulators[i].name.c_str());
                            zstr_send(server, fncs::BYE);
                            LDEBUG4 << "BYE sent to '" << simulators[i].name;
                        }
                        /* need to delete msg since we are breaking from loop */
                        zmsg_destroy(&msg);
                        write_coordination_metrics(
                            coordination_metrics_path,
                            active_dependency_mode,
                            scheduler_rounds,
                            dependency_updates,
                            simulators);
                        break;
                    }

                    /* update sim state */
                    simulators[index].time_requested = ULLONG_MAX;
                }
                else if (fncs::TIME_REQUEST == message_type) {
                    /* next frame is time requested */
                    frame = zmsg_next(msg);
                    if (!frame) {
                        LERROR << "TIME_REQUEST message missing time request frame";
                        broker_die(simulators, server);
                    }
                    /* convert time string */
                    {
                        istringstream iss(fncs::to_string(frame));
                        iss >> time_requested;
                    }
                    /* next frame is time last processed */
                    frame = zmsg_next(msg);
                    if (!frame) {
                        LERROR << "TIME_REQUEST message missing time last frame";
                        broker_die(simulators, server);
                    }
                    /* convert time string */
                    {
                        istringstream iss(fncs::to_string(frame));
                        iss >> time_last;
                    }

                    /* update sim state */
                    simulators[index].time_requested = time_requested;

                    LDEBUG4 << "TIME_REQUEST " << sender << " requested " << time_requested;
                }

                /* update sim state */
                simulators[index].time_last_processed = time_last;
                simulators[index].processing = false;

                --n_processing;

                /* if all sims are done, determine next time step */
                if (0 == n_processing) {
                    ++scheduler_rounds;
                    if (active_dependency_mode) {
                        for (size_t i=0; i<n_sims; ++i) {
                            active_states[i].requested = simulators[i].time_requested;
                            active_states[i].pending_time = simulators[i].pending_time;
                            active_states[i].messages_pending =
                                simulators[i].messages_pending;
                            active_states[i].processing = simulators[i].processing
                                || byes.count(simulators[i].name) != 0;
                        }
                        const vector<fncs::ActiveGrant> &grants =
                            fncs::select_active_grants(
                                active_states,
                                active_dependencies,
                                &active_scheduler_scratch);
                        for (size_t g=0; g<grants.size(); ++g) {
                            size_t i = grants[g].index;
                            fncs::time granted = grants[g].time;
                            ++n_processing;
                            simulators[i].processing = true;
                            simulators[i].messages_pending = false;
                            simulators[i].pending_time = ULLONG_MAX;
                            simulators[i].current_grant = granted;
                            ++simulators[i].grant_count;
                            zstr_sendm(server, simulators[i].name.c_str());
                            zstr_sendm(server, fncs::TIME_REQUEST);
                            zstr_sendf(server, "%llu", granted);
                            LDEBUG4 << "active-dependency grant " << granted
                                << " to " << simulators[i].name;
                        }
                    }
                    else {
                    vector<fncs::time> time_actionable(n_sims);
                    for (size_t i=0; i<n_sims; ++i) {
                        if (simulators[i].messages_pending) {
                            time_actionable[i] = 
                                  simulators[i].time_last_processed
                                + simulators[i].time_delta;
                        }
                        else {
                            time_actionable[i] = simulators[i].time_requested;
                        }
                    }
                    time_granted = *min_element(time_actionable.begin(),
                                                time_actionable.end());
                    LDEBUG4 << "time_granted = " << time_granted;
                    if (realtime_interval) {
#ifdef _WIN32
                        cerr << "realtime clock not yet supported on WIN32" << endl;
                        exit(EXIT_FAILURE);
#else
                        LDEBUG4 << "time_real = " << time_real;
                        while (time_granted > time_real) {
                            useconds_t u = (time_granted-time_real)/1000;
                            LDEBUG4 << "usleep(" << u << ")";
                            usleep(u);
                        }
                        LDEBUG4 << "time_real = " << time_real;
#endif
                    }
                    for (size_t i=0; i<n_sims; ++i) {
                        if (time_granted == time_actionable[i]) {
                            LDEBUG4 << "granting " << time_granted
                                << " to " << simulators[i].name;
                            ++n_processing;
                            simulators[i].processing = true;
                            simulators[i].messages_pending = false;
                            simulators[i].pending_time = ULLONG_MAX;
                            simulators[i].current_grant = time_granted;
                            ++simulators[i].grant_count;
                            zstr_sendm(server, simulators[i].name.c_str());
                            zstr_sendm(server, fncs::TIME_REQUEST);
                            zstr_sendf(server, "%llu", time_granted);
                        }
                        else {
                            /* fast forward time last processed */
                            fncs::time jump = (time_granted - simulators[i].time_last_processed) / simulators[i].time_delta;
                            simulators[i].time_last_processed += simulators[i].time_delta * jump;
                        }
                    }
                    }
                }
            }
            else if (fncs::PUBLISH == message_type) {
                string topic = "";
                bool found_one = false;

                LDEBUG4 << "PUBLISH received";

                /* did we receive message from a connected sim? */
                if (name_to_index.count(sender) == 0) {
                    LERROR << "simulator '" << sender << "' not connected";
                    broker_die(simulators, server);
                }
                size_t sender_index = name_to_index[sender];

                /* next frame is topic */
                frame = zmsg_next(msg);
                if (!frame) {
                    LERROR << "PUBLISH message missing topic";
                    broker_die(simulators, server);
                }
                topic = fncs::to_string(frame);

                LDEBUG4 << "PUBLISH received topic " << topic;

                if (topic == "__fncs/active_dependencies") {
                    frame = zmsg_next(msg);
                    if (!frame) {
                        LERROR << "active dependency update missing payload";
                        broker_die(simulators, server);
                    }
                    string value = fncs::to_string(frame);
                    if (sender != "orchestrator" || !parse_dependency_update(
                            value, name_to_index, &active_dependencies,
                            &dependency_epoch)) {
                        LERROR << "invalid active dependency update from " << sender;
                        broker_die(simulators, server);
                    }
                    ++dependency_updates;
                    LDEBUG4 << "installed active dependency epoch " << dependency_epoch;
                    zmsg_destroy(&msg);
                    continue;
                }

                if (do_trace) {
                    /* next frame is value payload */
                    frame = zmsg_next(msg);
                    if (!frame) {
                        LERROR << "PUBLISH message missing value";
                        broker_die(simulators, server);
                    }
                    string value = fncs::to_string(frame);
                    trace << simulators[sender_index].current_grant
                        << "\t" << topic
                        << "\t" << value
                        << endl;
                }

                /* send the message to subscribed sims */
#if 0
                for (size_t i=0; i<n_sims; ++i) {
                    bool found = false;
                    if (simulators[i].subscription_values.count(topic)) {
                        found = true;
                    }
                    if (found) {
                        zmsg_t *msg_copy = zmsg_dup(msg);
                        if (!msg_copy) {
                            LERROR << "failed to copy pub message";
                            broker_die(simulators, server);
                        }
                        /* swap out original sender with new destiation */
                        zframe_reset(zmsg_first(msg_copy),
                                simulators[i].name.c_str(),
                                simulators[i].name.size());
                        /* send it on */
									zmsg_send(&msg_copy, server);
									found_one = true;
									fncs::time message_time =
									    simulators[sender_index].current_grant;
									if (message_time < simulators[i].time_last_processed) {
										LERROR << "causality violation: message from " << sender
										    << " at " << message_time << " targets "
										    << simulators[i].name << " at "
										    << simulators[i].time_last_processed;
										broker_die(simulators, server);
									}
									simulators[i].messages_pending = true;
									simulators[i].pending_time = std::min(
									    simulators[i].pending_time, message_time);
                        LDEBUG4 << "pub to " << simulators[i].name;
                    }
                }
#else
                {
                    TopicMap::iterator iter = topic_to_indexes.find(topic);
                    if (iter != topic_to_indexes.end()) {
                        IndexVec &iv = iter->second;
                        size_t recipients = 0;
                        for (IndexVec::iterator candidate=iv.begin(); candidate!=iv.end(); ++candidate) {
                            if (0 == byes.count(simulators[*candidate].name)) {
                                ++recipients;
                            }
                        }
                        size_t delivered = 0;
                        IndexVec::iterator index;
                        for (index=iv.begin(); index!=iv.end(); index++) {
                            size_t i = *index;
							if (0 == byes.count(simulators[i].name)) {
								/* Transfer the original message to the last recipient.
								 * Most ComposeDES topics have one subscriber, so this
								 * avoids a zmsg_dup for the common directed-dispatch path. */
								++delivered;
								zmsg_t *msg_copy = delivered == recipients ? msg : zmsg_dup(msg);
								if (!msg_copy) {
									LERROR << "failed to copy pub message";
									broker_die(simulators, server);
								}
								/* swap out original sender with new destiation */
								zframe_reset(zmsg_first(msg_copy),
										simulators[i].name.c_str(),
										simulators[i].name.size());
								/* send it on */
                                    zmsg_send(&msg_copy, server);
								if (delivered == recipients) {
									msg = NULL;
								}
                                    found_one = true;
                                    fncs::time message_time =
                                        simulators[sender_index].current_grant;
                                    fncs::time recipient_frontier = simulators[i].processing
                                        ? simulators[i].current_grant
                                        : simulators[i].time_last_processed;
                                    if (message_time < recipient_frontier) {
                                        LERROR << "causality violation: message from " << sender
                                            << " at " << message_time << " targets "
                                            << simulators[i].name << " at " << recipient_frontier;
                                        broker_die(simulators, server);
                                    }
                                    simulators[i].messages_pending = true;
                                    simulators[i].pending_time = std::min(
                                        simulators[i].pending_time, message_time);
                                    LDEBUG4 << "pub to " << simulators[i].name;
							}
                        }
                    }
                }
#endif
                if (!found_one) {
                    LDEBUG4 << "dropping PUBLISH message '" << topic << "'";
                }
            }
            else if (fncs::DIE == message_type) {
                LDEBUG4 << "DIE received";

                /* did we receive message from a connected sim? */
                if (name_to_index.count(sender) == 0) {
                    LERROR << "simulator '" << sender << "' not connected";
                    broker_die(simulators, server);
                }

                broker_die(simulators, server);
            }
            else if (fncs::TIME_DELTA == message_type) {
                size_t index = 0; /* index of sim state */
                fncs::time time_delta;

                LDEBUG4 << "TIME_DELTA received";

                /* did we receive message from a connected sim? */
                if (name_to_index.count(sender) == 0) {
                    LERROR << "simulator '" << sender << "' not connected";
                    broker_die(simulators, server);
                }

                /* index of sim state */
                index = name_to_index[sender];

                /* next frame is time */
                frame = zmsg_next(msg);
                if (!frame) {
                    LERROR << "TIME_DELTA message missing time frame";
                    broker_die(simulators, server);
                }
                /* convert time string */
                {
                    istringstream iss(fncs::to_string(frame));
                    iss >> time_delta;
                }

                /* update sim state */
                simulators[index].time_delta = time_delta;
            }
            else {
                LERROR << "received unknown message type '"
                    << message_type << "'";
                broker_die(simulators, server);
            }

            zmsg_destroy(&msg);
        }
    }

    zsock_destroy(&server);
    zsys_shutdown(); /* without this, Windows will assert */

    if (trace.is_open()) {
        trace.close();
    }

    return 0;
}
