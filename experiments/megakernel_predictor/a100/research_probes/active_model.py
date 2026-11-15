"""Admission parameter ensembles and an adaptive configuration-score baseline."""

from dataclasses import asdict
import json
from pathlib import Path
from statistics import median

import numpy as np

from admission_model import Admission, PARAMETERS, WorkflowController, COMPUTE_COMPLETED, specs
from joint_cases import downstream

NOISE = .01
TRAIN_TEMPERATURE = .015


def read_runs(path):
    entries, timings = {}, {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = row["name"]
        if row["type"] == "meta":
            assert name not in entries
            entries[name] = row
            timings[name] = {}
        else:
            key = row["cap"], row["rule"]
            repetitions = timings[name].setdefault(key, {})
            assert row["repeat"] not in repetitions
            repetitions[row["repeat"]] = row["ms"]
    return {name: BenchCase(row,timings[name]) for name,row in entries.items()}


class BenchCase:
    def __init__(self, row, trials):
        self.name, self.cohort = row["name"], row["cohort"]
        self.layout, self.sms = row["layout"], row["sms"]
        self.deps = tuple(tuple(x) for x in row["deps"])
        self.kinds = tuple(row["kinds"])
        self.copies = frozenset(i for i,k in enumerate(self.kinds) if k == "io")
        self.labels = [(cap,label) for cap,label,_ in row["candidates"]]
        self.orders = [tuple(order) for _,_,order in row["candidates"]]
        assert set(self.labels) == set(trials)
        self.trials = [trials[c] for c in self.labels]
        assert self.trials and all(set(t) == set(self.trials[0]) for t in self.trials)
        self.work = [tuple(row["solo"][cap]["ms"]) for cap,_ in self.labels]
        self.pressure = [tuple(row["solo"][cap]["peak_cta_fraction"]) for cap,_ in self.labels]
        self.scale = float(median(sum(w) for w in self.work))
        self.fixed = self.labels.index(("8","io")) if ("8","io") in self.labels else 0
        self.features = self.config_features()
        ranks = np.argsort(self.orders, axis=1)/(len(self.deps)-1)
        caps = np.array([0 if c=="dma" else int(c)/64 for c,_ in self.labels])
        dma = np.array([c=="dma" for c,_ in self.labels], dtype=float)
        similarity = np.column_stack([1.2*dma, .8*caps, ranks*.85/np.sqrt(len(self.deps))])
        self.distance = ((similarity[:,None,:]-similarity[None,:,:])**2).sum(axis=2)
        self.config_cov = .04**2*np.exp(-2.5*self.distance)

    def timings(self, parity=None):
        return np.array([median(v for i,v in row.items() if parity is None or i%2 == parity)
                         for row in self.trials])

    def config_features(self):
        result = []
        for (cap,rule), order, work, resource in zip(self.labels,self.orders,self.work,self.pressure):
            total = len(self.deps)
            is_dma = cap=="dma"
            quota = 0 if is_dma else int(cap)/64
            rank = np.argsort(order)/(total-1)
            cpu_io = sum(work[i] for i in self.copies)/self.scale
            compute = sum(work[i] for i in range(total) if i not in self.copies)/self.scale
            critical = downstream(self.deps,work)
            path_time = max(critical)/self.scale
            placement = sum((1-rank[i])*critical[i] for i in range(total))/(sum(critical))
            copy_rank = np.mean([rank[i] for i in self.copies]) if self.copies else 0.
            root_compute = [i for i in range(total) if i not in self.copies and not self.deps[i]]
            first = min((rank[i] for i in root_compute), default=0.)
            peak = max(resource[i] for i in range(total) if i not in self.copies)
            result.append([path_time,cpu_io,compute,is_dma,quota,copy_rank,placement,first,
                           peak*quota,quota*compute,is_dma*compute,is_dma*copy_rank,
                           quota*copy_rank,quota*first,is_dma*(self.layout=="paged"),
                           *(rule == tag for tag in ("io","compute","critical","alternating"))])
        return np.array(result)

    def centered_features(self):
        return self.features - self.features.mean(axis=0)


def replay(case, config, model, concurrent):
    order, work, profile = case.orders[config], case.work[config], case.pressure[config]
    cap, _ = case.labels[config]
    is_dma = cap == "dma"
    copies = case.copies
    wf = WorkflowController(specs(case.deps,order,work), "adaptive-admission")
    tails = downstream(case.deps,work)
    positions = {node:i/(len(order)-1) for i,node in enumerate(order)}
    priority = {node: model.order_weight*(1-positions[node])
                + (1-model.order_weight)*tails[node]/max(tails) for node in order}
    if concurrent:
        # The static resident-CTA fraction already incorporates grid size,
        # registers, SMEM and block dimension. One shared fitted scaling factor
        # absorbs scheduling granularity and intra-stage resource variation.
        scale = (case.sms-model.free_sm)/case.sms*1.4
        claims = {
            node: (0. if is_dma else max(.025,profile[node])) if node in copies else
            min(1., .05 + scale*profile[node])
            for node in order
        }
    else:
        claims = {node: (0. if is_dma else int(cap)/case.sms) if node in copies else
                  1-model.free_sm/case.sms for node in order}
    ready = {int(c.event["payload"]["task_id"]): c for c in wf.initial_commands()}
    alive = {}
    time = 0.
    while ready or alive:
        for node in sorted(ready, key=lambda n: (-priority[n],n)):
            live_io = any(n in copies for n in alive)
            live_compute = any(n not in copies for n in alive)
            used = sum(claims[n] for n in alive)
            if node in copies and is_dma:
                fits = not live_io
            else:
                fits = used+claims[node] <= 1.+1e-12
                if node not in copies and not concurrent:
                    fits = fits and not live_compute
            if fits:
                alive[node] = [work[node],ready.pop(node)]
        if not alive:
            raise RuntimeError("Ready queue has no admissible GPU work")
        io_count = sum(i in copies for i in alive)
        compute_count = len(alive)-io_count
        sm_io = sum(claims[i] for i in alive if i in copies) * case.sms/64
        rates = {}
        for i in alive:
            memory = model.memory_contention
            if i in copies:
                rates[i] = 1/(io_count*(1+memory*compute_count))
            else:
                peer = sum(claims[j] for j in alive if j not in copies and i!=j)
                rates[i] = 1/(1+model.sm_contention*sm_io+memory*io_count+.20*peer)
        elapsed = min(left/rates[i] for i,(left,_) in alive.items())
        time += elapsed
        for i in alive:
            alive[i][0] -= elapsed*rates[i]
        for i in [n for n in alive if alive[n][0]<1e-8]:
            _, command = alive.pop(i)
            completion = dict(event_id=f"done/{i}", kind="compute.completed",
                              correlation_id=command.event["correlation_id"],
                              payload=dict(task_id=str(i),attempt=command.event["payload"]["attempt"],
                                           status="SUCCEEDED"))
            more = wf.handle_event(COMPUTE_COMPLETED,completion)
            ready.update((int(c.event["payload"]["task_id"]),c) for c in more)
    assert wf.done
    return time


def candidates_matrix(case, concurrent):
    return np.array([[replay(case,i,p,concurrent) for i in range(len(case.labels))]
                     for p in PARAMETERS]) / case.scale


def softmax(values):
    weights = np.exp(values-np.max(values))
    return weights/weights.sum()


def fit_models(calibration):
    chain = [case for case in calibration if case.cohort=="chain"]
    aligned = {}
    matrix = {f: {c.name:candidates_matrix(c,f) for c in calibration}
              for f in (False,True)}
    for label, dataset in (("chain",chain),("expanded",calibration)):
        x = np.concatenate([c.centered_features() for c in dataset])
        y = np.concatenate([(c.timings()-c.timings().mean())/c.scale for c in dataset])
        scales = np.maximum(.05,x.std(axis=0))
        xn = x/scales
        coef = np.linalg.solve(xn.T @ xn+1.5*np.eye(xn.shape[1]),xn.T@y)/scales
        aligned[f"regression-{label}"] = {"coefficients":coef.tolist()}
        for concurrent in (False,True):
            loss = []
            for case in dataset:
                t = case.timings()/case.scale
                p = matrix[concurrent][case.name]
                centered_residual = (p-t) - (p-t).mean(axis=1,keepdims=True)
                loss.append(np.mean(np.abs(centered_residual), axis=1))
            losses = np.mean(loss,axis=0)
            aligned[f"{'concurrent' if concurrent else 'original'}-{label}"] = {
                "ensemble":softmax(-losses/TRAIN_TEMPERATURE).tolist(),
                "loss":losses.tolist(),
            }
    return dict(model=aligned, parameter_grid=[asdict(p) for p in PARAMETERS],
                cases=dict(chain=len(chain),expanded=len(calibration)),
                configs=dict(chain=sum(len(c.labels) for c in chain),
                             expanded=sum(len(c.labels) for c in calibration)))


def align(predicted, observed):
    if not observed:
        return predicted
    idx = list(observed)
    values = np.array([observed[i] for i in idx])
    aligned = predicted+np.median(values-predicted[...,idx],axis=-1,keepdims=True)
    return aligned


class Ensemble:
    def __init__(self, predictions, weights):
        self.predictions = predictions
        self.weights = np.array(weights)

    def posterior(self, measured):
        ids = list(measured)
        if len(ids)<2:
            return self.weights
        pred = self.predictions[:,ids]
        real = np.array([measured[i] for i in ids])
        residual = pred-real
        residual -= residual.mean(axis=1,keepdims=True)
        # Student-like robust contrast loss. A candidate-common shift cancels.
        loss = np.log1p((residual/NOISE)**2).sum(axis=1)
        return softmax(np.log(self.weights+1e-300)-loss)

    def scores(self, measured):
        w = self.posterior(measured)
        return w @ align(self.predictions,measured)

    def pick(self, case, measured):
        if not measured:
            return case.fixed
        available = sorted(set(range(len(case.labels)))-set(measured))
        w = self.posterior(measured)
        p = align(self.predictions, measured)
        mean = w @ p
        options = p.copy()
        for j,y in measured.items():
            options[:,j] = y
        winners = options.argmin(axis=1)
        groups = sorted(set(winners))
        variance = np.zeros(len(case.labels))
        for winner in groups:
            mask = winners == winner
            mass = w[mask].sum()
            if mass > 1e-9:
                group_mean = w[mask] @ p[mask] / mass
                variance += mass*(group_mean-mean)**2
        overall = w @ (p-mean)**2
        information = variance/(NOISE**2 + np.maximum(0.,overall-variance))
        improvement = w @ np.maximum(0.,min(measured.values())-p)
        return max(available, key=lambda i: (information[i]+improvement[i]/NOISE,-i))


class ScoreRegression:
    def __init__(self, case, coeff):
        self.prior = case.features @ np.array(coeff)
        self.cov = case.config_cov
        self.n = len(case.labels)

    def posterior(self, measured):
        if not measured:
            return self.prior, np.sqrt(np.diag(self.cov))
        ids = list(measured)
        points = np.array([measured[i] for i in ids])
        base = self.prior + np.mean(points-self.prior[ids])
        if len(ids)==1:
            return base, np.sqrt(np.diag(self.cov))
        # Directly learn differences in candidate scores with an RBF residual.
        # It receives the same endpoint measurements as the parameter ensemble.
        contrasts = np.eye(self.n)[ids[1:]]-np.eye(self.n)[ids[0]]
        y = (points[1:]-points[0])-contrasts @ base
        cross = self.cov @ contrasts.T
        gram = contrasts @ cross + NOISE**2*(np.eye(len(y))+1.)
        gain = np.linalg.solve(gram,cross.T).T
        score = align(base+gain @ y,measured)
        variance = np.maximum(0.,np.diag(self.cov)-np.sum(gain*cross,axis=1))
        return score,np.sqrt(variance)

    def scores(self, measured):
        return self.posterior(measured)[0]

    def pick(self, case, measured):
        if not measured:
            return case.fixed
        score, uncertainty = self.posterior(measured)
        return min(set(range(self.n))-set(measured),key=lambda i:(score[i]-1.5*uncertainty[i],i))
