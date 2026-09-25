package comm;

import api.info.JobInfo;
import cloudsim.Cloudlet;
import cloudsim.Host;
import cloudsim.Log;
import cloudsim.core.CloudSim;
import cloudsim.core.SimEvent;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import gpu.GpuCloudlet;
import gpu.power.PowerGpuHost;
import workflow.GpuJob;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;

/** Compute-only FNCS worker. Workflow dependency state is owned by the orchestrator. */
public class WorkerCommEngine extends CommEngine {
    private static class DispatchMetadata {
        int attempt;
        String dispatchEventId;
        String correlationId;
        String planId;
    }

    private static class LocalPlanState {
        String planId;
        String dispatchEventId;
        JSONArray tasks;
        int nextTaskIndex;
        final List<JSONObject> completions = new ArrayList<>();
    }

    private final List<PowerGpuHost> hosts;
    private final List<FncsMessage> pendingMessages = new ArrayList<>();
    private final Map<String, DispatchMetadata> metadata = new HashMap<>();
    private final Map<String, LocalPlanState> localPlans = new HashMap<>();
    private final Set<String> seenDispatchEventIds = new HashSet<>();
    private final Map<String, List<JSONObject>> completionByDispatchEvent = new HashMap<>();
    private int cloudletId = 0;
    private int taskId = 0;
    private int gpuTaskId = 0;
    private boolean resourcesReady = false;
    private boolean endRequested = false;

    public WorkerCommEngine(String name, List<PowerGpuHost> hosts) throws Exception {
        super(name);
        this.hosts = hosts;
    }

    @Override
    protected void processResourceCharacteristics(SimEvent event) {
        super.processResourceCharacteristics(event);
        if (getDatacenterCharacteristicsList().size() == getDatacenterIdsList().size()) {
            resourcesReady = true;
            List<FncsMessage> queued = new ArrayList<>(pendingMessages);
            pendingMessages.clear();
            for (FncsMessage message : queued) {
                processFncsMessage(message);
            }
        }
    }

    @Override
    protected void processOtherEvent(SimEvent event) {
        if (event.getTag() == Api.RECEIVE_EVENT && event.getData() instanceof FncsMessage) {
            FncsMessage message = (FncsMessage) event.getData();
            if (!resourcesReady && "compute/dispatch".equals(message.topic)) {
                pendingMessages.add(message);
            } else {
                processFncsMessage(message);
            }
            return;
        }
        super.processOtherEvent(event);
    }

    private void processFncsMessage(FncsMessage message) {
        JSONObject batch = JSON.parseObject(message.value);
        String schemaVersion = batch.getString("schema_version");
        if (!"2.0".equals(schemaVersion)) {
            throw new IllegalArgumentException("Unsupported FNCS schema version: " + schemaVersion);
        }
        Api.observeWorkerBatch(batch);
        Api.setRunId(batch.getString("run_id"));
        JSONArray events = batch.getJSONArray("events");
        if (events == null) {
            return;
        }
        if ("compute/dispatch".equals(message.topic)) {
            for (int index = 0; index < events.size(); index++) {
                dispatch(events.getJSONObject(index));
            }
        } else if ("control".equals(message.topic)) {
            for (int index = 0; index < events.size(); index++) {
                if ("control.end".equals(events.getJSONObject(index).getString("kind"))) {
                    endRequested = true;
                    Api.requestWorkerEnd();
                }
            }
            finishIfDrained();
        }
    }

    private void dispatch(JSONObject event) {
        if ("compute.plan.dispatch".equals(event.getString("kind"))) {
            dispatchLocalPlan(event);
            return;
        }
        JSONObject payload = event.getJSONObject("payload");
        if (payload == null || payload.getJSONObject("task") == null) {
            throw new IllegalArgumentException("compute.dispatch is missing payload.task");
        }
        JSONObject taskObject = payload.getJSONObject("task");

        String taskName = payload.getString("task_id");
        String dispatchEventId = event.getString("event_id");
        if (dispatchEventId == null || dispatchEventId.isEmpty()) {
            throw new IllegalArgumentException("compute.dispatch is missing event_id");
        }
        if (seenDispatchEventIds.contains(dispatchEventId)) {
            List<JSONObject> completions = completionByDispatchEvent.get(dispatchEventId);
            if (completions != null) {
                for (JSONObject completion : completions) {
                    Api.publishWorkerEvent("compute/completed", completion);
                }
            }
            Log.printLine("Ignoring duplicate compute.dispatch " + dispatchEventId);
            return;
        }
        if (metadata.containsKey(taskName)) {
            throw new IllegalStateException("Task already has an active attempt: " + taskName);
        }
        seenDispatchEventIds.add(dispatchEventId);

        DispatchMetadata dispatchMetadata = new DispatchMetadata();
        dispatchMetadata.attempt = payload.getIntValue("attempt");
        dispatchMetadata.dispatchEventId = dispatchEventId;
        dispatchMetadata.correlationId = event.getString("correlation_id");
        metadata.put(taskName, dispatchMetadata);
        submitTask(taskObject, taskName, payload.getString("target_host"));
    }

    private void dispatchLocalPlan(JSONObject event) {
        JSONObject payload = event.getJSONObject("payload");
        JSONArray tasks = payload == null ? null : payload.getJSONArray("tasks");
        String planId = payload == null ? null : payload.getString("plan_id");
        String dispatchEventId = event.getString("event_id");
        if (planId == null || planId.isEmpty() || tasks == null || tasks.size() < 2) {
            throw new IllegalArgumentException(
                    "compute.plan.dispatch requires plan_id and at least two tasks");
        }
        if (dispatchEventId == null || dispatchEventId.isEmpty()) {
            throw new IllegalArgumentException("compute.plan.dispatch is missing event_id");
        }
        if (seenDispatchEventIds.contains(dispatchEventId)) {
            List<JSONObject> completions = completionByDispatchEvent.get(dispatchEventId);
            if (completions != null) {
                for (JSONObject completion : completions) {
                    Api.publishWorkerEvent("compute/completed", completion);
                }
            }
            Log.printLine("Ignoring duplicate compute.plan.dispatch " + dispatchEventId);
            return;
        }
        if (localPlans.containsKey(planId)) {
            throw new IllegalStateException("Local plan already active: " + planId);
        }
        String targetHost = payload.getString("target_host");
        for (int index = 0; index < tasks.size(); index++) {
            JSONObject task = tasks.getJSONObject(index);
            if (!targetHost.equals(task.getString("target_host"))) {
                throw new IllegalArgumentException("Local plan tasks must share target_host");
            }
            String taskName = task.getString("task_id");
            if (metadata.containsKey(taskName)) {
                throw new IllegalStateException("Task already has an active attempt: " + taskName);
            }
        }
        seenDispatchEventIds.add(dispatchEventId);
        LocalPlanState plan = new LocalPlanState();
        plan.planId = planId;
        plan.dispatchEventId = dispatchEventId;
        plan.tasks = tasks;
        localPlans.put(planId, plan);
        submitNextPlanTask(plan);
        Log.printLine("Worker dispatched local plan " + planId + " with " + tasks.size() + " tasks");
    }

    private void submitNextPlanTask(LocalPlanState plan) {
        JSONObject entry = plan.tasks.getJSONObject(plan.nextTaskIndex++);
        String taskName = entry.getString("task_id");
        DispatchMetadata dispatchMetadata = new DispatchMetadata();
        dispatchMetadata.attempt = entry.getIntValue("attempt");
        dispatchMetadata.dispatchEventId = plan.dispatchEventId;
        dispatchMetadata.correlationId = entry.getString("correlation_id");
        dispatchMetadata.planId = plan.planId;
        metadata.put(taskName, dispatchMetadata);
        submitTask(entry.getJSONObject("task"), taskName, entry.getString("target_host"));
    }

    private void submitTask(JSONObject taskObject, String taskName, String targetHost) {
        JobInfo jobInfo = JSON.parseObject(taskObject.toJSONString(), JobInfo.class);
        jobInfo.children = new ArrayList<>();
        GpuJob job = jobInfo.tran2Job(cloudletId++, taskId, gpuTaskId);
        taskId += job.getTasks().size();
        gpuTaskId += job.getTasks().size();
        job.setName(taskName);
        job.setUserId(getId());

        Host selected = null;
        for (PowerGpuHost host : hosts) {
            if (host.getName().equals(targetHost)) {
                selected = host;
                break;
            }
        }
        if (selected == null) {
            throw new IllegalArgumentException("Unknown target host for " + taskName + ": " + targetHost);
        }
        job.setHost(selected);
        job.setVmId(selected.getId());

        submitJob(job);
        Log.printLine("Worker dispatched task " + taskName + " to " + targetHost);
    }

    @Override
    protected void processCloudletReturn(SimEvent event) {
        GpuCloudlet task = (GpuCloudlet) event.getData();
        cloudletsSubmitted--;
        GpuJob job = (GpuJob) task;
        getCloudletReceivedList().add(job);

        DispatchMetadata dispatchMetadata = metadata.get(job.getName());
        if (dispatchMetadata == null) {
            throw new IllegalStateException("Missing dispatch metadata for " + job.getName());
        }

        String status = "SUCCEEDED";
        if (task.getCloudletStatus() == Cloudlet.FAILED) {
            status = "FAILED";
        } else if (task.getCloudletStatus() == Cloudlet.CANCELED) {
            status = "CANCELED";
        }

        JSONObject payload = new JSONObject(true);
        payload.put("task_id", job.getName());
        payload.put("attempt", dispatchMetadata.attempt);
        payload.put("status", status);
        payload.put("start_time_ns", (long) Math.ceil(task.getExecStartTime() * 1000.0));
        payload.put("finish_time_ns", (long) Math.ceil(CloudSim.clock() * 1000.0));
        payload.put("dispatch_event_id", dispatchMetadata.dispatchEventId);

        JSONObject completed = new JSONObject(true);
        completed.put("event_id", Api.nextWorkerEventId());
        completed.put("kind", "compute.completed");
        completed.put("correlation_id", dispatchMetadata.correlationId);
        completed.put("payload", payload);
        metadata.remove(job.getName());
        if (dispatchMetadata.planId == null) {
            List<JSONObject> completions = new ArrayList<>();
            completions.add(completed);
            completionByDispatchEvent.put(dispatchMetadata.dispatchEventId, completions);
            Api.publishWorkerEvent("compute/completed", completed);
        } else {
            LocalPlanState plan = localPlans.get(dispatchMetadata.planId);
            if (plan == null) {
                throw new IllegalStateException("Missing local plan " + dispatchMetadata.planId);
            }
            plan.completions.add(completed);
            if ("SUCCEEDED".equals(status) && plan.nextTaskIndex < plan.tasks.size()) {
                submitNextPlanTask(plan);
                return;
            }
            if (!"SUCCEEDED".equals(status)) {
                appendCanceledPlanCompletions(plan, payload.getLongValue("finish_time_ns"));
            }
            completionByDispatchEvent.put(
                    dispatchMetadata.dispatchEventId, new ArrayList<>(plan.completions));
            for (JSONObject planCompletion : plan.completions) {
                Api.publishWorkerEvent("compute/completed", planCompletion);
            }
            localPlans.remove(plan.planId);
        }
        finishIfDrained();
    }

    private void appendCanceledPlanCompletions(LocalPlanState plan, long finishTimeNs) {
        while (plan.nextTaskIndex < plan.tasks.size()) {
            JSONObject entry = plan.tasks.getJSONObject(plan.nextTaskIndex++);
            JSONObject payload = new JSONObject(true);
            payload.put("task_id", entry.getString("task_id"));
            payload.put("attempt", entry.getIntValue("attempt"));
            payload.put("status", "CANCELED");
            payload.put("start_time_ns", finishTimeNs);
            payload.put("finish_time_ns", finishTimeNs);
            payload.put("dispatch_event_id", plan.dispatchEventId);

            JSONObject completed = new JSONObject(true);
            completed.put("event_id", Api.nextWorkerEventId());
            completed.put("kind", "compute.completed");
            completed.put("correlation_id", entry.getString("correlation_id"));
            completed.put("payload", payload);
            plan.completions.add(completed);
        }
    }

    private void finishIfDrained() {
        if (endRequested && cloudletsSubmitted <= 0) {
            Log.printLine("Worker received orchestrator end and is drained");
            doComplete();
        }
    }

    @Override
    protected boolean ifFinish() {
        return endRequested && cloudletsSubmitted <= 0;
    }
}
