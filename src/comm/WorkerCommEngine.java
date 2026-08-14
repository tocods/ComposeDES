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

/** Compute-only FNCS worker. Workflow dependency state is owned by the orchestrator. */
public class WorkerCommEngine extends CommEngine {
    private static class DispatchMetadata {
        int attempt;
        String dispatchEventId;
        String correlationId;
    }

    private final List<PowerGpuHost> hosts;
    private final List<FncsMessage> pendingMessages = new ArrayList<>();
    private final Map<String, DispatchMetadata> metadata = new HashMap<>();
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
                }
            }
            finishIfDrained();
        }
    }

    private void dispatch(JSONObject event) {
        JSONObject payload = event.getJSONObject("payload");
        JSONObject taskObject = payload.getJSONObject("task");
        if (payload == null || taskObject == null) {
            throw new IllegalArgumentException("compute.dispatch is missing payload.task");
        }

        String taskName = payload.getString("task_id");
        if (metadata.containsKey(taskName)) {
            Log.printLine("Ignoring duplicate compute.dispatch for " + taskName);
            return;
        }

        JobInfo jobInfo = JSON.parseObject(taskObject.toJSONString(), JobInfo.class);
        jobInfo.children = new ArrayList<>();
        GpuJob job = jobInfo.tran2Job(cloudletId++, taskId, gpuTaskId);
        taskId += job.getTasks().size();
        gpuTaskId += job.getTasks().size();
        job.setName(taskName);
        job.setUserId(getId());

        String targetHost = payload.getString("target_host");
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

        DispatchMetadata dispatchMetadata = new DispatchMetadata();
        dispatchMetadata.attempt = payload.getIntValue("attempt");
        dispatchMetadata.dispatchEventId = event.getString("event_id");
        dispatchMetadata.correlationId = event.getString("correlation_id");
        metadata.put(taskName, dispatchMetadata);

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
        payload.put("finish_time_ns", (long) Math.ceil(CloudSim.clock() * 1000.0));
        payload.put("dispatch_event_id", dispatchMetadata.dispatchEventId);

        JSONObject completed = new JSONObject(true);
        completed.put("event_id", Api.nextWorkerEventId());
        completed.put("kind", "compute.completed");
        completed.put("correlation_id", dispatchMetadata.correlationId);
        completed.put("payload", payload);
        Api.publishWorkerEvent("compute/completed", completed);
        finishIfDrained();
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
