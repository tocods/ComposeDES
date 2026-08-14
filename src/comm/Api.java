package comm;

import cloudsim.Log;
import faulttolerant.FaultTolerantTags;
import fncs.JNIfncs;
import cloudsim.core.CloudSim;
import com.alibaba.fastjson.JSONObject;

import java.rmi.UnexpectedException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Api {
    public static List<String> values = new ArrayList<>();
    private static final Map<String, List<JSONObject>> workerValues = new LinkedHashMap<>();
    private static boolean workerMode = false;
    private static String runId = "";
    private static long workerEventSequence = 0;
    private static long batchSequence = 0;
    public static enum CommType {
        SEND_WITHOUT_REPLY,
        SEND_REPLY
    }

    public final static int RECEIVE_EVENT = FaultTolerantTags.FAULT_TAG_LAST + 1;

    private static String[] topics = {"cloudsim/transfer", "net"};
    public static void publish(CommType type, String value) {
        values.add(value);
        //JNIfncs.publish(topics[type.ordinal()], value);
    }

    public static void sendEnd() {
        JNIfncs.publish("cloudsim/end", "end");
    }



    public static void truePublish() {
        if(!values.isEmpty()) {
            StringBuilder out = new StringBuilder();
            out.append(values.get(0));
            values.remove(0);
            for(String s: values)
                out.append("/").append(s);
            JNIfncs.publish(topics[0], out.toString());
            values.clear();
        }
        if(workerMode && !workerValues.isEmpty()) {
            long logicalTimeNs = (long) Math.ceil(CloudSim.clock() * 1000.0);
            for(Map.Entry<String, List<JSONObject>> entry: workerValues.entrySet()) {
                JSONObject batch = new JSONObject(true);
                batch.put("schema_version", "2.0");
                batch.put("run_id", runId);
                batch.put("batch_id", String.format("gpusim-batch-%09d", ++batchSequence));
                batch.put("logical_time_ns", logicalTimeNs);
                batch.put("events", entry.getValue());
                JNIfncs.publish(entry.getKey(), batch.toJSONString());
            }
            workerValues.clear();
        }
    }

    public static void setWorkerMode(boolean enabled) {
        workerMode = enabled;
        values.clear();
        workerValues.clear();
    }

    public static boolean isWorkerMode() {
        return workerMode;
    }

    public static void setRunId(String value) {
        if(runId.isEmpty()) {
            runId = value;
        } else if(!runId.equals(value)) {
            throw new IllegalArgumentException("FNCS run_id changed from " + runId + " to " + value);
        }
    }

    public static String nextWorkerEventId() {
        return String.format("gpusim-event-%09d", ++workerEventSequence);
    }

    public static void publishWorkerEvent(String topic, JSONObject event) {
        workerValues.computeIfAbsent(topic, ignored -> new ArrayList<>()).add(event);
    }


    public static String[] getEvents() {
        return JNIfncs.get_events();
    }

    public static String getValue(String s) {
        return JNIfncs.get_value(s);
    }

    public static String[] getValues(String s) {
        return JNIfncs.get_values(s);
    }

    public static long timeRequest(long next_time) {
        //Log.printLine("时间请求： " + next_time);
        return JNIfncs.time_request(next_time);
    }

    public static void end() {
        JNIfncs.end();
    }

}
