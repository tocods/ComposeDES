package api.service;

import java.util.ArrayList;
import java.util.List;

public class Parameters {
    public static boolean enableGPUShare = false;
    public static Double endTime = 0.0;

    public static List<NetworkRecord> networkRecords = new ArrayList<>();

    public static void recordNetworkStart(String jobName, double start, int packetCount, long totalBytes) {
        NetworkRecord record = new NetworkRecord();
        record.jobName = jobName;
        record.start = start;
        record.end = -1.0;
        record.packetCount = packetCount;
        record.totalBytes = totalBytes;
        networkRecords.add(record);
    }

    public static void recordNetworkEnd(String jobName, double end) {
        for(int i = networkRecords.size() - 1; i >= 0; i--) {
            NetworkRecord record = networkRecords.get(i);
            if(record.jobName.equals(jobName) && record.end < 0.0) {
                record.end = end;
                return;
            }
        }
    }

    public static class NetworkRecord {
        public String jobName = "";
        public double start = 0.0;
        public double end = -1.0;
        public int packetCount = 0;
        public long totalBytes = 0;
    }
}
