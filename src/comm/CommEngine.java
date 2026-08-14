package comm;

import cloudsim.Cloudlet;
import cloudsim.Log;
import cloudsim.core.CloudSim;
import cloudsim.core.SimEvent;
import faulttolerant.FaultRecord;
import faulttolerant.GPUWorkflowFaultEngine;
import gpu.GpuCloudlet;
import workflow.GpuJob;
import workflow.WorkflowSimTags;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CommEngine extends GPUWorkflowFaultEngine {
    private Map<String, GpuJob> waittingJobs;
    public CommEngine(String name) throws Exception {
        super(name);
        waittingJobs = new HashMap<>();
    }

    @Override
    protected void processOtherEvent(SimEvent event){
        switch (event.getTag()) {
            case Api.RECEIVE_EVENT:
                String src_name = event.getData().toString();
                Log.printLine("从网络仿真器传来事件： 任务" + src_name + " 网络传输完成");
                GpuJob waittingJob = waittingJobs.get(src_name);
                if (waittingJob == null) {
                    Log.printLine("WARNING: 收到未知任务的网络回包: " + src_name);
                    break;
                }
                waittingJob.receivePacket();
                if(waittingJob.ifReceiveAll()) {
                    api.service.Parameters.recordNetworkEnd(src_name, CloudSim.clock());
                    waittingJobs.remove(src_name);
                    getCloudletReceivedList().add(waittingJob);
                    deliverReadyChildren(waittingJob);
                    if(ifFinish()) {
                        Log.printLine("仿真结束");
                        Api.sendEnd();
                        doComplete();
                    }
                }
                break;
            default:
                super.processOtherEvent(event);
                break;
        }
    }

    @Override
    protected boolean ifFinish() {
        boolean cloudletEmpty = getCloudletList().isEmpty();
        Log.printLine(cloudletEmpty + " (size=" + getCloudletList().size() + ") / " + cloudletsSubmitted);
        if (!cloudletEmpty && getCloudletList().size() <= 15150 && getCloudletList().size() >= 15146) {
            Log.printLine("=== 剩余未提交任务 (" + getCloudletList().size() + "), childrenMap size=" + childrenMap.size() + " ===");
            int count = 0;
            for (Cloudlet c : getCloudletList()) {
                if (count++ >= 20) break;
                GpuJob j = (GpuJob) c;
                int remaining = remainingParentCount.getOrDefault(j.getName(), -1);
                Log.printLine("  " + j.getName() + " remainingParents=" + remaining
                    + " parentCount=" + j.getParent().size()
                    + " packets=" + j.packets.size());
            }
        }
        return cloudletEmpty && cloudletsSubmitted <= 0 && waittingJobs.isEmpty();
    }

    @Override
    protected void processCloudletReturn(SimEvent ev) {
        GpuCloudlet task = (GpuCloudlet) ev.getData();
        Log.printLine(CloudSim.clock() + ": " + task.getName() + "  返回");
        if(task.getCloudletStatus() == Cloudlet.FAILED) {
            getCloudletReceivedList().add(task);
            return;
        }
        cloudletsSubmitted--;
        if(task.getCloudletStatus() == Cloudlet.CANCELED) {
            DecimalFormat format = new DecimalFormat("###.##");
            FaultRecord record = new FaultRecord();
            record.type = FaultRecord.FaultType.TIME_OVER;
            record.ifFalseAlarm = "False";
            record.fault = task.getName();
            record.time = format.format(CloudSim.clock());
            record.redundancyAfter = -1.0;
            record.redundancyBefore = -1;
            record.failReason = ((GpuJob) task).type;
            faulttolerant.Parameters.faultRecordList.add(record);
        }
        GpuJob job = (GpuJob) task;

        if(!job.packets.isEmpty()) {
            int packetCount = 0;
            long totalBytes = 0;
            for(Packet p: job.packets) {
                Log.printLine(job.getName() + " 发包 " + p.toString());
                Api.publish(Api.CommType.SEND_WITHOUT_REPLY, p.toString());
                packetCount++;
                try {
                    totalBytes += Long.parseLong(p.getTxt());
                } catch (NumberFormatException ignored) {
                    // Keep the network timing record even if packet size is malformed.
                }
            }
            api.service.Parameters.recordNetworkStart(job.getName(), CloudSim.clock(), packetCount, totalBytes);
            // 等待 NS-3 网络仿真完成后回包，不立即解锁子任务
            waittingJobs.put(job.getName(), job);
        } else {
            getCloudletReceivedList().add(job);
            deliverReadyChildren(job);
        }
        if(ifFinish()) {
            Log.printLine("仿真结束");
            Api.sendEnd();
            doComplete();
        }
    }
}
