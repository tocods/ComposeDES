package workflow;


import cloudsim.*;
import comm.Packet;
import faulttolerant.FaultRecord;
import gpu.GpuCloudlet;
import gpu.GpuTask;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 为什么要有1个GpuJob类？
 * 因为1个任务可以包含多个内核，而GpuCloudlet只能包含1个内核
 */
public class GpuJob extends GpuCloudlet {

    private int execTime = 0;

    public String type = "";
    private double period;

    private double deadline;
    // 1个任务包含多个内核
    private List<GpuCloudlet> tasks;

    private Host host;

    private FaultRecord record = null;

    public long doneLen = -1;

    public List<Packet> packets;

    public List<String> parents;

    public GpuJob(int gpuCloudletId, long cloudletLength, int pesNumber, long cloudletFileSize, long cloudletOutputSize, UtilizationModel utilizationModelCpu, UtilizationModel utilizationModelRam, UtilizationModel utilizationModelBw, GpuTask gpuTask) {
        super(gpuCloudletId, cloudletLength, pesNumber, cloudletFileSize, cloudletOutputSize, utilizationModelCpu, utilizationModelRam, utilizationModelBw, gpuTask);
        host = null;
        tasks = new ArrayList<>();
        packets = new ArrayList<>();
        parents = new ArrayList<>();
    }

    public GpuJob(int jobId, long jobLength, GpuTask t) {
        super(jobId, jobLength, 1, 0, 0, new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull(), t);
        host = null;
        tasks = new ArrayList<>();
        packets = new ArrayList<>();
        parents = new ArrayList<>();
    }

    @Override
    protected void setGpuTask(GpuTask gpuTask) {
    }

    public void receivePacket() {
        if(!packets.isEmpty())
            packets.remove(packets.size() - 1);
    }

    public boolean ifReceiveAll() {
        return packets.isEmpty();
    }

    @Override
    public boolean isFinished() {
        if (index == -1) {
            return false;
        }

        boolean completed = false;

        // if result is 0 or -ve then this Cloudlet has finished
        final long finish = resList.get(index).finishedSoFar;
        final long result = cloudletLength - finish;
        if (result <= 0.0) {
            completed = true;
        }
        for(GpuCloudlet cl: getTasks()) {
            if(cl.getGpuTask() != null) {
                completed = completed & cl.getGpuTask().isFinished();
            }
        }
        return completed;
    }

    public void setExecTime(int execTime) {
        this.execTime = execTime;
    }

    public int getExecTime() {
        return execTime;
    }

    @Override
    public void reset() {
        super.reset();
        for(GpuCloudlet cl: getTasks()) {
            if(cl.getGpuTask() != null) {
                cl.getGpuTask().reset();
            }
        }
    }

    public Boolean ifHasKernel() {
        for(GpuCloudlet cl: getTasks()) {
            if(cl.getGpuTask() == null)
                continue;
            Log.printLine(cl.getGpuTask().hardware);
            if(Objects.equals(cl.getGpuTask().hardware, "GPU") || Objects.equals(cl.getGpuTask().hardware, "gpu"))
                return true;
        }
        return false;
    }

    public void addGpuTask(GpuCloudlet gpuTask) {
        tasks.add(gpuTask);
    }

    public List<GpuCloudlet> getTasks() {
        return tasks;
    }

    public void setHost(Host host) {
        this.host = host;
    }

    public Host getHost() {
        return this.host;
    }

    public double getPeriod() {
        return period;
    }

    public void setPeriod(double period) {
        this.period = period;
    }

    public void setDeadline(Double deadline) {
        this.deadline = deadline;
    }

    public double getDeadline() {
        return deadline;
    }

    @Override
    public GpuTask getGpuTask() {
        return super.getGpuTask();
    }

    public FaultRecord getRecord() {
        return record;
    }

    public void setRecord(FaultRecord record) {
        this.record = record;
    }
}
