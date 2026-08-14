package api.info;

import cloudsim.Log;
import cloudsim.UtilizationModelFull;
import com.alibaba.fastjson.annotation.JSONField;
import comm.Packet;
import de.vandermeer.asciitable.AsciiTable;
import de.vandermeer.skb.interfaces.transformers.textformat.TextAlignment;
import faulttolerant.faultGenerator.FaultGenerator;
import gpu.GpuCloudlet;
import gpu.GpuTask;
import workflow.GpuJob;
import workflow.taskCluster.BasicClustering;
import workflow.taskCluster.ClusteringInterface;

import java.util.ArrayList;
import java.util.List;

public class JobInfo {

    public double period = 0.0;

    public double deadline;

    public String name;

    public CPUTaskInfo cpuTask;

    public GPUTaskInfo gpuTask;

    public FaultGenerator generator = null;

    public String host;

    public List<ChildInfo> children;

    public GpuJob tran2Job(int id, int taskIdStart, int gpuIdStart) {
        if(gpuTask == null || gpuTask.kernels.isEmpty()) {
            List<GpuCloudlet> tasks = new ArrayList<>();
            GpuCloudlet gpuCloudlet = cpuTask.tran2Cloudlet(taskIdStart, null);
            tasks.add(gpuCloudlet);
            BasicClustering clustering = new BasicClustering();
            GpuJob job = clustering.addTasks2Job(tasks);
            job.setNumberOfPes(gpuCloudlet.getNumberOfPes());
            job.setName(name);
            job.setCloudletId(id);
            job.setPeriod(period);
            return job;
        }
        List<GpuCloudlet> tasks = new ArrayList<>();
        List<GpuTask> kernels = gpuTask.tran2GpuTask(id, gpuIdStart, cpuTask.ram);
        int kernelNum = kernels.size();
        for(int i = 0; i < kernelNum; i++) {
//            if(i == 0) {
//                GpuCloudlet gpuCloudlet = cpuTask.tran2Cloudlet(taskIdStart, kernels.get(i));
//                tasks.add(gpuCloudlet);
//            }else {
//                int taskId = taskIdStart + i;
//                GpuCloudlet gpuCloudlet = new GpuCloudlet(taskId, 0, 0, 0, 0,
//                        new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull(), kernels.get(i), false);
//                gpuCloudlet.setName("Task_" + taskId + "_" + i);
//                gpuCloudlet.setRam(0);
//                tasks.add(gpuCloudlet);
//            }
            int taskId = taskIdStart + i;
            GpuCloudlet gpuCloudlet = new GpuCloudlet(taskId, 0, 0, 0, 0,
                    new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull(), kernels.get(i), false);
            gpuCloudlet.setName("Task_" + taskId + "_" + i);
            gpuCloudlet.setRam(0);
            tasks.add(gpuCloudlet);
        }
        BasicClustering clustering = new BasicClustering();
        GpuJob job = clustering.addTasks2Job(tasks);
        job.setCloudletId(id);
        job.setPeriod(period);
        job.setName(name);
        job.setDeadline(deadline);
        if(!children.isEmpty()) {
            for(ChildInfo p: children) {
                // 只为跨主机通信创建 packet（有 src_host 和 dst_host）
                if (p.src_host != null && !p.src_host.isEmpty()
                        && p.dst_host != null && !p.dst_host.isEmpty()) {
                    Packet packet = new Packet();
                    packet.setTxt(String.valueOf(p.size));
                    packet.setSrc(p.src_host);
                    packet.setSrc_task_name(name);
                    packet.setDst(p.dst_host);
                    job.packets.add(packet);
                }
            }
        }
        return job;
    }

    public String print() {
        AsciiTable at = new AsciiTable();
        at.addRule();
        at.addRow("name", "period", "memory", "");
        at.addRule();
        at.addRow(name, period, cpuTask.ram, "");
        at.addRule();
        if(generator != null) {
            at.addRow("faultGenerator", "scale", "shape", "");
            at.addRule();
            at.addRow(generator.getType(), generator.getScale(), generator.getShape(), "");
            at.addRule();
        }
        at.addRow("Device", "cores", "mi", "");
        at.addRule();
        at.addRow("CPU", cpuTask.pesNumber, cpuTask.length, "");
        at.addRule();
        if(gpuTask != null && !gpuTask.kernels.isEmpty()) {
            int i = 0;
            at.addRow("Kernel", "blockNum", "threadPerBlock", "FLOP");
            at.addRule();
            for(GPUTaskInfo.Kernel kernel: gpuTask.kernels) {
                at.addRow(i, kernel.blockNum, kernel.blockNum, kernel.threadLength);
                at.addRule();
                i++;
            }
        }
        at.setTextAlignment(TextAlignment.CENTER);
        Log.printLine(at.render());
        return at.render();
    }
}
