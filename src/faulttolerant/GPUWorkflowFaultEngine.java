package faulttolerant;

import cloudsim.Cloudlet;
import cloudsim.Log;
import cloudsim.core.CloudSim;
import cloudsim.core.CloudSimTags;
import cloudsim.core.SimEvent;
import comm.Api;
import comm.Packet;
import gpu.GpuCloudlet;
import gpu.GpuHost;
import gpu.GpuTask;
import workflow.GPUWorkflowEngine;
import workflow.GpuJob;
import workflow.Parameters;
import workflow.WorkflowSimTags;
import workflow.taskCluster.BasicClustering;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class GPUWorkflowFaultEngine extends GPUWorkflowEngine {

    public GPUWorkflowFaultEngine(String name) throws Exception {
        super(name);
    }

    @Override
    protected void processOtherEvent(SimEvent event) {
        switch (event.getTag()) {
            default:
                super.processOtherEvent(event);
                break;
        }
    }

    @Override
    protected void submitRepeatJob(GpuJob job)  {
        //Log.printLine(CloudSim.clock() + ": " + job.getName() + "需要进入下一周期, 已执行" + job.getExecTime() + "次， 此时任务是否返回：" + this.jobRepeat.get(job.getName()));
        int datacenterId = host2Datacenter.get(job.getVmId());
        BasicClustering clustering = new BasicClustering();
        List<GpuCloudlet> cloudlets = new ArrayList<>();
        for(GpuCloudlet cl: job.getTasks()) {
            cloudlets.add(cl.getNewCloudlet());
        }
        GpuJob jobRepeat = clustering.addTasks2Job(cloudlets);
        jobRepeat.setCloudletId(job.getCloudletId());
        jobRepeat.setHost(job.getHost());
        jobRepeat.setVmId(job.getVmId());
        jobRepeat.setExecTime(job.getExecTime() + 1);
        jobRepeat.setName(job.getName());
        jobRepeat.setUserId(job.getUserId());
        jobRepeat.setPeriod(job.getPeriod());
        jobRepeat.setDeadline(job.getDeadline());
        jobRepeat.setRam(job.getRam());
        jobRepeat.setNumberOfPes(job.getNumberOfPes());
        for(GpuCloudlet cl: jobRepeat.getTasks()) {
            if(cl.getGpuTask() == null)
                continue;
            cl.getGpuTask().setHost(job.getHost());
        }
        double period = job.getPeriod();
        double delay = job.getPeriod();
        //Log.printLine(jobRepeat.getExecTime());
//        if(this.jobRepeat.get(job.getName()) == 0) {
//            //Log.printLine(CloudSim.clock() + " : " + job.getName() + "超时，在" + job.getHost().getName() + "上");
//            sendNow(datacenterId, WorkflowSimTags.WORKFLOW_CLOUDLET_OUT, job);
//            jobRepeat.setExecTime(job.getExecTime());
//        }
        if(period == 0)
            return;
        if(job.getExecTime() == workflow.Parameters.maxTime - 1 && workflow.Parameters.duration == Double.MAX_VALUE) {
            return;
        }
//        List<GpuJob> jobs = new ArrayList<>();
//        jobs.add(jobRepeat);
//        jobAllocationInterface.setJobs(jobs);
        try {
//            jobAllocationInterface.run();
//            // 这之后应该分为2种情况：任务成功分配或者未能成功分配
//            if(!jobAllocationInterface.getJobs().isEmpty()) {
            sendNow(datacenterId, CloudSimTags.CLOUDLET_SUBMIT, jobRepeat);
            cloudletsSubmitted++;
            getCloudletSubmittedList().add(job);
            send(getId(), delay, WorkflowSimTags.WORKFLOW_CLOUDLET_NEXT_PERIOD, jobRepeat);
            send(getId(), jobRepeat.getDeadline(), WorkflowSimTags.WORKFLOW_CLOUDLET_DEADLINE, jobRepeat);
            this.jobRepeat.put(jobRepeat.getName(), 0);
//            }else {
//
//            }
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void processCloudletReturn(SimEvent ev) {
        GpuCloudlet task = (GpuCloudlet) ev.getData();
        DecimalFormat fomat = new DecimalFormat("##.##");
        Log.printLine(fomat.format(CloudSim.clock()) + " 任务返回： " + task.getName() + "已执行" + ((GpuJob)task).getExecTime() + "次");
        String text = "你好，我是客户端。";
        Packet p = new Packet();
        p.setSrc(((GpuJob)task).getHost().getName());
        p.setDst(((GpuJob)task).getHost().getName());
        p.setTxt(text);
//        byte[] request = text.getBytes();
//        Parameters.socket.connect("tcp://*:8080");
//        Parameters.socket.send(request, 0);
//        System.out.println("发送消息：" + text);
//        // 等待回复
//        byte[] reply = Parameters.socket.recv(0);
//        String response = new String(reply);
//        System.out.println("接收到回复：" + response);
        Api.publish(Api.CommType.SEND_WITHOUT_REPLY, p.toString());
        double period = ((GpuJob)task).getPeriod();
        double delay = ((GpuJob)task).getPeriod();
        send(getId(), delay, WorkflowSimTags.WORKFLOW_CLOUDLET_NEXT_PERIOD, task);
        send(getId(), ((GpuJob)task).getDeadline(), WorkflowSimTags.WORKFLOW_CLOUDLET_DEADLINE, task);
        if(task.getCloudletStatus() == Cloudlet.FAILED) {
            //Log.printLine("执行失败" + ((GpuJob) task).getExecTime());
            getCloudletReceivedList().add(task);
            return;
        }
        cloudletsSubmitted--;
        if(((GpuJob)task).getExecTime() == workflow.Parameters.maxTime - 1 && ((GpuJob)task).getPeriod() != 0) {
            jobNeedRepeat--;
        }
        if(jobRepeat.containsKey(task.getName()) && task.getCloudletStatus() == Cloudlet.SUCCESS) {
            if(jobRepeat.get(task.getName()) == 0) {
                doNext((GpuJob) task);
            }
        }
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
            //task.setExecStartTime(CloudSim.clock() - ((GpuJob) task).getPeriod());
            Log.printLine(CloudSim.clock() + ": 任务" + task.getName() + "执行超时, 执行开始时间： " + ((GpuJob)task).getExecStartTime());
            getCloudletReceivedList().add(task);
        }else {
            //Log.printLine("执行完成");
            getCloudletReceivedList().add(task);
        }
        // 优化：仅处理直接子任务
        deliverReadyChildren((GpuJob) task);
        if(ifFinish()) {
            Log.printLine("仿真结束");
            // 任务全部执行完成，仿真结束
            doComplete();
        }
    }
}
