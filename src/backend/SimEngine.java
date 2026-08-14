package backend;

import api.Api;
import api.Result;
import api.util.JsonUtil;
import api.util.ParseUtil;
import api.info.*;
import cloudsim.*;
import cloudsim.power.models.PowerModel;
import cloudsim.provisioners.BwProvisionerSimple;
import cloudsim.provisioners.RamProvisionerSimple;
import faulttolerant.faultGenerator.FaultGenerator;
import gpu.*;
import gpu.allocation.VideoCardAllocationPolicy;
import gpu.allocation.VideoCardAllocationPolicyNull;
import gpu.power.PowerGpuHost;
import gpu.power.models.GpuHostPowerModelLinear;

import org.apache.commons.math3.util.Pair;
import workflow.GpuJob;
import workflow.Parameters;
import fncs.JNIfncs;
import org.zeromq.ZMQ;
import org.zeromq.ZContext;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.*;

import static api.service.Parameters.networkRecords;

public class SimEngine {
    private List<PowerGpuHost> hosts;

    private List<GpuCloudlet> jobs;

    private Parameters.JobAllocationAlgorithm algorithm;

    public void setAlgorithm(Parameters.JobAllocationAlgorithm algorithm) {
        this.algorithm = algorithm;
    }

    private boolean mergeEnabled = true;
    public void setMergeEnabled(boolean mergeEnabled) { this.mergeEnabled = mergeEnabled; }
    private int mergeMode = 0;
    public void setMergeMode(int mergeMode) { this.mergeMode = mergeMode; }

    private Api api;

    private ParseUtil parser;

    private JsonUtil jsonParser;

    private int cloudletId;

    private int taskId;

    private int gpuId;

    public SimEngine() {
        api = new Api();
        algorithm = Parameters.JobAllocationAlgorithm.RR;
        parser = new ParseUtil();
        jsonParser = new JsonUtil();
        cloudletId = 0;
        taskId = 0;
        gpuId = 0;
        this.hosts = new ArrayList<>();
        this.jobs = new ArrayList<>();
        resetParamForSim();
    }

    public void resetParamForSim() {
        // 此项为本次仿真持续时间，解析输入文件前设置为无上限
        Parameters.duration = Double.MAX_VALUE;
        // 此项为各个主机和任务的错误生成器，解析输入文件前不存在
        faulttolerant.Parameters.host2FaultInject = new HashMap<>();
        faulttolerant.Parameters.job2FaultInject = new HashMap<>();
        faulttolerant.Parameters.faultRecordList = new ArrayList<>();
        networkRecords = new ArrayList<>();
    }

    public Result parseXmlOfHost(String path) {
        Log.printLine(String.join("", Collections.nCopies(100, "-")));
        Log.printLine("解析主机信息文件 " + path);
        Result ret = null;
        String mess = parser.parseHostXml(new File(path));
        for(HostInfo hostInfo: parser.getHostInfos()) {
            ret = addHost(hostInfo.videoCardInfos, hostInfo.cpuInfos, hostInfo.ram, hostInfo.name);
            if(ret.ifError()) {
                Log.printLine(ret.getMessage());
                hosts = new ArrayList<>();
                return ret;
            }
        }
        return Result.success(mess);
    }

    public Result parseJsonOfHost(String path) {
        Log.printLine(String.join("", Collections.nCopies(100, "-")));
        Log.printLine("解析主机信息文件 " + path);
        Result ret = null;
        List<HostInfo> hostInfos = jsonParser.parseHosts(path);
        for(HostInfo hostInfo: hostInfos) {
            ret = addHost(hostInfo.videoCardInfos, hostInfo.cpuInfos, hostInfo.ram, hostInfo.name);
            if(ret.ifError()) {
                Log.printLine(ret.getMessage());
                hosts = new ArrayList<>();
                return ret;
            }
        }
        return Result.success("");
    }


    public Result parseXmlOfJob(String path) {
        Log.printLine(String.join("", Collections.nCopies(100, "-")));
        Log.printLine("解析任务信息文件 " + path);
        GpuJob j = null;
        String mes = parser.parseJobXml(new File(path));
        for(JobInfo jobInfo: parser.getJobInfos()) {
            j = addJob(jobInfo);
            if(jobInfo.generator != null) {
                GpuJob job = (GpuJob) jobs.get(jobs.size() - 1);
                faulttolerant.Parameters.job2FaultInject.put(job.getName(), jobInfo.generator);
                for(GpuCloudlet cl: job.getTasks()) {
                    faulttolerant.Parameters.job2FaultInject.put(cl.getName(), jobInfo.generator);
                }
            }
        }
        return Result.success(mes);
    }

    public Result parseJsonOfJob(String path) {
        Log.printLine(String.join("", Collections.nCopies(100, "-")));
        Log.printLine("解析任务信息文件 " + path);
        GpuJob j = null;
        List<JobInfo> jobInfos = jsonParser.parseJobs(path);
        Map<String, HashSet<GpuCloudlet>> c2p = new HashMap<>();
        Map<String, JobInfo> name2Info = new HashMap<>();
        for(JobInfo jobInfo: jobInfos) {
            j = addJob(jobInfo);
            name2Info.put(jobInfo.name, jobInfo);
            for(ChildInfo c: jobInfo.children) {
                if(c2p.containsKey(c.child)) {
                    c2p.get(c.child).add(j);
                }else{
                    HashSet<GpuCloudlet> tmp = new HashSet<>();
                    tmp.add(j);
                    c2p.put(c.child, tmp);
                }
            }
            if(jobInfo.generator != null) {
                GpuJob job = (GpuJob) jobs.get(jobs.size() - 1);
                faulttolerant.Parameters.job2FaultInject.put(job.getName(), jobInfo.generator);
                for(GpuCloudlet cl: job.getTasks()) {
                    faulttolerant.Parameters.job2FaultInject.put(cl.getName(), jobInfo.generator);
                }
            }
        }
        for(GpuCloudlet jo: jobs){
            GpuJob job = (GpuJob) jo;
            if(c2p.containsKey(job.getName())) {
                HashSet<GpuCloudlet> ps = c2p.get(job.getName());
                for(GpuCloudlet p: ps)
                    job.addParent(p);
            }
        }

        if (mergeMode == 1) {
            // 新合并：同 rank 整链合并 + 收集跨 rank packet（非阻塞通信模型）
            int merged = mergeLinearChains(c2p, name2Info, true);
            Log.printLine("[mergeMode=1] 同 rank 合并 + 跨 rank 通信: 原始 " + jobInfos.size()
                + " -> 合并后 " + jobs.size() + " (合并了 " + merged + " 个任务)");
        } else if (mergeEnabled) {
            // 旧合并：纯 rank 内链合并，忽略跨 rank
            int merged = mergeLinearChains(c2p, name2Info, false);
            Log.printLine("任务合并完成: 原始 " + jobInfos.size() + " -> 合并后 " + jobs.size()
                + " (合并了 " + merged + " 个任务)");
        } else {
            Log.printLine("任务合并已禁用，任务数: " + jobs.size());
        }

        return Result.success("");
    }

    // 判断两个任务名是否属于同一 rank（rX_ 前缀相同）
    private static boolean sameRank(String a, String b) {
        int u1 = a.indexOf('_'), u2 = b.indexOf('_');
        if (u1 < 0 || u2 < 0) return false;
        return a.substring(0, u1).equals(b.substring(0, u2));
    }

    /**
     * 合并线性依赖链中连续的任务。
     * 每条约18K任务的线性管道合并为单个聚合任务，
     * 将146K任务减少到8个（每条链1个）。
     */
    private int mergeLinearChains(Map<String, HashSet<GpuCloudlet>> c2p,
                                   Map<String, JobInfo> name2Info,
                                   boolean collectCrossRankPackets) {
        // 反向映射: 任务名 -> 该任务所依赖的子任务列表（即谁以此为父任务）
        Map<String, List<String>> parent2Children = new HashMap<>();
        for (Map.Entry<String, HashSet<GpuCloudlet>> e : c2p.entrySet()) {
            String child = e.getKey();
            for (GpuCloudlet parent : e.getValue()) {
                parent2Children.computeIfAbsent(parent.getName(), k -> new ArrayList<>()).add(child);
            }
        }

        // 找根任务（无父任务者）——每条链的起点
        Set<String> allNames = new HashSet<>();
        for (GpuCloudlet jo : jobs)
            allNames.add(jo.getName());

        // 找根任务：忽略跨 rank 父依赖，仅检查同 rank 父依赖
        List<String> roots = new ArrayList<>();
        for (String name : allNames) {
            HashSet<GpuCloudlet> parents = c2p.get(name);
            if (parents == null || parents.isEmpty()) {
                roots.add(name);
            } else {
                boolean hasIntraRankParent = false;
                for (GpuCloudlet p : parents) {
                    if (sameRank(name, p.getName())) {
                        hasIntraRankParent = true;
                        break;
                    }
                }
                if (!hasIntraRankParent)
                    roots.add(name);
            }
        }

        // 对每条链，做 DFS 收集所有任务并合并
        int totalMerged = 0;
        Map<String, GpuJob> name2Job = new HashMap<>();
        for (GpuCloudlet jo : jobs)
            name2Job.put(jo.getName(), (GpuJob) jo);

        List<GpuJob> mergedJobs = new ArrayList<>();
        Set<String> consumed = new HashSet<>();

        // collectCrossRankPackets=true 时不使用延迟根/分段逻辑，直接整链合并
        List<String> allRoots = new ArrayList<>(roots);

        for (int rootIdx = 0; rootIdx < allRoots.size(); rootIdx++) {
            String root = allRoots.get(rootIdx);
            if (consumed.contains(root)) continue;

            // DFS 收集同一 rank 内的连续链
            List<String> chain = new ArrayList<>();
            Set<String> chainVisited = new HashSet<>();
            String current = root;
            while (current != null && !chainVisited.contains(current)) {
                chainVisited.add(current);
                chain.add(current);
                List<String> children = parent2Children.get(current);
                String next = null;
                if (children != null) {
                    for (String child : children) {
                        if (!sameRank(current, child) || chainVisited.contains(child) || consumed.contains(child))
                            continue;
                        HashSet<GpuCloudlet> childParents = c2p.get(child);
                        boolean intraParentOutsideChain = false;
                        if (childParents != null) {
                            for (GpuCloudlet p : childParents) {
                                if (sameRank(child, p.getName()) && !chain.contains(p.getName())) {
                                    intraParentOutsideChain = true;
                                    break;
                                }
                            }
                        }
                        if (collectCrossRankPackets) {
                            // 新模式：不因跨 rank 父依赖断链，整链合并
                            next = child;
                            break;
                        } else {
                            // 旧/分段模式：检查跨 rank 父依赖
                            boolean hasCrossRankParent = false;
                            if (childParents != null) {
                                for (GpuCloudlet p : childParents) {
                                    if (!sameRank(child, p.getName())) {
                                        hasCrossRankParent = true;
                                        break;
                                    }
                                }
                            }
                            if (hasCrossRankParent) continue;
                            if (!intraParentOutsideChain) {
                                next = child;
                                break;
                            }
                        }
                    }
                }
                current = next;
            }

            if (chain.size() <= 1) {
                // 单任务链（已被合并或已经是叶子），保留原样
                for (String name : chain) {
                    if (!consumed.contains(name)) {
                        mergedJobs.add(name2Job.get(name));
                        consumed.add(name);
                    }
                }
                continue;
            }

            // 合并整条链为一个聚合任务
            GpuJob first = name2Job.get(chain.get(0));
            GpuJob last = name2Job.get(chain.get(chain.size() - 1));

            // (1) 聚合 GPU 计算量：累加链上每个任务的 (blockLength * numberOfBlocks)
            //     cpu_task 是屎山代码中的假属性，不予考虑
            long totalGpuBlockLen = 0;
            int  totalGpuBlocks  = 0;
            for (String name : chain) {
                GpuJob j = name2Job.get(name);
                for (GpuCloudlet sub : j.getTasks()) {
                    GpuTask gt = sub.getGpuTask();
                    if (gt != null) {
                        totalGpuBlockLen += (long) gt.getBlockLength() * gt.getNumberOfBlocks();
                        totalGpuBlocks  += gt.getNumberOfBlocks();
                    }
                }
            }
            // 将聚合后的 GPU 工作量写入 first 的第一个子任务，其余子任务清空
            List<GpuCloudlet> firstTasks = first.getTasks();
            if (!firstTasks.isEmpty() && totalGpuBlockLen > 0) {
                GpuTask firstGt = firstTasks.get(0).getGpuTask();
                if (firstGt != null) {
                    firstGt.setBlockLength(totalGpuBlockLen);
                    firstGt.setNumberOfBlocks(1);
                }
            }

            // (3) 聚合跨 rank 的 packets（仅 collectCrossRankPackets=true 时收集全部）
            first.packets.clear();
            if (collectCrossRankPackets) {
                for (String name : chain) {
                    GpuJob j = name2Job.get(name);
                    for (comm.Packet pkt : j.packets) {
                        // 保留跨 rank packet，供 NS3 通信仿真
                        // （同 rank packet 的吸收隐含了同节点通信成本为零）
                        if (!sameRank(pkt.getSrc(), pkt.getDst())) {
                            first.packets.add(pkt);
                        }
                    }
                }
            } else {
                // 旧模式：仅保留最后一个任务的 packet
                first.packets.addAll(last.packets);
            }

            // 更新 parent2Children: 跨 rank 子任务的父引用从链中任务改为 first
            if (collectCrossRankPackets) {
                List<String> crossRankChildren = new ArrayList<>();
                for (String name : chain) {
                    if (name.equals(chain.get(0))) continue; // first 自己不变
                    List<String> children = parent2Children.getOrDefault(name, Collections.emptyList());
                    for (String child : children) {
                        if (!sameRank(name, child)) {
                            crossRankChildren.add(child);
                        }
                    }
                    parent2Children.remove(name);
                }
                if (!crossRankChildren.isEmpty()) {
                    parent2Children.computeIfAbsent(chain.get(0), k -> new ArrayList<>()).addAll(crossRankChildren);
                }
            } else {
                List<String> crossChildren = new ArrayList<>();
                for (int i = 1; i < chain.size(); i++) {
                    String name = chain.get(i);
                    List<String> children = parent2Children.getOrDefault(name, Collections.emptyList());
                    for (String child : children) {
                        if (!sameRank(name, child)) {
                            crossChildren.add(child);
                        }
                    }
                    parent2Children.remove(name);
                }
                if (!crossChildren.isEmpty()) {
                    parent2Children.computeIfAbsent(chain.get(0), k -> new ArrayList<>()).addAll(crossChildren);
                }
            }

            // (4) 依赖关系：first 继承链首的父关系（已在 parseJsonOfJob 中设置），
            //     last 的子任务依赖转移至 first（若存在跨链依赖）

            mergedJobs.add(first);
            consumed.add(chain.get(0));
            totalMerged += chain.size() - 1;

            // 标记链中其余任务为已消费
            for (int i = 1; i < chain.size(); i++) {
                consumed.add(chain.get(i));
            }

            // 更新 c2p: 跨 rank 子任务引用改为 first（跨 rank 依赖用非阻塞模型，
            // 不加到 remainingParentCount 中，仅通过 packet 通信）
            if (collectCrossRankPackets) {
                for (String name : chain) {
                    if (name.equals(chain.get(0))) continue;
                    for (String childName : parent2Children.getOrDefault(name, Collections.emptyList())) {
                        if (!sameRank(name, childName)) {
                            HashSet<GpuCloudlet> parents = c2p.get(childName);
                            if (parents != null) {
                                parents.remove(name2Job.get(name));
                                // 跨 rank 边不加到 c2p，实现非阻塞通信模型
                            }
                        }
                    }
                }
            } else {
                for (String name : chain) {
                    if (name.equals(chain.get(0))) continue;
                    for (String childName : parent2Children.getOrDefault(name, Collections.emptyList())) {
                        if (!sameRank(name, childName)) {
                            HashSet<GpuCloudlet> parents = c2p.get(childName);
                            if (parents != null) {
                                parents.remove(name2Job.get(name));
                                parents.add(first);
                            }
                        }
                    }
                }
            }
        }

        // 添加未被链合并覆盖的任务
        for (String name : allNames) {
            if (!consumed.contains(name)) {
                mergedJobs.add(name2Job.get(name));
            }
        }

        Log.printLine("原始任务数: " + allNames.size() + ", 合并后任务数: " + mergedJobs.size()
            + " (减少 " + (allNames.size() - mergedJobs.size()) + ")");

        jobs = new ArrayList<>(mergedJobs);
        return totalMerged;
    }

    public Result parseJsonOfFault(String path) {
        Log.printLine(String.join("", Collections.nCopies(100, "-")));
        Log.printLine("解析错误注入信息文件 " + path);
        Result ret = null;
        List<FaultInfo> faultInfos = jsonParser.parseFaults(path);
        for(FaultInfo faultInfo: faultInfos) {
            Host h = null;
            for(Host host: hosts)
                if(host.getName().equals(faultInfo.aim))
                    h = host;
            if(h != null) {
                if(faulttolerant.Parameters.host2FaultInject.containsKey(h))
                    faulttolerant.Parameters.host2FaultInject.get(h).add(faultInfo.tran2Generator());
                else {
                    List<FaultGenerator> generators = new ArrayList<>();
                    generators.add(faultInfo.tran2Generator());
                    faulttolerant.Parameters.host2FaultInject.put(h, generators);
                }
            }
        }
        return Result.success("");
    }

    public Result start(String outputPath) {
        api.setHosts(hosts);
        api.setTasks(jobs);
        return api.start(algorithm, outputPath);
    }


    /**
     * 创建1个新的物理节点
     * @param videoCardInfos 显卡信息
     * @param cpuInfos CPU信息
     * @param ram 内存大小
     */
    public Result addHost(List<VideoCardInfo> videoCardInfos, List<CPUInfo> cpuInfos, int ram, String name) {
        // 对GPU建模
        List<VideoCard> videoCards = new ArrayList<>(videoCardInfos.size());
        for(int videoCardId = 0; videoCardId < videoCardInfos.size(); videoCardId ++) {
            videoCards.add(videoCardInfos.get(videoCardId).tran2Entity(videoCardId));
        }
        VideoCardAllocationPolicy videoCardAllocationPolicy = new VideoCardAllocationPolicyNull(videoCards);

        // 对CPU建模
        List<Pe> peList = new ArrayList<>();
        for(CPUInfo cpuInfo: cpuInfos) {
           peList.addAll(cpuInfo.tran2Pes());
        }


        //以下为仿真中无需使用的参数
        long storage = GpuHostTags.DUAL_INTEL_XEON_E5_2620_V3_STORAGE;
        int bw = GpuHostTags.DUAL_INTEL_XEON_E5_2620_V3_BW;
        VmScheduler vmScheduler = new VmSchedulerTimeShared(peList);
        //以上为仿真中无需使用的参数

        // 主机能耗模型
        double hostMaxPower = 200;
        double hostStaticPowerPercent = 0.70;
        PowerModel powerModel = new GpuHostPowerModelLinear(hostMaxPower, hostStaticPowerPercent);


        // 对主机建模
        PowerGpuHost newHost = new PowerGpuHost(hosts.size(), GpuHostTags.DUAL_INTEL_XEON_E5_2620_V3,
                new RamProvisionerSimple(ram * 1024), new BwProvisionerSimple(bw), storage, peList, vmScheduler,
                videoCardAllocationPolicy, powerModel);
        GpuCloudletSchedulerTimeShared cloudletSchedulerTimeShared = new GpuCloudletSchedulerTimeShared();
        cloudletSchedulerTimeShared.setRam(ram * 1024);
        List<Double> mips = new ArrayList<>();
        for(Pe pe : peList)
            mips.add(Double.valueOf((Integer)(pe.getMips())));
        cloudletSchedulerTimeShared.updateJobProcessing(0, mips, mips, mips);
        newHost.setCloudletScheduler(cloudletSchedulerTimeShared);

        newHost.setName(name);
        hosts.add(newHost);
        return Result.success(null);
    }

    /**
     * 创建1个新的任务

     */
    public GpuJob addJob(JobInfo jobInfo) {
        GpuJob job = jobInfo.tran2Job(cloudletId, taskId, gpuId);
        Log.printLine(jobInfo.host);
        if(!jobInfo.host.equals("")) {
            for(Host host: hosts) {
                if(host.getName().equals(jobInfo.host)) {
                    job.setVmId(host.getId());
                    job.setHost(host);
                    Log.printLine("I found it");
                }
            }
        }

        jobs.add(job);
        cloudletId ++;
        taskId += job.getTasks().size();
        gpuId += job.getTasks().size();
        return job;
    }

    public Parameters.JobAllocationAlgorithm getAlgorithm(Integer i) {
        switch (i) {
            case 0:
                return Parameters.JobAllocationAlgorithm.RR;
            case 1:
                return Parameters.JobAllocationAlgorithm.RANDOM;
            default:
                return Parameters.JobAllocationAlgorithm.RR;
        }
    }

    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("Usage: SimEngine <outDir> <hosts.json> <jobs.json> <faults.json> [algorithm] [logToFile] [enableMerge]");
            System.err.println("  outDir      : output directory for result XML files");
            System.err.println("  hosts.json  : host configuration file (JSON)");
            System.err.println("  jobs.json   : job configuration file (JSON)");
            System.err.println("  faults.json : fault injection configuration file (JSON)");
            System.err.println("  algorithm   : job allocation algorithm (-1=RR, 0=RANDOM, 1=MCT, 2=OLS, 3=ACO)");
            System.err.println("  logToFile   : whether to write log to gpusim.log (true/false, default true)");
            System.err.println("  enableMerge : whether to merge linear task chains (true/false, default true)");
            System.err.println("  mergeMode   : 1=new cross-rank merge, other=old merge (default 0)");
            System.err.println("  mode        : legacy or worker (default legacy; worker ignores jobs.json)");
            System.exit(1);
        }
        JNIfncs.initialize();
        SimEngine engine = new SimEngine();
        String outPath = args[0];
        String hostPath = args[1];
        String jobPath = args[2];
        String faultPath = args[3];
        Integer algorithm = -1;
        if (args.length >= 5) {
            algorithm = Integer.parseInt(args[4]);
        }
        boolean logToFile = true;
        if (args.length >= 6) {
            logToFile = Boolean.parseBoolean(args[5]);
        }
        boolean enableMerge = true;
        if (args.length >= 7) {
            enableMerge = Boolean.parseBoolean(args[6]);
        }
        int mergeMode = 0;
        if (args.length >= 8) {
            mergeMode = Integer.parseInt(args[7]);
        }
        boolean workerMode = args.length >= 9 && "worker".equalsIgnoreCase(args[8]);
        comm.Api.setWorkerMode(workerMode);
        // mode=1 强制启用合并（带跨 rank 分段支持）
        if (mergeMode == 1) {
            enableMerge = true;
        }
        engine.setMergeEnabled(enableMerge);
        engine.setMergeMode(mergeMode);

        if (logToFile) {
            try {
                FileOutputStream fileOut = new FileOutputStream(outPath + "/gpusim.log");
                OutputStream dualOut = new OutputStream() {
                    public void write(int b) throws IOException {
                        System.out.write(b);
                        fileOut.write(b);
                    }
                    public void write(byte[] b, int off, int len) throws IOException {
                        System.out.write(b, off, len);
                        fileOut.write(b, off, len);
                    }
                    public void flush() throws IOException {
                        System.out.flush();
                        fileOut.flush();
                    }
                    public void close() throws IOException {
                        fileOut.close();
                    }
                };
                Log.setOutput(new PrintStream(dualOut, true));
            } catch (IOException e) {
                Log.printLine("Warning: could not open gpusim.log, logging to stdout only");
            }
        }

        engine.parseJsonOfHost(hostPath);
        if (!workerMode) {
            engine.parseJsonOfJob(jobPath);
        }
        engine.parseJsonOfFault(faultPath);
        engine.setAlgorithm(engine.getAlgorithm(algorithm));
        engine.start(outPath);
    }
}
