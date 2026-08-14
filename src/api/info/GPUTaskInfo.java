package api.info;

import cloudsim.Log;
import cloudsim.UtilizationModel;
import cloudsim.UtilizationModelFull;
import gpu.GpuTask;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class GPUTaskInfo {
    public long taskInputSize;
    public long taskOutputSize;
    public long requestedGddramSize;

    public List<Kernel> kernels;

    public static class Kernel {

        public Integer blockNum;
        public Integer threadNum;
        public long threadLength;

        public String hardware;

        public Double requestedGddramSize;

        public Double taskInputSize;

        public Double taskOutputSize;

        public Double utilization;

        public Integer calcuType;
    }

    /**
     *
     * @param cloudletId 所属总任务的ID
     * @return
     */
    public List<GpuTask> tran2GpuTask(int cloudletId, int idStart, Double ram) {
        if(kernels.isEmpty())
            return null;
        List<GpuTask> gpuTasks = new ArrayList<>();
        int id = 0;
        for(Kernel kernel: kernels) {
            long taskLength = kernel.threadLength * 1000; // 以GFLOP为基准，而GPU任务在输入文件里单位是TFLOP
            if(Objects.equals(kernel.hardware, "CPU") || Objects.equals(kernel.hardware, "cpu"))
                taskLength = kernel.threadLength;
            UtilizationModel gpuUtilizationModel = new UtilizationModelFull();
            UtilizationModel gddramUtilizationModel = new UtilizationModelFull();
            UtilizationModel gddramBwUtilizationModel = new UtilizationModelFull();
//            if(kernels.size() == 1) {
//                GpuTask gpuTask = new GpuTask(id + idStart, taskLength, kernel.blockNum, taskInputSize, taskOutputSize,
//                        requestedGddramSize, gpuUtilizationModel, gddramUtilizationModel, gddramBwUtilizationModel);
//                gpuTask.setName("Kernel_" + cloudletId + "_" + id);
//                gpuTask.calcuType = kernel.calcuType;
//                gpuTask.hardware = kernel.hardware;
//                gpuTasks.add(gpuTask);
//                break;
//            }
//            if(id == 0) {
//                GpuTask gpuTask = new GpuTask(id + idStart, taskLength, kernel.blockNum, taskInputSize, 0,
//                        requestedGddramSize, gpuUtilizationModel, gddramUtilizationModel, gddramBwUtilizationModel);
//                gpuTask.setName("Kernel_" + cloudletId + "_" + id);
//                gpuTasks.add(gpuTask);
//            } else if(id == kernels.size() - 1) {
//                GpuTask gpuTask = new GpuTask(id + idStart, taskLength, kernel.blockNum, taskInputSize, taskOutputSize,
//                        requestedGddramSize, gpuUtilizationModel, gddramUtilizationModel, gddramBwUtilizationModel);
//                gpuTask.setName("Kernel_" + cloudletId + "_" + id);
//                gpuTasks.add(gpuTask);
//            } else {
//                GpuTask gpuTask = new GpuTask(id + idStart, taskLength, kernel.blockNum, taskInputSize, taskOutputSize,
//                        requestedGddramSize, gpuUtilizationModel, gddramUtilizationModel, gddramBwUtilizationModel);
//                gpuTask.setName("Kernel_" + cloudletId + "_" + id);
//                gpuTasks.add(gpuTask);
//            }

            GpuTask gpuTask = new GpuTask(id + idStart, taskLength, kernel.blockNum, kernel.taskInputSize.longValue(), kernel.taskOutputSize.longValue(),
                    kernel.requestedGddramSize.longValue(), gpuUtilizationModel, gddramUtilizationModel, gddramBwUtilizationModel);
            if(Objects.equals(kernel.hardware, "CPU") || Objects.equals(kernel.hardware, "cpu"))
                gpuTask = new GpuTask(id + idStart, taskLength, (Integer) kernel.blockNum, kernel.taskInputSize.longValue(), kernel.taskOutputSize.longValue(),
                        ram.longValue(), gpuUtilizationModel, gddramUtilizationModel, gddramBwUtilizationModel);
            gpuTask.setName("Kernel_" + cloudletId + "_" + id);
            Log.printLine(gpuTask.getRequestedGddramSize() + "asa");
            gpuTask.setThreadsPerBlock(kernel.threadNum);
            if(kernel.utilization != null)
                gpuTask.setUtilization(kernel.utilization);
            gpuTask.calcuType = kernel.calcuType;
            gpuTask.hardware = kernel.hardware;
            gpuTasks.add(gpuTask);
            id ++;
        }
        return gpuTasks;
    }
}
