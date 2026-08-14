package workflow;

import gpu.core.GpuCloudSimTags;

public class WorkflowSimTags {

    public final static int WOKFLOW_DATACENTER_EVENT = GpuCloudSimTags.GPU_TAG_LAST + 1;

    public final static int WORKFLOW_CLOUDLET_FINISH = GpuCloudSimTags.GPU_TAG_LAST + 2;

    public final static int WORKFLOW_CLOUDLET_NEXT_PERIOD = GpuCloudSimTags.GPU_TAG_LAST + 3;

    public final static int WORKFLOW_CLOUDLET_END_SIM = GpuCloudSimTags.GPU_TAG_LAST + 4;

    public final static int WORKFLOW_CLOUDLET_OUT = GpuCloudSimTags.GPU_TAG_LAST + 5;

    public final static int WORKFLOW_CLOUDLET_DEADLINE = GpuCloudSimTags.GPU_TAG_LAST + 6;

    public final static int WORKFLOW_KEEP_ALIVE = GpuCloudSimTags.GPU_TAG_LAST + 7;

    public final static int WORKFLOW_TAG_LAST = GpuCloudSimTags.GPU_TAG_LAST + 7;

}
