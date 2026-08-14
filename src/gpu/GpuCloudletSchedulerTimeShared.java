package gpu;

import cloudsim.*;
import cloudsim.core.CloudSim;
import faulttolerant.FaultRecord;
import workflow.GpuJob;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * {@link GpuCloudletSchedulerTimeShared} extends
 * {@link CloudletSchedulerTimeShared} to schedule {@link GpuCloudlet}s.
 * 
 * @author Ahmad Siavashi
 * 
 */
public class GpuCloudletSchedulerTimeShared extends CloudletSchedulerTimeShared implements GpuCloudletScheduler {

	private List<GpuTask> gpuTaskList;
	private int ram;
	public void setRam(int r) {
		ram = r;
	}

	/**
	 * {@link CloudletSchedulerTimeShared} with GpuCloudlet support. Assumes all PEs have same MIPS
	 * capacity.
	 */
	public GpuCloudletSchedulerTimeShared() {
		super();
		setGpuTaskList(new ArrayList<GpuTask>());
	}


	public double updateJobProcessing(double currentTime, List<Double> mipsShare, List<Double> imipsShare, List<Double> mmipsShare) {
		setCurrentMipsShare(mipsShare);
		double timeSpam = currentTime - getPreviousTime();
		Log.printLine("有" + getCloudletExecList().size() + "个任务正在执行队列");
		Log.printLine("有" + getCloudletPausedList().size() + "个任务正在等待队列");
		List<ResCloudlet> toRemoves = new ArrayList<>();
		HashMap<String, Boolean> hasJob = new HashMap<>();
		for (ResCloudlet rcl : getCloudletExecList()) {
			ResGpuCloudlet rgcl = (ResGpuCloudlet) rcl;
			if(!rgcl.ifFromGpu)
				toRemoves.add(rcl);
			if(hasJob.containsKey(((GpuCloudlet)rgcl.getCloudlet()).getName())) {
				toRemoves.add(rcl);
			}
			hasJob.put(((GpuCloudlet)rgcl.getCloudlet()).getName(), true);
		}
		getCloudletExecList().removeAll(toRemoves);
		toRemoves = new ArrayList<>();
		hasJob = new HashMap<>();
		for (ResCloudlet rcl : getCloudletPausedList()) {
			ResGpuCloudlet rgcl = (ResGpuCloudlet) rcl;
			if(hasJob.containsKey(((GpuCloudlet)rgcl.getCloudlet()).getName())) {
				toRemoves.add(rcl);
			}
			hasJob.put(((GpuCloudlet)rgcl.getCloudlet()).getName(), true);
		}
		getCloudletPausedList().removeAll(toRemoves);
		for (ResCloudlet rcl : getCloudletExecList()) {
			if(rcl.getRemainingCloudletLength() == rcl.getCloudletTotalLength()) {
				timeSpam = currentTime - rcl.getExecStartTime();
			}
			ResGpuCloudlet cl = (ResGpuCloudlet)rcl;
			//Log.printLine(((GpuCloudlet)cl.getCloudlet()).getName() + "将被执行");
			double cap = getCapacity(mipsShare);
			if(cl.getGpuTask().calcuType == 1)
				cap = getCapacity(imipsShare);
			if(cl.getGpuTask().calcuType == 2)
				cap = getCapacity(mmipsShare);
			rcl.updateCloudletFinishedSoFar((long) (cap * timeSpam * rcl.getNumberOfPes() * Consts.MILLION));
			Log.printLine(CloudSim.clock() + ": 任务被执行， 计算能力： " + (cap  * rcl.getNumberOfPes()) + "间隔时间： " + timeSpam + " 剩余长度：" + rcl.getRemainingCloudletLength() + " 总长度：" + rcl.getCloudletTotalLength());
		}

		// GPU 任务的 cloudlet 在 paused list 中（由 GpuTaskScheduler 独立调度），
		// exec list 为空时返回 MAX_VALUE，防止 updateCloudletProcessing() 每 1 单位时间
		// 注册一次 VM_DATACENTER_EVENT，造成时间无法跳跃到下一事件。
		if (getCloudletExecList().size() == 0) {
			setPreviousTime(currentTime);
			return Double.MAX_VALUE;
		}

		// check finished cloudlets
		double nextEvent = Double.MAX_VALUE;
		List<ResCloudlet> toRemove = new ArrayList<ResCloudlet>();
		List<ResCloudlet> resCloudlets = getCloudletExecList();
		for (ResCloudlet rcl : resCloudlets) {
			long remainingLength = rcl.getRemainingCloudletLength();
			//Log.printLine("remain: " + rcl.getRemainingCloudletLength());
			if (remainingLength == 0) {// finished: remove from the list
				//Log.printLine("remove");
				toRemove.add(rcl);
				cloudletFinish(rcl);
			}
		}
		getCloudletExecList().removeAll(toRemove);
		double ramInUse = 0.0;
		toRemove = new ArrayList<>();
		for(ResCloudlet cl: getCloudletExecList()) {
			ramInUse += ((GpuCloudlet)cl.getCloudlet()).getRam();
		}
		double ramAvailable = ram - ramInUse;
		for(ResCloudlet rcl: getCloudletWaitingList()) {
			if(rcl.getRam() <= ramAvailable) {
				ramAvailable -= rcl.getRam();
				toRemove.add(rcl);
			}
		}
		getCloudletWaitingList().removeAll(toRemove);
//		if(!toRemove.isEmpty())
//			Log.printLine("lalalaalaa");
		getCloudletExecList().addAll(toRemove);

		// estimate finish time of cloudlets
		for (ResCloudlet rcl : getCloudletExecList()) {
			double estimatedFinishTime = currentTime
					+ (rcl.getRemainingCloudletLength() / (getCapacity(mipsShare) * rcl.getNumberOfPes()));
			if (estimatedFinishTime - currentTime < CloudSim.getMinTimeBetweenEvents()) {
				estimatedFinishTime = currentTime + CloudSim.getMinTimeBetweenEvents();
			}

			if (estimatedFinishTime < nextEvent) {
				nextEvent = estimatedFinishTime;
			}
		}

		setPreviousTime(currentTime);
		//Log.printLine("nextEvent: " + nextEvent + "current: " + currentTime);
		return nextEvent;
	}

	@Override
	public double cloudletSubmitWithoutGPU(Cloudlet cloudlet, double fileTransferTime) {
		ResGpuCloudlet rcl = new ResGpuCloudlet((GpuCloudlet) cloudlet);
		rcl.ifGPU = false;
		rcl.setCloudletStatus(Cloudlet.INEXEC);
		for (int i = 0; i < cloudlet.getNumberOfPes(); i++) {
			rcl.setMachineAndPeId(0, i);
		}
		long addLength = 0;
		for(GpuCloudlet gpuCloudlet: ((GpuJob)cloudlet).getTasks()) {
			if(gpuCloudlet.getGpuTask() != null) {
				addLength += gpuCloudlet.getGpuTask().getTaskTotalLength();
			}
		}
		rcl.addLength(addLength);

		double ramInUse = 0.0;
		for(ResCloudlet cl: getCloudletExecList()) {
			ramInUse += ((GpuJob)cl.getCloudlet()).getRam();
		}
		double ramAvailable = ram - ramInUse;
		if(ramAvailable < ((GpuCloudlet) cloudlet).getRam()) {
			getCloudletWaitingList().add(rcl);
		}
		else {
			//Log.printLine("bbbb");
			getCloudletExecList().add(rcl);
		}


		// use the current capacity to estimate the extra amount of
		// time to file transferring. It must be added to the cloudlet length
		double extraSize = getCapacity(getCurrentMipsShare()) * fileTransferTime;
		long length = (long) (cloudlet.getCloudletLength() + extraSize);
		cloudlet.setCloudletLength(length);


		return (length + addLength) / getCapacity(getCurrentMipsShare());
	}

	public double taskSubmit(GpuTask gt) {
		Log.printLine(gt.getName() + "被提交: " + gt.getRequestedGddramSize() + " resID: " + gt.getResId());
//		for(ResCloudlet cl: getCloudletExecList()) {
//			ResGpuCloudlet rcl = (ResGpuCloudlet) cl;
//			Log.printLine("存在：" + ((GpuCloudlet)rcl.getCloudlet()).getName());
//		}
		GpuCloudlet cc = new GpuCloudlet(gt.getTaskId(), gt.getTaskTotalLength(),gt.getThreadsPerBlock() * gt.getNumberOfBlocks(), 0,gt.getTaskOutputSize(),
				new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull(), gt, false, gt.getRequestedGddramSize());
		cc.setResourceParameter(0, 0,0);
		cc.setName(gt.getName() + "_代理");
		//Log.printLine(cc.getRam() + "被需要");
		ResGpuCloudlet toAdd = new ResGpuCloudlet(gt, cc);
		toAdd.setCloudletStatus(Cloudlet.INEXEC);
		for (ResCloudlet rcl : getCloudletPausedList()) {
			ResGpuCloudlet rgcl = (ResGpuCloudlet) rcl;
			if (gt.isThisResCloudlet(rgcl)) {
				//getCloudletPausedList().remove(rgcl);
				//rgcl.setCloudletStatus(Cloudlet.INEXEC);
				double ramInUse = 0.0;
				for(ResCloudlet cl: getCloudletExecList()) {
					ramInUse += ((GpuCloudlet)cl.getCloudlet()).getRam();
				}
				double ramAvailable = ram - ramInUse;
				//Log.printLine(ramAvailable + "可以用");
				if(ramAvailable < cc.getRam()) {
					//Log.printLine("aaa " + ramAvailable + " < " + ((GpuCloudlet) cloudlet).getRam());
					Log.printLine("内存不足");
					toAdd.type = "RAM";
					getCloudletWaitingList().add(toAdd);
				}
				else {
					//Log.printLine("3 " + ((GpuCloudlet)toAdd.getCloudlet()).getName());
					getCloudletExecList().add(toAdd);
				}
				break;
			}
		}
//		Log.printLine("待执行长度：" + toAdd.getCloudletTotalLength());
//		Log.printLine("计算能力：" + (getCapacity(getCurrentMipsShare()) * toAdd.getNumberOfPes()));
//		Log.printLine("任务需求核心：" + toAdd.getNumberOfPes());
		return toAdd.getCloudletTotalLength() / (getCapacity(getCurrentMipsShare()) * toAdd.getNumberOfPes());
	}

	@Override
	public double cloudletSubmit(Cloudlet cloudlet, double fileTransferTime) {
		ResGpuCloudlet rcl = new ResGpuCloudlet((GpuCloudlet) cloudlet);
		Log.printLine(CloudSim.clock() + ":" + ((GpuCloudlet) cloudlet).getName() + "被提交到CPU的任务调度器上 " + getCloudletPausedList().size());
		Log.printLine("！有" + getCloudletExecList().size() + "个任务正在执行队列");
		Log.printLine("！有" + getCloudletPausedList().size() + "个任务正在等待队列");
//		for(ResCloudlet cl: getCloudletExecList()) {
//			ResGpuCloudlet rcl2 = (ResGpuCloudlet) cl;
//			Log.printLine("!存在：" + ((GpuCloudlet)rcl2.getCloudlet()).getName());
//		}
		rcl.setCloudletStatus(Cloudlet.INEXEC);
		for (int i = 0; i < cloudlet.getNumberOfPes(); i++) {
			rcl.setMachineAndPeId(0, i);
		}
		rcl.setCloudletStatus(GpuCloudlet.PAUSED);
		getCloudletPausedList().add(rcl);
		GpuTask gt = rcl.getRemainGpuTasks().get(0);
		getGpuTaskList().add(gt);
//		Log.printLine("！有" + getCloudletExecList().size() + "个任务正在执行队列");
//		Log.printLine("！有" + getCloudletPausedList().size() + "个任务正在等待队列");
		return -1;
//		double ramInUse = 0.0;
//		for(ResCloudlet cl: getCloudletExecList()) {
//			ramInUse += ((GpuJob)cl.getCloudlet()).getRam();
//		}
//		double ramAvailable = ram - ramInUse;
//		if(ramAvailable < ((GpuCloudlet) cloudlet).getRam()) {
//			//Log.printLine("aaa " + ramAvailable + " < " + ((GpuCloudlet) cloudlet).getRam());
//			getCloudletWaitingList().add(rcl);
//		}
//		else
//			getCloudletExecList().add(rcl);
//
//		// use the current capacity to estimate the extra amount of
//		// time to file transferring. It must be added to the cloudlet length
//		double extraSize = getCapacity(getCurrentMipsShare()) * fileTransferTime;
//		long length = (long) (cloudlet.getCloudletLength() + extraSize);
//		cloudlet.setCloudletLength(length);
//		if(cloudlet.getCloudletLength() == 0) {
//			getCloudletPausedList().add(cloudlet);
//			return -1;
//		}
//		return cloudlet.getCloudletLength() / getCapacity(getCurrentMipsShare());
	}

	@Override
	public void cloudletFinish(ResCloudlet rcl) {
		ResGpuCloudlet rgcl = (ResGpuCloudlet) rcl;
		if(rgcl.ifFromGpu) {
			GpuTask task = rgcl.getGpuTask();
			for (ResCloudlet cl : getCloudletPausedList()) {
				ResGpuCloudlet rgcl2 = (ResGpuCloudlet) cl;
				if (task.isThisResCloudlet(rgcl2)) {
					task.setFinishTime(CloudSim.clock());
					if(!rgcl2.finishGpuTasks(task)) {
						getGpuTaskList().add(rgcl2.getRemainGpuTasks().get(0));
						Log.printLine( task.getName() + " 执行结束，接下来执行： " + rgcl2.getRemainGpuTasks().get(0).getName());
					}else {
						Log.printLine(((GpuCloudlet)rgcl2.getCloudlet()).getName() + "所有内核执行完成");
						super.cloudletFinish(rgcl2);
						getCloudletPausedList().remove(cl);
						return;
					}
				}
			}
			return;
		}
		if (!rgcl.hasGpuTask() || !rgcl.ifGPU) {
			super.cloudletFinish(rcl);
			//Log.printLine(((GpuCloudlet)rcl.getCloudlet()).getName() + "不包含要GPU执行的内核");
		} else {
			List<GpuTask> gt = rgcl.getGpuTasks();
			getGpuTaskList().addAll(gt);
			//Log.printLine(((GpuCloudlet)rcl.getCloudlet()).getName() + "包含" + gt.size() + "个GPU执行的内核");
			try {
				rgcl.setCloudletStatus(GpuCloudlet.PAUSED);
				getCloudletPausedList().add(rgcl);
			} catch (Exception e) {
				e.printStackTrace();
				CloudSim.abruptallyTerminate();
			}
		}
	}

	protected List<GpuTask> getGpuTaskList() {
		return gpuTaskList;
	}

	protected void setGpuTaskList(List<GpuTask> gpuTaskList) {
		this.gpuTaskList = gpuTaskList;
	}

	@Override
	public boolean hasGpuTask() {
		//Log.printLine(getGpuTaskList().size());
		return !getGpuTaskList().isEmpty();
	}

	@Override
	public GpuTask getNextGpuTask() {
		//Log.printLine("getNextGpuTask");
		if (hasGpuTask()) {
			return getGpuTaskList().remove(0);
		}
		return null;
	}

	@Override
	public boolean notifyGpuTaskCompletion(GpuTask gt) {
		//Log.printLine("notifyGpuTaskCompletion: " + gt.getTaskId());
		ResGpuCloudlet toRemove = null;
		for (ResCloudlet rcl : getCloudletPausedList()) {
			ResGpuCloudlet rgcl = (ResGpuCloudlet) rcl;
			if (gt.isThisResCloudlet(rgcl)) {
				if(rgcl.finishGpuTasks(gt)) {

					//Log.printLine("a rescloudlet finish");
					//rgcl.setCloudletStatus(GpuCloudlet.SUCCESS);
					//rgcl.finalizeCloudlet();
					super.cloudletFinish(rgcl);
					toRemove = rgcl;
					Log.printLine(((GpuCloudlet)rgcl.getCloudlet()).getName() + "所有内核执行完成");
				}else {
					Log.printLine(gt.getName() + " 执行结束，接下来执行： " + rgcl.getRemainGpuTasks().get(0).getName());
					getGpuTaskList().add(rgcl.getRemainGpuTasks().get(0));
				}
			}
		}
		if(toRemove == null)
			return false;
		getCloudletPausedList().remove(toRemove);
		return true;
	}


}
