package gpu.power;

import cloudsim.*;
import gpu.GpuDatacenter;
import gpu.VideoCard;
import gpu.allocation.VideoCardAllocationPolicy;
import gpu.core.GpuCloudSimTags;
import cloudsim.core.SimEvent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * {@link PowerGpuDatacenter} extends {@link GpuDatacenter} to enable simulation
 * of power-aware data centers.
 *
 * @author Ahmad Siavashi
 *
 */
public class PowerGpuDatacenter extends GpuDatacenter {

	/** host-energy mapping. */
	private Map<PowerGpuHost, Double> hostEnergyMap;
	/** host-cpu energy mapping. */
	private Map<PowerGpuHost, Double> hostCpuEnergyMap;
	/** host-videoCard energy mapping. */
	private Map<PowerGpuHost, Map<PowerVideoCard, Double>> hostVideoCardEnergyMap;

	/** to set aside idle hosts from power calculations. **/
	private boolean powerSavingMode;

	private Map<PowerGpuHost, List<Double>> cpuUtils;

	private Map<PowerGpuHost, List<Double>> ramUtils;

	private Map<PowerGpuHost, Map<Integer, List<Double>>> gpuUtils;


	/**
	 * @see GpuDatacenter#GpuDatacenter(String,
	 *      DatacenterCharacteristics, VmAllocationPolicy, List, double)
	 *      GpuDatacenter(String, DatacenterCharacteristics, VmAllocationPolicy,
	 *      List, double)
	 */
	@SuppressWarnings("unchecked")
	public PowerGpuDatacenter(String name, DatacenterCharacteristics characteristics,
			VmAllocationPolicy vmAllocationPolicy, List<Storage> storageList, double schedulingInterval)
			throws Exception {
		super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval);
		cpuUtils = new HashMap<>();
		ramUtils = new HashMap<>();
		gpuUtils = new HashMap<>();
		setHostEnergyMap(new HashMap<PowerGpuHost, Double>());
		setHostCpuEnergyMap(new HashMap<PowerGpuHost, Double>());
		setHostVideoCardEnergyMap(new HashMap<PowerGpuHost, Map<PowerVideoCard, Double>>());
		for (Host host : getCharacteristics().getHostList()) {
			PowerGpuHost powerGpuHost = (PowerGpuHost) host;
			getHostEnergyMap().put(powerGpuHost, 0.0);
			getHostCpuEnergyMap().put(powerGpuHost, 0.0);
			getHostVideoCardEnergyMap().put(powerGpuHost, new HashMap<PowerVideoCard, Double>());
			VideoCardAllocationPolicy videoCardAllocationPolicy = powerGpuHost.getVideoCardAllocationPolicy();
			int gpuNum = 0;
			if (powerGpuHost.isGpuEquipped()) {
				for (PowerVideoCard videoCard : (List<PowerVideoCard>) videoCardAllocationPolicy.getVideoCards()) {
					getHostVideoCardEnergyMap().get(powerGpuHost).put(videoCard, 0.0);
					gpuNum += videoCard.getPgpuList().size();
				}
			}
			cpuUtils.put(powerGpuHost, new ArrayList<>());
			Log.printLine(powerGpuHost.getName() + " 设置");
			ramUtils.put(powerGpuHost, new ArrayList<>());
			Map<Integer, List<Double>> gpuUtil = new HashMap<>();
			for(int i = 0; i < gpuNum; i++)
				gpuUtil.put(i, new ArrayList<>());
			gpuUtils.put(powerGpuHost, gpuUtil);
		}
		setPowerSavingMode(false);
	}

	@SuppressWarnings("unchecked")
	protected void updatePower(double deltaTime) {
		for (Host host : getHostList()) {
			PowerGpuHost powerGpuHost = (PowerGpuHost) host;
			if (isPowerSavingMode() && powerGpuHost.isIdle()) {
				// Assume unused machines are powered off
				continue;
			}
			cpuUtils.get(powerGpuHost).add(powerGpuHost.getCurrentCpuUtilization());
			ramUtils.get(powerGpuHost).add(powerGpuHost.getCurrentRamUtilization());
			List<Double> gpuUtilsCurrent = powerGpuHost.getCurrentGpuUtilization();
			if(gpuUtilsCurrent.get(0) != Double.MAX_VALUE) {
				for(int i = 0; i < gpuUtilsCurrent.size(); i++) {
					gpuUtils.get(powerGpuHost).get(i).add(gpuUtilsCurrent.get(i));
				}
			}
			double hostCpuPower = powerGpuHost.getCurrentHostCpuPower();
			double hostCpuEnergy = getHostCpuEnergyMap().get(powerGpuHost);
			double hostCpuDeltaEnergy = hostCpuPower * deltaTime;
			hostCpuEnergy += hostCpuDeltaEnergy;
			getHostCpuEnergyMap().put(powerGpuHost, hostCpuEnergy);
			double videoCardsEnergy = 0.0;
			for (Entry<VideoCard, Double> videoCardPowerEntry : powerGpuHost.getCurrentVideoCardsPower().entrySet()) {
				PowerVideoCard videoCard = (PowerVideoCard) videoCardPowerEntry.getKey();
				double videoCardPower = videoCardPowerEntry.getValue();
				double videoCardDeltaEnergy = videoCardPower * deltaTime;
				double videoCardEnergy = getHostVideoCardEnergyMap().get(powerGpuHost).get(videoCard);
				videoCardEnergy += videoCardDeltaEnergy;
				getHostVideoCardEnergyMap().get(powerGpuHost).put(videoCard, videoCardEnergy);
				videoCardsEnergy += videoCardEnergy;
			}
			double hostTotalEnergy = hostCpuEnergy + videoCardsEnergy;
			getHostEnergyMap().put(powerGpuHost, hostTotalEnergy);
		}
	}

	public List<Double> getCpuUtilOfHost(PowerGpuHost h) {
		return cpuUtils.get(h);
	}

	public List<Double> getRamUtilOfHost(PowerGpuHost h) {
		return ramUtils.get(h);
	}

	public Map<Integer, List<Double>> getGpuUtilOfHost(PowerGpuHost h) {
		return gpuUtils.get(h);
	}

	// GPU_VM_DATACENTER_POWER_EVENT 自循环已移除。
	// 利用率/电力数据改为在任务事件中按需记录。
	@Override
	public void startEntity() {
		super.startEntity();
	}

	@Override
	protected void processOtherEvent(SimEvent ev) {
		super.processOtherEvent(ev);
		switch (ev.getTag()) {
		case GpuCloudSimTags.GPU_VM_DATACENTER_POWER_EVENT:
			updatePower(getSchedulingInterval());
			break;
		}
	}

	/**
	 * @return consumed energy so far
	 */
	public double getConsumedEnergy() {
		Double totalEnergy = 0.0;
		for (Double hostEnergy : getHostEnergyMap().values()) {
			totalEnergy += hostEnergy;
		}
		return totalEnergy.doubleValue();
	}

	/**
	 * @return the hostEnergyMap
	 */
	public Map<PowerGpuHost, Double> getHostEnergyMap() {
		return hostEnergyMap;
	}

	/**
	 * @param hostEnergyMap the hostEnergyMap to set
	 */
	protected void setHostEnergyMap(Map<PowerGpuHost, Double> hostEnergyMap) {
		this.hostEnergyMap = hostEnergyMap;
	}

	/**
	 * @return the hostVideoCardEnergyMap
	 */
	public Map<PowerGpuHost, Map<PowerVideoCard, Double>> getHostVideoCardEnergyMap() {
		return hostVideoCardEnergyMap;
	}

	/**
	 * @param hostVideoCardEnergyMap the hostVideoCardEnergyMap to set
	 */
	protected void setHostVideoCardEnergyMap(Map<PowerGpuHost, Map<PowerVideoCard, Double>> hostVideoCardEnergyMap) {
		this.hostVideoCardEnergyMap = hostVideoCardEnergyMap;
	}

	/**
	 * @return the hostCpuEnergyMap
	 */
	public Map<PowerGpuHost, Double> getHostCpuEnergyMap() {
		return hostCpuEnergyMap;
	}

	/**
	 * @param hostCpuEnergyMap the hostCpuEnergyMap to set
	 */
	protected void setHostCpuEnergyMap(Map<PowerGpuHost, Double> hostCpuEnergyMap) {
		this.hostCpuEnergyMap = hostCpuEnergyMap;
	}

	public boolean isPowerSavingMode() {
		return powerSavingMode;
	}

	public void setPowerSavingMode(boolean consolidate) {
		this.powerSavingMode = consolidate;
	}

}
