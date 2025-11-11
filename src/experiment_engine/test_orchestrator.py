import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class TestOrchestrator:
    def __init__(self, cfg: Dict[str, Any]):
        self.trials: Dict[str, Dict[str, Any]] = cfg.get("trials", {})
        self.liveTrials: Dict[str, Dict[str, Any]] = {}
        self.logEntries: List[Dict[str, Any]] = []
        self._bootTrials()

    def _bootTrials(self):
        currTime = datetime.now()
        for tName, tCfg in self.trials.items():
            sTimeStr = tCfg.get("startTime")
            eTimeStr = tCfg.get("endTime")
            if sTimeStr and eTimeStr:
                sTime = datetime.fromisoformat(sTimeStr)
                eTime = datetime.fromisoformat(eTimeStr)
                if sTime <= currTime <= eTime:
                    self.liveTrials[tName] = tCfg

    def _assignArm(self, trialConfig: Dict[str, Any], subjId: str) -> Optional[str]:
        arms = trialConfig.get("arms", {})
        totalTraffic = trialConfig.get("totalTrafficRatio", 1.0)
        if random.random() >= totalTraffic:
            return None
        armRatios = {
            aName: aD.get("allocationRatio", 0.0) for aName, aD in arms.items()
        }
        sumRatios = sum(armRatios.values())
        if sumRatios == 0:
            return None
        r = random.random() * sumRatios
        cumRatio = 0
        for armName, ratio in armRatios.items():
            cumRatio += ratio
