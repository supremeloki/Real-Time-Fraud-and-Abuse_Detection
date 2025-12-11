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
            if r <= cumRatio:
                return armName
        return None

    def getTrialAssignment(self, subjId: str, evt: Dict[str, Any]) -> Dict[str, Any]:
        assignments = {}
        for tName, tCfg in self.liveTrials.items():
            assignedArm = self._assignArm(tCfg, subjId)
            if assignedArm:
                assignments[tName] = assignedArm
                self.logEntries.append(
                    {
                        "moment": datetime.now().isoformat(),
                        "entityId": subjId,
                        "eventId": evt.get("eventId"),
                        "trialName": tName,
                        "assignedArm": assignedArm,
                        "evtContext": {
                            k: evt.get(k) for k in ["userId", "driverId", "fareAmt"]
                        },
                    }
                )
        return assignments

    def getLog(self) -> List[Dict[str, Any]]:
        return self.logEntries


if __name__ == "__main__":
    import time

    trialSetup = {
        "trials": {
            "algo_v1_v2": {
                "name": "Algo_V1_vs_V2_Detect",
                "startTime": (datetime.now() - timedelta(hours=1)).isoformat(),
                "endTime": (datetime.now() + timedelta(days=7)).isoformat(),
                "totalTrafficRatio": 0.5,
                "arms": {
                    "control_v1": {
                        "description": "Uses Alg V1",
                        "allocationRatio": 0.5,
                    },
                    "test_v2": {"description": "Uses Alg V2", "allocationRatio": 0.5},
                },
            },
            "new_feature_test": {
                "name": "New_UI_Feature",
                "startTime": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "endTime": (datetime.now() + timedelta(days=2)).isoformat(),
                "totalTrafficRatio": 0.2,
                "arms": {
                    "control_old": {"description": "Old UI", "allocationRatio": 0.5},
                    "test_new": {"description": "New UI", "allocationRatio": 0.5},
                },
            },
        }
    }

    orch = TestOrchestrator(trialSetup)

    for i in range(20):
        uId = f"user_{i}"
        evtTime = datetime.now() + timedelta(seconds=i * 10)
        evt = {
            "eventId": f"evt_{i}",
            "eventTimestamp": evtTime.isoformat(),
            "userId": uId,
            "driverId": f"driver_{i%3}",
            "fareAmt": 50000 + i * 1000,
        }
        assignments = orch.getTrialAssignment(uId, evt)
        if assignments:
            print(f"Evt {evt['eventId']} (User {uId}) -> Assigns: {assignments}")
        else:
            print(f"Evt {evt['eventId']} (User {uId}) -> No assign.")
        time.sleep(0.1)

    print(f"\nTotal trial decisions: {len(orch.getLog())}")
