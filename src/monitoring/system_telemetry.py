import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque

import numpy as np


class SystemTelemetry:
    def __init__(self, buffSize: int = 1000, winSec: int = 3600):
        self.buffSize = buffSize
        self.winSec = winSec
        self.lats: deque = deque(maxlen=buffSize)
        self.errs: deque = deque(maxlen=buffSize)
        self.thrput: deque = deque(maxlen=buffSize)
        self.lastThrputFlush = datetime.now()
        self.currEvtCount = 0

    def addLatency(self, latencyMs: float):
        self.lats.append((datetime.now(), latencyMs))

    def addError(self, errType: str, details: Optional[Dict[str, Any]] = None):
        self.errs.append((datetime.now(), {"type": errType, "details": details or {}}))

    def incThroughput(self, count: int = 1):
        self.currEvtCount += count

    def _flushThrput(self):
        now = datetime.now()
        if (now - self.lastThrputFlush).total_seconds() >= 1:
            self.thrput.append((now, self.currEvtCount))
            self.currEvtCount = 0
            self.lastThrputFlush = now

    def _trimOld(self):
        cutOffTime = datetime.now() - timedelta(seconds=self.winSec)
        for metricDeque in [self.lats, self.errs, self.thrput]:
            while metricDeque and metricDeque[0][0] < cutOffTime:
                metricDeque.popleft()

    def getAggregated(self) -> Dict[str, Any]:
        self._flushThrput()
        self._trimOld()
        metrics = {}
        latencies = [l[1] for l in self.lats]
        if latencies:
            metrics["latencyAvgMs"] = float(np.mean(latencies))
            metrics["latencyP90Ms"] = float(np.percentile(latencies, 90))
            metrics["latencyMaxMs"] = float(np.max(latencies))
            metrics["latencyCount"] = len(latencies)
        else:
            metrics["latencyAvgMs"] = 0.0
            metrics["latencyP90Ms"] = 0.0
            metrics["latencyMaxMs"] = 0.0
            metrics["latencyCount"] = 0
        errorTypes = [e[1]["type"] for e in self.errs]
        if errorTypes:
            uniqueErrs, counts = np.unique(errorTypes, return_counts=True)
            metrics["errorCounts"] = dict(zip(uniqueErrs, [int(c) for c in counts]))
            metrics["totalErrors"] = len(errorTypes)
        else:
            metrics["errorCounts"] = {}
            metrics["totalErrors"] = 0
        throughputs = [t[1] for t in self.thrput]
        if throughputs:
            metrics["thrputAvgEps"] = float(np.mean(throughputs))
            metrics["thrputMaxEps"] = float(np.max(throughputs))
            metrics["thrputTotalEvents"] = int(np.sum(throughputs))
        else:
            metrics["thrputAvgEps"] = 0.0
            metrics["thrputMaxEps"] = 0.0
            metrics["thrputTotalEvents"] = 0
        metrics["windowEnd"] = datetime.now().isoformat()
        metrics["windowDurationSec"] = self.winSec
        return metrics


if __name__ == "__main__":
    import json
    import random

    telemetry = SystemTelemetry(buffSize=50, winSec=60)

    for i in range(30):
        telemetry.addLatency(random.uniform(10, 100))
        if i % 5 == 0:
            telemetry.addLatency(random.uniform(200, 500))
        if i % 7 == 0:
            telemetry.addError("ModelFail", {"mdlId": "v2"})
        if i % 13 == 0:
            telemetry.addError("DbError")
        telemetry.incThroughput()
        time.sleep(0.1)

    metrics1 = telemetry.getAggregated()
    print(json.dumps(metrics1, indent=2))

    time.sleep(65)

    for i in range(5):
        telemetry.addLatency(random.uniform(50, 150))
        telemetry.incThroughput()
        time.sleep(0.1)

    metrics2 = telemetry.getAggregated()
    print(json.dumps(metrics2, indent=2))
