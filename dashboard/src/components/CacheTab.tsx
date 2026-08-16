import {
  ContextUtilizationBar,
  CumulativeCachedTokensTile,
  HitRateGauge,
} from "./CacheTiles";
import { EvictionReasonHistogram } from "./EvictionReasonHistogram";
import { PrefillTPSSparkline } from "./PrefillTPSSparkline";
import { SessionBankGrid } from "./SessionBankGrid";
import { TTFTDistribution } from "./TTFTDistribution";

export function CacheTab() {
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12">
        <SessionBankGrid />
      </div>

      <div className="col-span-12 lg:col-span-4">
        <CumulativeCachedTokensTile />
      </div>
      <div className="col-span-12 lg:col-span-4">
        <HitRateGauge />
      </div>
      <div className="col-span-12 lg:col-span-4">
        <ContextUtilizationBar />
      </div>

      <div className="col-span-12 lg:col-span-7">
        <EvictionReasonHistogram />
      </div>
      <div className="col-span-12 lg:col-span-5">
        <PrefillTPSSparkline />
      </div>
      <div className="col-span-12">
        <TTFTDistribution />
      </div>
    </div>
  );
}
