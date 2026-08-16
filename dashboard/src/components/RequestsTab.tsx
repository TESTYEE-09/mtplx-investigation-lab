import { InFlightList } from "./InFlightList";
import { RequestLogTable } from "./RequestLogTable";

export function RequestsTab() {
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12">
        <InFlightList />
      </div>
      <div className="col-span-12">
        <RequestLogTable />
      </div>
    </div>
  );
}
