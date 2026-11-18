import {mlpRecords} from './fixtures/mlp-records.js';
export function rankGroups(stage,tp,fsdp,index) {
  const base = stage*tp*fsdp;
  const row = Math.floor(index/tp);
  const column = index%tp;
  return {
    rank:base+index, row, column,
    tp:Array.from({length:tp},(_,t)=>base+row*tp+t),
    fsdp:Array.from({length:fsdp},(_,f)=>base+f*tp+column),
  };
}

// Dependencies describe one unfused MLP and one backward invocation in a layer FSDP unit.
// Each edge names the specific value needed by its consumer.
export function mlpGraph(reshard) {
  const records=mlpRecords(reshard);
  return {nodes:records,edges:records.flatMap(node=>(node.inputs??[]).map(([source,value,role='data'])=>({
    source,target:node.id,value,role,
  })))};
}
