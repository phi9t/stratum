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
  const nodes = [];
  const edges = [];
  function add(id,label,formula,kind,column,row,inputs=[]) {
    nodes.push({id,label,formula,kind,column,row});
    for (const [source,value,role='data'] of inputs) edges.push({source,target:id,value,role});
  }
  add('x','Save MLP input','X · [N, D]','saved',0,2);
  add('gather-f','FSDP all-gather · forward','Materialize full TP-local weights for this layer.','fsdp',0,0);
  add('gate','Gate projection','G = X Wgate · [N, K]','gemm',1,1,[['x','X'],['gather-f','Wgate','weight']]);
  add('up','Up projection','U = X Wup · [N, K]','gemm',1,3,[['x','X'],['gather-f','Wup','weight']]);
  add('product','SiLU × up','H = SiLU(G) ⊙ U · [N, K]','compute',2,2,[['gate','G'],['up','U']]);
  add('g','Save gate preactivation','G · [N, K]','saved',2,0,[['gate','G']]);
  add('u','Save up activation','U · [N, K]','saved',2,4,[['up','U']]);
  add('h','Save down input','H · [N, K]','saved',3,3,[['product','H']]);
  add('down','Down projection','P = H Wdown · [N, D], partial','gemm',3,1,[['product','H'],['gather-f','Wdown','weight']]);
  add('y','TP all-reduce · forward','Sum partial P across the TP row to produce replicated Y.','tp',4,1,[['down','P']]);
  add('dy','Incoming gradient','dY · [N, D], already replicated within TP','input',0,7);
  if (reshard) add('gather-b','FSDP all-gather · backward','Restore TP-local weights from persistent shards.','fsdp',0,6);
  const gather = reshard?'gather-b':'gather-f';
  add('dw-down','Down weight gradient','dWdown = Hᵀ dY · [K, D]','gemm',1,5,[['h','H','saved'],['dy','dY']]);
  add('dh','Down input gradient','dH = dY Wdownᵀ · [N, K]','gemm',1,7,[['dy','dY'],[gather,'Wdown','weight']]);
  add('product-b','SiLU × up backward','dG = dH ⊙ U ⊙ SiLU′(G) · dU = dH ⊙ SiLU(G)','compute',2,7,
    [['dh','dH'],['g','G','saved'],['u','U','saved']]);
  add('dw-gate','Gate weight gradient','dWgate = Xᵀ dG · [D, K]','gemm',3,5,[['x','X','saved'],['product-b','dG']]);
  add('dw-up','Up weight gradient','dWup = Xᵀ dU · [D, K]','gemm',3,6,[['x','X','saved'],['product-b','dU']]);
  add('dx-gate','Gate input contribution','dXgate = dG Wgateᵀ · [N, D]','gemm',3,7,[['product-b','dG'],[gather,'Wgate','weight']]);
  add('dx-up','Up input contribution','dXup = dU Wupᵀ · [N, D]','gemm',3,8,[['product-b','dU'],[gather,'Wup','weight']]);
  add('dx-local','Sum local contributions','dXpartial = dXgate + dXup · [N, D]','compute',4,7,[['dx-gate','dXgate'],['dx-up','dXup']]);
  add('dx','TP all-reduce · backward','Sum dXpartial across the TP row to produce dX.','tp',5,7,[['dx-local','dXpartial']]);
  add('other','Other layer gradients','Remaining parameter gradients from this decoder layer.','input',4,4);
  add('rs','FSDP reduce-scatter','Reduce across data rows and return the owned gradient shards. Normalization follows training configuration.','fsdp',5,5,
    [['dw-down','dWdown'],['dw-gate','dWgate'],['dw-up','dWup'],['other','Other gradients']]);
  return {nodes,edges};
}
