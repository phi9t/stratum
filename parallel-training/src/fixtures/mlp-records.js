export function mlpRecords(reshard) {
  const nodes = [];
  function add(id,label,formula,kind,inputs=[]) {
    const backward=['dy','gather-b','dw-down','dh','product-b','dw-gate','dw-up','dx-gate','dx-up','dx-local','dx','other','rs'].includes(id);
    const phase=backward?'backward':'forward';
    const collective = kind==='tp'||kind==='fsdp' ? {
      axis:kind, operation:id==='rs'?'reduce-scatter':kind==='tp'?'all-reduce':'all-gather',
      unit:kind==='fsdp'?'decoder-layer':null,
    } : undefined;
    const module=phase==='forward'?'forward':'backward';
    nodes.push({id,label,formula,kind,phase,module,inputs,...(collective?{collective}:{})});
  }
  add('x','Save MLP input','X · [N, D]','saved');
  add('gather-f','FSDP all-gather · forward','Materialize full TP-local weights for this layer.','fsdp');
  add('gate','Gate projection','G = X Wgate · [N, K]','gemm',[['x','X'],['gather-f','Wgate','weight']]);
  add('up','Up projection','U = X Wup · [N, K]','gemm',[['x','X'],['gather-f','Wup','weight']]);
  add('product','SiLU × up','H = SiLU(G) ⊙ U · [N, K]','compute',[['gate','G'],['up','U']]);
  add('g','Save gate preactivation','G · [N, K]','saved',[['gate','G']]);
  add('u','Save up activation','U · [N, K]','saved',[['up','U']]);
  add('h','Save down input','H · [N, K]','saved',[['product','H']]);
  add('down','Down projection','P = H Wdown · [N, D], partial','gemm',[['product','H'],['gather-f','Wdown','weight']]);
  add('y','TP all-reduce · forward','Sum partial P across the TP row to produce replicated Y.','tp',[['down','P']]);
  add('dy','Incoming gradient','dY · [N, D], already replicated within TP','input');
  if (reshard) add('gather-b','FSDP all-gather · backward','Restore TP-local weights from persistent shards.','fsdp');
  const gather = reshard?'gather-b':'gather-f';
  add('dw-down','Down weight gradient','dWdown = Hᵀ dY · [K, D]','gemm',[['h','H','saved'],['dy','dY']]);
  add('dh','Down input gradient','dH = dY Wdownᵀ · [N, K]','gemm',[['dy','dY'],[gather,'Wdown','weight']]);
  add('product-b','SiLU × up backward','dG = dH ⊙ U ⊙ SiLU′(G) · dU = dH ⊙ SiLU(G)','compute',
    [['dh','dH'],['g','G','saved'],['u','U','saved']]);
  add('dw-gate','Gate weight gradient','dWgate = Xᵀ dG · [D, K]','gemm',[['x','X','saved'],['product-b','dG']]);
  add('dw-up','Up weight gradient','dWup = Xᵀ dU · [D, K]','gemm',[['x','X','saved'],['product-b','dU']]);
  add('dx-gate','Gate input contribution','dXgate = dG Wgateᵀ · [N, D]','gemm',[['product-b','dG'],[gather,'Wgate','weight']]);
  add('dx-up','Up input contribution','dXup = dU Wupᵀ · [N, D]','gemm',[['product-b','dU'],[gather,'Wup','weight']]);
  add('dx-local','Sum local contributions','dXpartial = dXgate + dXup · [N, D]','compute',[['dx-gate','dXgate'],['dx-up','dXup']]);
  add('dx','TP all-reduce · backward','Sum dXpartial across the TP row to produce dX.','tp',[['dx-local','dXpartial']]);
  add('other','Other layer gradients','Remaining parameter gradients from this decoder layer.','input');
  add('rs','FSDP reduce-scatter','Reduce across data rows and return the owned gradient shards. Normalization follows training configuration.','fsdp',
    [['dw-down','dWdown'],['dw-gate','dWgate'],['dw-up','dWup'],['other','Other gradients']]);
  return nodes;
}
