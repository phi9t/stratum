import {meshRealization} from './realization.js';
import {validateGraph} from './semantic.js';

// Accept the existing buildPlanGraph() result. Never recompute its ledger.
// Exported Model Explorer input ids can repeat; the edge occurrence disambiguates
// a logical input port while retaining the upstream input/output identifiers.
export function adaptPlan(built) {
  const {graph:source,ops,resolved}=built;
  const modules=new Map([['root',{id:'root',label:'Plan'}]]);
  const index=new Map();
  const graph={
    schema:'stratum-semantic-graph/v1',id:source.id,title:resolved.m.name,scope:'derived scenario',
    sources:[],modules:[],parameters:[],tensors:[],
    model:{nodes:[],edges:[]},
    realization:{nodes:[],edges:[],meshes:[],groups:[],placements:[]},
    execution:{scope:'forward plan only',instances:[]},observations:[],
  };
  const operations=new Map(ops.map(op=>[op.id,op]));
  for (const node of source.nodes) {
    let parent='root',prefix='';
    for (const segment of node.namespace.split('/').filter(Boolean)) {
      prefix=prefix?prefix+'/'+segment:segment;
      const id='module:'+prefix;
      if (!modules.has(id)) modules.set(id,{id,label:segment,parent});
      parent=id;
    }
    const op=operations.get(node.id);
    const entry={id:node.id,label:node.label,formula:op?.tpMode??op?.kind??'Imported operation',
      kind:op?.kind??'compute',phase:'forward',module:parent,ports:[]};
    if (op?.kind==='collective'||op?.kind==='p2p') {
      entry.communication={kind:op.kind,stage:op.stage,scope:'inherited from pure plan engine'};
      graph.realization.nodes.push(entry);
    } else graph.model.nodes.push(entry);
    index.set(node.id,entry);
    for (const metadata of node.outputsMetadata) {
      const spec=metadata.attrs.find(attr=>attr.key==='shape_per_tp_rank')??metadata.attrs.find(attr=>attr.key==='shape');
      const shape=spec?JSON.parse(spec.value):null;
      graph.tensors.push({id:node.id+'/'+metadata.id,label:metadata.id,shape,
        ...(shape?{}:{unknownReason:'Upstream plan has no output shape.'}),
        dtype:'plan activation precision',layout:op?.tpMode??'declared by plan'});
      entry.ports.push({id:'out:'+metadata.id,tensor:node.id+'/'+metadata.id,direction:'output'});
    }
    if (op?.params!==undefined) graph.parameters.push({
      id:'parameter:'+node.id,owner:node.id,logicalElements:op.params,tpElements:op.rankParams,
      shape:null,unknownReason:'Plan exports element counts, not parameter tensor axes.',
    });
  }
  for (const node of source.nodes) node.incomingEdges.forEach((edge,occurrence)=>{
    const tensor=edge.sourceNodeId+'/'+edge.sourceNodeOutputId;
    const producer=index.get(edge.sourceNodeId);
    if (!producer) throw new Error('Missing plan producer '+edge.sourceNodeId);
    if (!graph.tensors.some(value=>value.id===tensor)) {
      graph.tensors.push({id:tensor,label:edge.sourceNodeOutputId,shape:null,
        unknownReason:'Upstream plan has no shape for this dependency.',dtype:'unspecified',layout:'unspecified'});
      producer.ports.push({id:'out:'+edge.sourceNodeOutputId,tensor,direction:'output'});
    }
    const port='in:'+edge.targetNodeInputId+':'+occurrence;
    index.get(node.id).ports.push({id:port,direction:'input',tensor});
    const dependency={id:edge.sourceNodeId+'/'+edge.sourceNodeOutputId+'->'+node.id+'/'+port,
      source:{node:edge.sourceNodeId,port:'out:'+edge.sourceNodeOutputId},
      target:{node:node.id,port},tensor,role:'data',upstream:{...edge}};
    const communication=producer.communication||index.get(node.id).communication;
    (communication?graph.realization.edges:graph.model.edges).push(dependency);
  });
  graph.modules=[...modules.values()];
  if (resolved.plan&&resolved.stages) {
    for (const stage of resolved.stages) {
      const placement=meshRealization({stage:stage.stage,tp:resolved.plan.tp,fsdp:resolved.plan.fsdp,module:'module:pp'+stage.stage});
      for (const key of ['meshes','groups','placements']) graph.realization[key].push(...placement[key]);
    }
  }
  const defects=validateGraph(graph);
  if (defects.length) throw new Error(defects.join('\n'));
  return graph;
}
