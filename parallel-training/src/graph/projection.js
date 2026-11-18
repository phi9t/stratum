import {allNodes,allEdges} from './semantic.js';

// Contracted edges keep their semantic id and exact source/target port.
// Multiple tensors between the same visible modules remain independent edges.
export function projectGraph(graph,collapsed=new Set()) {
  const nodes=allNodes(graph);
  const modules=new Map(graph.modules.map(module=>[module.id,module]));
  const representative=new Map();
  for (const node of nodes) {
    let visible=node.id;
    let module=node.module;
    while (module) {
      if (collapsed.has(module)) visible=module;
      module=modules.get(module)?.parent;
    }
    representative.set(node.id,visible);
  }
  const viewNodes=new Map();
  for (const node of nodes) {
    const id=representative.get(node.id);
    if (!viewNodes.has(id)) viewNodes.set(id,id===node.id?{...node,ports:[...node.ports]}:
      {id,label:modules.get(id).label,formula:'Collapsed module',kind:'module',module:id,ports:[]});
  }
  const edges=[],internalEdges=[];
  for (const edge of allEdges(graph)) {
    const source=representative.get(edge.source.node),target=representative.get(edge.target.node);
    if (source===target) {internalEdges.push(edge);continue;}
    const handles={};
    for (const [side,id] of [['source',source],['target',target]]) {
      const endpoint=edge[side];
      const handle=id===endpoint.node?endpoint.port:'boundary:'+endpoint.node+':'+endpoint.port;
      const visible=viewNodes.get(id);
      if (!visible.ports.some(port=>port.id===handle)) visible.ports.push({
        id:handle,direction:side==='source'?'output':'input',tensor:edge.tensor,endpoint});
      handles[side+'Handle']=handle;
    }
    edges.push({id:edge.id,source,target,...handles,role:edge.role,tensor:edge.tensor,original:edge,members:[edge.id]});
  }
  return {nodes:[...viewNodes.values()],edges,internalEdges,representative};
}
