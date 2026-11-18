export const allNodes = graph => [...graph.model.nodes,...graph.realization.nodes];
export const allEdges = graph => [...graph.model.edges,...graph.realization.edges];

// Canonical identity is independent of layout, visibility and selection.
export function defineGraph({id,title,scope,records,shapes,modules,parameters=[],sources=[],context={},realization={}}) {
  const graph={
    schema:'stratum-semantic-graph/v1',id,title,scope,sources,modules,parameters,tensors:[],
    model:{nodes:[],edges:[]},
    realization:{nodes:[],edges:[],meshes:[],placements:[],groups:[],...realization},
    execution:{scope:'derived scenario',instances:[]}, observations:[],
  };
  const index=new Map();
  for (const record of records) {
    const {inputs=[],...fields}=record;
    const node={...fields,ports:[]};
    index.set(node.id,node);
    (node.collective?graph.realization.nodes:graph.model.nodes).push(node);
  }
  const tensorIndex=new Map();
  for (const record of records) {
    for (const [occurrence,input] of (record.inputs??[]).entries()) {
      const [source,value,role='data',inputPort]=input;
      const producer=index.get(source);
      if (!producer) throw new Error('Missing producer '+source);
      const tensorId=source+'/'+value;
      if (!tensorIndex.has(tensorId)) {
        const spec=shapes[value];
        if (!spec) throw new Error('Missing tensor contract '+value);
        const tensor={id:tensorId,label:value,...spec};
        tensorIndex.set(tensorId,tensor);
        graph.tensors.push(tensor);
        producer.ports.push({id:'out:'+value,direction:'output',tensor:tensorId});
      }
      const target=index.get(record.id);
      const port=inputPort??'in:'+source+':'+value+':'+occurrence;
      target.ports.push({id:port,direction:'input',tensor:tensorId});
      const edge={id:source+'/'+value+'->'+target.id+'/'+port,source:{node:source,port:'out:'+value},
        target:{node:target.id,port},tensor:tensorId,role};
      (producer.collective||target.collective?graph.realization.edges:graph.model.edges).push(edge);
    }
  }
  // Explicit outputs include graph terminals, even when no downstream consumer exists.
  for (const record of records) for (const value of record.outputs??[]) {
    const tensorId=record.id+'/'+value;
    if (tensorIndex.has(tensorId)) continue;
    if (!shapes[value]) throw new Error('Missing tensor contract '+value);
    graph.tensors.push({id:tensorId,label:value,...shapes[value]});
    index.get(record.id).ports.push({id:'out:'+value,direction:'output',tensor:tensorId});
  }
  const rank=context.rank??0;
  for (const node of allNodes(graph)) {
    const instance={id:[id,context.layer??'fixture',context.step??0,context.microbatch??1,rank,node.id].join('/'),
      node:node.id,phase:node.phase,step:context.step??0,microbatch:context.microbatch??1,rank};
    graph.execution.instances.push(instance);
  }
  const defects=validateGraph(graph);
  if (defects.length) throw new Error(defects.join('\n'));
  return graph;
}

export function validateGraph(graph) {
  const errors=[];
  if (!['derived scenario','illustrative'].includes(graph.scope)) errors.push('Unsupported graph scope');
  if (graph.schema!=='stratum-semantic-graph/v1') errors.push('Unsupported graph schema');
  const nodes=allNodes(graph),edges=allEdges(graph);
  const index=new Map(nodes.map(n=>[n.id,n]));
  const tensors=new Map(graph.tensors.map(t=>[t.id,t]));
  const modules=new Map(graph.modules.map(m=>[m.id,m]));
  if (index.size!==nodes.length || modules.size!==graph.modules.length ||
      tensors.size!==graph.tensors.length || new Set(edges.map(e=>e.id)).size!==edges.length) errors.push('Duplicate identity');
  for (const module of graph.modules) {
    const visited=new Set([module.id]);
    let parent=module.parent;
    while (parent) {
      if (!modules.has(parent)||visited.has(parent)) {errors.push('Invalid module ancestry '+module.id);break;}
      visited.add(parent);
      parent=modules.get(parent).parent;
    }
  }
  for (const tensor of graph.tensors) {
    if (tensor.shape===null) {
      if (!tensor.unknownReason) errors.push('Missing shape reason '+tensor.id);
    } else if (!Array.isArray(tensor.shape)||tensor.shape.some(d=>!Number.isInteger(d)||d<=0)) errors.push('Invalid shape '+tensor.id);
    if (!tensor.dtype||!tensor.layout) errors.push('Missing tensor type '+tensor.id);
  }
  for (const node of nodes) {
    if (['position','row','column'].some(key=>Object.hasOwn(node,key))) errors.push('View geometry in canonical node '+node.id);
    if (modules.has(node.id)) errors.push('Node/module identity collision '+node.id);
    if (!modules.has(node.module)) errors.push('Missing module '+node.id);
    if (!['forward','backward','state'].includes(node.phase)) errors.push('Missing phase '+node.id);
    if (new Set(node.ports.map(p=>p.id)).size!==node.ports.length) errors.push('Duplicate port '+node.id);
    for (const port of node.ports) if (!tensors.has(port.tensor)) errors.push('Missing tensor '+node.id+'/'+port.id);
  }
  const downstream=new Map(nodes.map(n=>[n.id,[]])),degree=new Map(nodes.map(n=>[n.id,0]));
  for (const edge of edges) {
    if (!['data','saved','weight','ordering'].includes(edge.role)) errors.push('Invalid dependency role '+edge.id);
    for (const [endpoint,direction] of [[edge.source,'output'],[edge.target,'input']]) {
      const port=index.get(endpoint.node)?.ports.find(p=>p.id===endpoint.port);
      if (!port||port.direction!==direction||port.tensor!==edge.tensor) errors.push('Invalid endpoint '+edge.id);
    }
    if (!tensors.has(edge.tensor)) errors.push('Missing edge tensor '+edge.id);
    if (downstream.has(edge.source.node)&&degree.has(edge.target.node)) {
      downstream.get(edge.source.node).push(edge.target.node);
      degree.set(edge.target.node,degree.get(edge.target.node)+1);
    }
  }
  const ready=nodes.filter(n=>degree.get(n.id)===0).map(n=>n.id);
  let visited=0;
  while (ready.length) {
    const id=ready.pop(); visited++;
    for (const target of downstream.get(id)) {
      degree.set(target,degree.get(target)-1);
      if (degree.get(target)===0) ready.push(target);
    }
  }
  if (visited!==nodes.length) errors.push('Cyclic tensor dependencies');
  const parameters=new Map(graph.parameters.map(parameter=>[parameter.id,parameter]));
  if (parameters.size!==graph.parameters.length) errors.push('Duplicate parameter identity');
  for (const parameter of graph.parameters) {
    if (parameter.shape===null) {
      if (!parameter.unknownReason) errors.push('Missing parameter shape reason '+parameter.id);
    } else if (!Array.isArray(parameter.shape)||parameter.shape.some(d=>!Number.isInteger(d)||d<=0)) errors.push('Invalid parameter shape '+parameter.id);
  }
  for (const tensor of graph.tensors) if (tensor.parameter) {
    const parameter=parameters.get(tensor.parameter);
    if (!parameter||JSON.stringify(parameter.shape)!==JSON.stringify(tensor.shape)) errors.push('Parameter tensor mismatch '+tensor.id);
  }
  const meshes=new Map(graph.realization.meshes.map(mesh=>[mesh.id,mesh]));
  for (const mesh of graph.realization.meshes) {
    if (mesh.axes.some(axis=>!Number.isInteger(axis.size)||axis.size<1)||new Set(mesh.axes.map(axis=>axis.id)).size!==mesh.axes.length) errors.push('Invalid mesh axes '+mesh.id);
    const size=mesh.axes.reduce((n,axis)=>n*axis.size,1);
    if (new Set(mesh.ranks).size!==mesh.ranks.length||mesh.ranks.length!==size) errors.push('Invalid mesh '+mesh.id);
  }
  const ranks=new Set(graph.realization.meshes.flatMap(mesh=>mesh.ranks));
  for (const group of graph.realization.groups) {
    const mesh=meshes.get(group.mesh);
    const axis=mesh?.axes.find(axis=>axis.id===group.axis);
    if (!axis||group.ranks.length!==axis.size) errors.push('Invalid group axis '+group.id);
    if (new Set(group.ranks).size!==group.ranks.length||group.ranks.some(rank=>!ranks.has(rank))) errors.push('Invalid rank group '+group.id);
  }
  for (const placement of graph.realization.placements) {
    const mesh=meshes.get(placement.mesh);
    if (!modules.has(placement.module)||!mesh||placement.ranks.some(rank=>!mesh.ranks.includes(rank))) errors.push('Invalid placement '+placement.module);
  }
  const instances=graph.execution.instances;
  if (new Set(instances.map(instance=>instance.id)).size!==instances.length) errors.push('Duplicate execution instance');
  for (const instance of instances) {
    if (!index.has(instance.node)||instance.phase!==index.get(instance.node).phase ||
        !Number.isInteger(instance.microbatch)||instance.microbatch<1||
        (ranks.size&&!ranks.has(instance.rank))) errors.push('Invalid execution instance '+instance.id);
  }
  return errors;
}
