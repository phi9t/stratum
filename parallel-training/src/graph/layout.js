import ELK from 'elkjs/lib/elk.bundled.js';
const elk=new ELK();

// ELK is the shared layout engine already used by the plan explorer.
// Coordinates and route sections are a disposable projection of canonical ports.
export async function layoutGraph(view) {
  const children=view.nodes.map(node=>{
    const inputs=node.ports.filter(port=>port.direction==='input');
    const outputs=node.ports.filter(port=>port.direction==='output');
    const height=Math.max(100,32+Math.max(inputs.length,outputs.length)*18);
    const ports=node.ports.map(port=>({
      id:node.id+'/'+port.id,width:1,height:1,
      properties:{'port.side':port.direction==='input'?'WEST':'EAST'},
    }));
    return {id:node.id,width:230,height,ports,
      layoutOptions:{'elk.portConstraints':'FIXED_ORDER'}};
  });
  const edges=view.edges.map(edge=>({id:edge.id,
    sources:[edge.source+'/'+edge.sourceHandle],targets:[edge.target+'/'+edge.targetHandle]}));
  const result=await elk.layout({
    id:'projection',children,edges,
    layoutOptions:{'elk.algorithm':'layered','elk.direction':'RIGHT',
      'elk.spacing.nodeNode':'32','elk.layered.spacing.nodeNodeBetweenLayers':'80',
      'elk.edgeRouting':'ORTHOGONAL','elk.layered.nodePlacement.strategy':'BRANDES_KOEPF',
      'elk.padding':'[top=24,left=24,bottom=24,right=24]'},
  });
  const geometry=new Map(result.children.map(node=>[node.id,node]));
  const routes=new Map(result.edges.map(edge=>[edge.id,edge.sections??[]]));
  return {
    width:result.width,height:result.height,
    nodes:view.nodes.map(node=>({...node,position:{x:geometry.get(node.id).x,y:geometry.get(node.id).y},
      width:geometry.get(node.id).width,height:geometry.get(node.id).height,
      portGeometry:geometry.get(node.id).ports})),
    edges:view.edges.map(edge=>({...edge,sections:routes.get(edge.id)})),
  };
}
