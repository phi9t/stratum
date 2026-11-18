import {useEffect,useMemo,useState} from 'react';
import {Background,BaseEdge,Controls,Handle,MarkerType,Position,ReactFlow} from '@xyflow/react';
import {allNodes,allEdges} from './semantic.js';
import {projectGraph} from './projection.js';
import {layoutGraph} from './layout.js';

const tones={data:'#526775',saved:'#8d580c',weight:'#7c43ae',ordering:'#526775'};

function GraphNode({data}) {
  const {node,onSelect,onExpand,selected}=data;
  return <div className={'mlp-node '+node.kind}>
    {node.ports.map(port=>{
      const geometry=node.portGeometry.find(p=>p.id===node.id+'/'+port.id);
      return <Handle key={port.id} id={port.id} type={port.direction==='input'?'target':'source'}
        position={port.direction==='input'?Position.Left:Position.Right}
        style={{top:geometry?.y??0}}/>;
    })}
    <button style={{height:node.height}} aria-pressed={selected} onClick={node.kind==='module'?onExpand:onSelect}
      aria-label={node.kind==='module'?'Expand '+node.label:node.label}>
      <strong>{node.label}</strong><span>{node.formula}</span>
      {node.kind==='module'&&<span>Expand module</span>}
    </button>
  </div>;
}
function DependencyEdge({id,data,style,markerEnd}) {
  const path=data.sections.map(section=>{
    const points=[section.startPoint,...section.bendPoints??[],section.endPoint];
    return points.map((point,i)=>(i?'L':'M')+point.x+' '+point.y).join(' ');
  }).join(' ');
  return <BaseEdge id={id} path={path} style={style} markerEnd={markerEnd}/>;
}
const nodeTypes={semantic:GraphNode};
const edgeTypes={dependency:DependencyEdge};

export default function GraphExplorer({graph,selected,onSelect,collapsed,onToggle,panel,onPanel}) {
  const [layoutState,setGeometry]=useState(null);
  const [error,setError]=useState('');
  const view=useMemo(()=>projectGraph(graph,collapsed),[graph,collapsed]);
  useEffect(()=>{
    let active=true;
    setGeometry(null);setError('');
    layoutGraph(view).then(result=>{if(active)setGeometry({view,result});})
      .catch(reason=>{if(active)setError(reason.message);});
    return ()=>{active=false;};
  },[view]);
  const geometry=layoutState?.view===view?layoutState.result:null;
  const index=new Map(allNodes(graph).map(node=>[node.id,node]));
  const tensors=new Map(graph.tensors.map(tensor=>[tensor.id,tensor]));
  const op=index.get(selected)??allNodes(graph)[0];
  const incoming=allEdges(graph).filter(edge=>edge.target.node===op.id);
  const instance=graph.execution.instances.find(entry=>entry.node===op.id);
  const visible=view.representative.get(op.id);
  const nodes=(geometry?.nodes??[]).map(node=>({
    id:node.id,type:'semantic',position:node.position,style:{width:node.width,height:node.height},
    data:{node,selected:visible===node.id,onSelect:()=>onSelect(node.id),onExpand:()=>onToggle(node.id)},
  }));
  const edges=(geometry?.edges??[]).map(edge=>({
    id:edge.id,source:edge.source,target:edge.target,sourceHandle:edge.sourceHandle,targetHandle:edge.targetHandle,
    type:'dependency',data:{sections:edge.sections},
    markerEnd:{type:MarkerType.ArrowClosed,color:tones[edge.role]},
    style:{stroke:tones[edge.role],strokeWidth:edge.original.target.node===op.id?2.4:1.2,
      opacity:edge.original.target.node===op.id||edge.original.source.node===op.id?1:0.45,
      strokeDasharray:edge.role==='saved'?'5 4':undefined},
    ariaLabel:tensors.get(edge.tensor).label+' from '+index.get(edge.original.source.node).label+' to '+index.get(edge.original.target.node).label,
  }));
  return <div className="graph-explorer">
    <div className="graph-tabs" role="group" aria-label="Graph views">
      <button aria-pressed={panel==='dependencies'} onClick={()=>onPanel('dependencies')}>Dependencies</button>
      <button aria-pressed={panel==='evidence'} onClick={()=>onPanel('evidence')}>Evidence &amp; execution</button>
    </div>
    <div className="graph-inspector" aria-live="polite">
      <h3>{op.label}</h3><p>{op.formula}</p>
      <p className="small muted">{op.phase} · {op.module}{visible!==op.id?' · selected inside collapsed module':''}</p>
      {incoming.length>0&&<ul className="mlp-inputs">
        {incoming.map(edge=><li key={edge.id}>
          <code>{tensors.get(edge.tensor).label}</code>{edge.role==='saved'?' saved by ':edge.role==='weight'?' supplied by ':' from '}
          <button onClick={()=>onSelect(edge.source.node)}>{index.get(edge.source.node).label}</button>
        </li>)}
      </ul>}
      <details><summary>Tensor ports · {op.ports.length}</summary>
        <div className="message-scroll"><table className="message-table">
          <thead><tr><th>Port</th><th>Tensor identity</th><th>Shape</th><th>Type / layout</th></tr></thead>
          <tbody>{op.ports.map(port=>{
            const tensor=tensors.get(port.tensor);
            return <tr key={port.id}><td>{port.direction}</td><td><code>{tensor.id}</code></td>
              <td>{tensor.shape?'['+tensor.shape.join(', ')+']':'Unknown'}</td>
              <td>{tensor.dtype} · {tensor.layout}{tensor.unknownReason?' · '+tensor.unknownReason:''}</td></tr>;
          })}</tbody>
        </table></div>
      </details>
    </div>
    {panel==='dependencies'?<>
      <div className="module-controls" role="group" aria-label="Collapse modules">
        {graph.modules.map(module=><button key={module.id} aria-pressed={collapsed.has(module.id)}
          onClick={()=>onToggle(module.id)}>{collapsed.has(module.id)?'Expand ':'Collapse '}{module.label}</button>)}
      </div>
      <div className="key small"><span>Tensor dependency</span><span className="saved-key">Dashed amber · saved tensor</span><span className="key-gradient">Purple · parameter availability</span></div>
      {error&&<p role="alert">{error}</p>}
      {!geometry&&!error&&<p role="status">Laying out dependencies…</p>}
      {geometry&&<div className="map-scroll semantic-scroll" tabIndex={0} role="region" aria-label="Scrollable semantic graph">
        <div className="map-flow semantic-flow" style={{width:geometry.width,height:geometry.height}}>
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
            nodesConnectable={false} nodesDraggable={false} panOnDrag={false} zoomOnScroll={false}
            panOnScroll={false} preventScrolling={false} defaultViewport={{x:0,y:0,zoom:1}}>
            <Background gap={24}/><Controls showInteractive={false}/>
          </ReactFlow>
        </div>
      </div>}
      <p className="small muted">{view.edges.length} visible tensor edges · {view.internalEdges.length} internal to collapsed modules</p>
    </>:<div className="graph-evidence">
      <p><strong>{graph.scope}</strong> · {graph.schema}</p>
      <p>Execution scope — {graph.execution.scope}</p>
      {instance&&<p>Selected invocation — <code>{instance.id}</code></p>}
      <p>{graph.observations.length} observed runtime events</p>
      {instance&&graph.realization.groups.filter(group=>group.ranks.includes(instance.rank)).map(group=>
        <p key={group.id}>{group.axis.toUpperCase()} group — {group.ranks.join(', ')}</p>)}
      {graph.scope==='illustrative'&&<p>Teaching topology with authored dimensions. Model checkpoint qualification is pending.</p>}
      {graph.sources.map(source=><p key={source.id}><a href={source.url}>{source.title}</a>
        {source.revision?' @ '+source.revision:''}<br/>{source.scope}</p>)}
      {graph.realization.policy&&<p>FSDP unit — {graph.realization.policy.unit}. Reshard after forward — {String(graph.realization.policy.reshardAfterForward)}.</p>}
    </div>}
  </div>;
}
