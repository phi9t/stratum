import {useMemo, useState} from 'react';
import {Background, Controls, Handle, MarkerType, Position, ReactFlow} from '@xyflow/react';
import {mlpGraph,rankGroups} from './mlp.js';
import shape from './mlp-shape.json';

function Operation({data}) {
  return <div className={'mlp-node '+data.kind}>
    <Handle type="target" position={Position.Left}/>
    <button aria-pressed={data.selected} onClick={data.select}>
      <strong>{data.label}</strong><span>{data.formula}</span>
    </button>
    <Handle type="source" position={Position.Right}/>
  </div>;
}
const nodeTypes = {operation:Operation};
const edgeTone = {data:'#526775',saved:'#8d580c',weight:'#7c43ae'};

export default function MlpDrilldown({plan, stage, chunk, chunks}) {
  const [layer,setLayer] = useState(chunk.first);
  const [reshard,setReshard] = useState(true);
  const [operation,setOperation] = useState('dw-down');
  const [rank,setRank] = useState(0);
  const [axis,setAxis] = useState('both');
  const graph = useMemo(()=>mlpGraph(reshard),[reshard]);
  const op = graph.nodes.find(node=>node.id===operation) ?? graph.nodes[0];
  const groups = rankGroups(stage,plan.tp,plan.fsdp,rank);
  const incoming = graph.edges.filter(edge=>edge.target===op.id);
  const index = new Map(graph.nodes.map(node=>[node.id,node]));
  function select(node) {
    setOperation(node.id);
    if (node.kind==='tp'||node.kind==='fsdp') setAxis(node.kind);
  }
  const nodes = graph.nodes.map(node=>({
    id:node.id, type:'operation',
    position:{x:18+node.column*214,y:16+node.row*82+(node.row>=5?24:0)},
    style:{width:182},
    data:{...node,selected:node.id===op.id,select:()=>select(node)},
  }));
  const edges = graph.edges.map(edge=>{
    const active = edge.target===op.id || edge.source===op.id;
    const color = edgeTone[edge.role];
    return {
      ...edge, id:edge.source+'-'+edge.target+'-'+edge.value,
      type:'smoothstep', pathOptions:{offset:12}, zIndex:active?1:0,
      style:{stroke:color,strokeWidth:active?2.4:1.2,opacity:active?1:0.34,
        strokeDasharray:edge.role==='saved'?'5 4':undefined},
      markerEnd:{type:MarkerType.ArrowClosed,color},
      ariaLabel:index.get(edge.source).label+' supplies '+edge.value+' to '+index.get(edge.target).label,
    };
  });
  return <section id="mlp" className="mlp" aria-labelledby="mlp-heading">
    <div className="section-head">
      <h2 id="mlp-heading">Qwen3 MLP dependencies</h2>
      <span className="small muted">Stage {stage}{chunks>1?' · C'+(chunk.chunk+1):''} · layer FSDP unit</span>
    </div>
    <div className="controls mlp-controls">
      <label>Layer<select value={layer} onChange={e=>setLayer(Number(e.target.value))}>
        {Array.from({length:chunk.last-chunk.first+1},(_,i)=>chunk.first+i).map(n=><option key={n}>{n}</option>)}
      </select></label>
      <label>Weights after forward<select value={reshard?'reshard':'retain'} onChange={e=>{
        const value = e.target.value==='reshard';
        setReshard(value);
        if (!value&&operation==='gather-b') setOperation('gather-f');
      }}><option value="reshard">Release gathered weights</option><option value="retain">Keep gathered weights</option></select></label>
      <label>Operation<select value={op.id} onChange={e=>select(index.get(e.target.value))}>
        {graph.nodes.map(node=><option key={node.id} value={node.id}>{node.label}</option>)}
      </select></label>
    </div>
    <div className="mlp-details">
      <div>
        <h3>Layer {layer} · {op.label}</h3>
        <p>{op.formula}</p>
        {incoming.length>0&&<ul className="mlp-inputs">
          {incoming.map(edge=><li key={edge.source+'-'+edge.value}>
            <code>{edge.value}</code>{edge.role==='saved'?' saved by ':edge.role==='weight'?' supplied by ':' from '}
            <button onClick={()=>select(index.get(edge.source))}>{index.get(edge.source).label}</button>
          </li>)}
        </ul>}
        <p className="small muted">N = {plan.seq*plan.microBatch} tokens · D = {shape.width} model features · K = {shape.featuresPerRank} MLP features per TP rank</p>
      </div>
      <div>
        <h3>Rank {groups.rank} · FSDP {groups.row} × TP {groups.column}</h3>
        <div className="rank-axis">
          <button aria-pressed={axis==='tp'} onClick={()=>setAxis(axis==='tp'?'both':'tp')}>TP row {groups.row}</button>
          <button aria-pressed={axis==='fsdp'} onClick={()=>setAxis(axis==='fsdp'?'both':'fsdp')}>FSDP column {groups.column}</button>
        </div>
        <div className="rank-mesh" style={{gridTemplateColumns:'repeat('+plan.tp+', minmax(0,1fr))'}} role="group" aria-label="Stage ranks, TP across rows and FSDP down columns">
          {Array.from({length:plan.tp*plan.fsdp},(_,i)=>{
            const number = stage*plan.tp*plan.fsdp+i;
            const tp = axis!=='fsdp'&&groups.tp.includes(number);
            const fsdp = axis!=='tp'&&groups.fsdp.includes(number);
            return <button key={i} className={tp?'rank-tp':fsdp?'rank-fsdp':''}
              onClick={()=>setRank(i)} aria-label={'Rank '+number} aria-pressed={i===rank}>{number}</button>;
          })}
        </div>
        {axis!=='fsdp'&&<p className="small muted">TP ranks {groups.tp.join(', ')} cooperate on the same input.</p>}
        {axis!=='tp'&&<p className="small muted">FSDP ranks {groups.fsdp.join(', ')} share the same TP shard across different data inputs.</p>}
      </div>
    </div>
    <div className="key small mlp-legend"><span>Forward above · backward below</span><span className="saved-key">Dashed amber · saved tensor</span><span className="key-gradient">Purple · weight availability</span></div>
    <div className="map-scroll mlp-scroll" tabIndex={0} role="region" aria-label="Scrollable MLP graph">
      <div className="map-flow mlp-flow">
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} nodesConnectable={false} nodesDraggable={false}
          zoomOnScroll={false} panOnScroll={false} preventScrolling={false}
          defaultViewport={{x:0,y:0,zoom:1}} onNodeClick={(_,node)=>select(index.get(node.id))}>
          <Background gap={24}/><Controls showInteractive={false}/>
        </ReactFlow>
      </div>
    </div>
    <p className="small muted scope">Logical dependencies for an unfused training scenario. Layer collectives include parameters outside the MLP. The stage timeline uses whole forward and backward slots.</p>
  </section>;
}
