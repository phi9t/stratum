import {useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Background, Controls, Handle, MarkerType, Position, ReactFlow} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './explorer.css';
import './canvas.css';
import './site.css';
import plans from './plans.json';
import {buildSchedule} from './schedule.js';
import {inspectSchedule} from './inspect.js';
import MlpDrilldown from './MlpDrilldown.jsx';

const bytes = amount => (amount / 2 ** 20).toFixed(1) + ' MiB';
const flops = amount => (amount / 1e12).toFixed(2) + ' TFLOP';
const name = (node, chunks) => 'Stage ' + node.stage + (chunks > 1 ? ' · C' + (node.chunk + 1) : '');
const options = [1,2,4,8,16];
const passes = {F:'Activation', B:'Gradient'};

function Select({label, value, choices, onChange}) {
  return <label>{label}<select value={value} onChange={event=>onChange(Number(event.target.value))}>
    {choices.map(choice => <option key={choice.value ?? choice} value={choice.value ?? choice}>{choice.label ?? choice}</option>)}
  </select></label>;
}

function Stage({data}) {
  const {node, chunks, tp, fsdp} = data;
  return <div className={'stage s' + (node.stage % 6) + (data.selected ? ' selected' : '')}>
    <Handle id="f-in" type="target" position={Position.Left} style={{top:'32%'}}/>
    <Handle id="f-out" type="source" position={Position.Right} style={{top:'32%'}}/>
    <Handle id="b-in" type="target" position={Position.Right} style={{top:'73%'}}/>
    <Handle id="b-out" type="source" position={Position.Left} style={{top:'73%'}}/>
    <Handle id="f-wrap-out" type="source" position={Position.Bottom} style={{left:'70%'}}/>
    <Handle id="b-wrap-in" type="target" position={Position.Bottom} style={{left:'30%'}}/>
    <Handle id="f-wrap-in" type="target" position={Position.Top} style={{left:'70%'}}/>
    <Handle id="b-wrap-out" type="source" position={Position.Top} style={{left:'30%'}}/>
    <button className="stage-head" onClick={data.select} aria-pressed={data.selected}>
      <span className="title">{name(node,chunks)}</span>
      <span className="meta">Layers {node.first}–{node.last} · ranks {node.stage*tp*fsdp}–{(node.stage+1)*tp*fsdp-1}</span>
      <span className="meta">{bytes(node.params)} params / rank</span>
    </button>
    <div className="mesh" style={{gridTemplateColumns:'repeat('+tp+',1fr)'}} role="img" aria-label={fsdp + ' FSDP rows by ' + tp + ' tensor-parallel columns'}>
      {Array.from({length:tp*fsdp},(_,index)=><span className="tile" key={index}/>)}
    </div>
  </div>;
}
const nodeTypes = {stage:Stage};

function Connectivity({plan, count, chunks, selected, selectedChunk, setSelected, focused, setFocused, messages, children}) {
  const nodes = plan.nodes.map((node,index)=>({
    id:String(index),type:'stage',
    position:{x:24+node.stage*254, y:24+node.chunk*190},
    style:{width:198,height:112},
    data:{node,chunks,tp:plan.tp,fsdp:plan.fsdp,selected:selected===node.stage&&selectedChunk===node.chunk,select:()=>setSelected(node.stage,node.chunk)}
  }));
  const edges = plan.nodes.slice(1).flatMap((node,index)=>['F','B'].map(pass=>{
    const forward = pass==='F';
    const wrap = node.chunk !== plan.nodes[index].chunk;
    const kind = pass.toLowerCase();
    const id = index + pass;
    const active = focused === id;
    const color = forward ? '#127280' : '#7c43ae';
    return {
      id,source:String(forward?index:index+1),target:String(forward?index+1:index),
      sourceHandle:kind+(wrap?'-wrap':'')+'-out',targetHandle:kind+(wrap?'-wrap':'')+'-in',
      type:'smoothstep', pathOptions:{offset:wrap?16:12,stepPosition:wrap?0.31:0.5},
      style:{stroke:color,strokeWidth:active?4:2,opacity:focused && !active ? 0.22 : 1},
      markerEnd:{type:MarkerType.ArrowClosed,color},
      ariaLabel:passes[pass]+' from '+name(plan.nodes[forward?index:index+1],chunks)+' to '+name(plan.nodes[forward?index+1:index],chunks),
      zIndex:active?4:0,
    };
  }));
  return <section aria-labelledby="handoff-heading">
    <div className="section-head"><h2 id="handoff-heading">Stage connectivity</h2><span className="small muted">{bytes(plan.activationBytes)} forward activation per boundary and microbatch</span></div>
    <div className="map-scroll" tabIndex={0} role="region" aria-label="Scrollable stage connectivity">
      <div className="map-flow" style={{minWidth:count*254,height:chunks===1?166:367}}>
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} nodesConnectable={false} nodesDraggable={false}
          panOnDrag={false} zoomOnScroll={false} panOnScroll={false} preventScrolling={false}
          defaultViewport={{x:0,y:0,zoom:1}} onEdgeClick={(_,edge)=>setFocused(focused===edge.id?null:edge.id)}>
          <Background gap={24}/><Controls showInteractive={false}/>
        </ReactFlow>
      </div>
    </div>
    <div className="key small"><span className="key-activation">Activation →</span><span className="key-gradient">Gradient →</span></div>
    {children}
    {messages.length>0 && <details className="messages" open>
      <summary>Traced microbatch · {messages.length} messages</summary>
      <div className="message-scroll"><table className="message-table">
        <thead><tr><th>Message</th><th>From</th><th>To</th><th>Ready at</th><th>Used at</th></tr></thead>
        <tbody>{messages.map(message=>{
          const id = message.boundary+message.pass;
          return <tr key={id} className={focused===id?'highlight':''}>
            <td><button onClick={()=>setFocused(focused===id?null:id)} aria-pressed={focused===id}>{passes[message.pass]}</button></td>
            <td>{name(plan.nodes[message.from],chunks)}</td><td>{name(plan.nodes[message.to],chunks)}</td>
            <td>{message.ready}</td><td>{message.use}</td>
          </tr>;
        })}</tbody>
      </table></div>
    </details>}
  </section>;
}

function Timeline({rows, inventory, chunks, trace, selected, details, message}) {
  const scale = Math.max(...inventory.map(values=>Math.max(...values)));
  const y = live => 29 - live/scale*24;
  return <section aria-labelledby="timeline-heading">
    <div className="section-head"><h2 id="timeline-heading">Schedule &amp; live activations</h2>
      <span className="small muted">Retained forward states · shared 0–{scale} scale</span>
    </div>
    <div className="key small"><span className="slot-f">F · Forward</span><span className="slot-b">B · Backward</span>
      <span className="slot-idle">· Idle</span><span className="key-live">━ Live forward state</span></div>
    <div className="schedule-scroll" tabIndex={0} role="region" aria-label="Scrollable schedule and activation timeline">
      <table className="schedule-table timeline" aria-label="Device groups by normalized time">
        <thead><tr><th scope="col">Stage / time →</th>
          {rows[0].map((_,slot)=><th scope="col" key={slot}>{(slot+1)/chunks}</th>)}
        </tr></thead>
        {rows.map((row,index)=><tbody key={index} className={selected===index?'highlight':''}>
          <tr><th scope="row">Stage {index}
            {details[index].map(node=><small key={node.chunk}>{chunks>1?'C'+(node.chunk+1)+' · ':''}layers {node.first}–{node.last}</small>)}
            <small className="peak-live">Peak {Math.max(...inventory[index])} live</small>
          </th>
            {row.map((task,slot)=>{
              const label = task ? (task.pass==='F'?'Forward':'Backward')+', microbatch '+task.batch+
                (chunks>1?', chunk '+(task.chunk+1):'') : 'Idle';
              const marksMessage = message && (
                (index===message.from%rows.length && slot===message.ready*chunks-1) ||
                (index===message.to%rows.length && slot===message.use*chunks)
              );
              const color = task ? 'slot-'+task.pass.toLowerCase()+(trace&&trace!==task.batch?' other-batch':'') : 'slot-idle';
              return <td key={slot} aria-label={label} title={label} className={color+(marksMessage?' message-task':'')}>
                {task?<>{task.pass}{task.batch}{chunks>1&&<small>C{task.chunk+1}</small>}</>:'·'}
              </td>;
            })}
          </tr>
          <tr className="inventory"><th scope="row"><small>Stage {index} live</small></th>
            {row.map((task,slot)=>{
              const start = inventory[index][slot];
              const end = inventory[index][slot+1];
              const label = 'Time '+(slot+1)/chunks+' · '+end+' live'+(start!==end?(end>start?' · state retained':' · state freed'):'');
              return <td key={slot} title={label} aria-label={label}>
                <svg viewBox="0 0 40 34" preserveAspectRatio="none" aria-hidden="true">
                  <path className="inventory-base" d="M0 29H40"/>
                  <path className="inventory-fill" d={'M0 29V'+y(start)+'H40V29Z'}/>
                  <path className="inventory-line" d={'M0 '+y(start)+'H40V'+y(end)}/>
                </svg>
              </td>;
            })}
          </tr>
        </tbody>)}
      </table>
    </div>
    <p className="small muted">Forward state enters the track at the end of F and leaves at the end of B. Each interleaved chunk contributes a separate state.</p>
  </section>;
}

function App() {
  const [chunks,setChunks] = useState(1);
  const [count,setCount] = useState(4);
  const [requestedBatches,setBatches] = useState(8);
  const [requestedTrace,setTrace] = useState(1);
  const [selected,setSelected] = useState(0);
  const [requestedChunk,setChunk] = useState(0);
  const selectedChunk = Math.min(requestedChunk,chunks-1);
  const [focused,setFocused] = useState(null);
  const choices = options.filter(n=>chunks===1||n%count===0);
  const batches = choices.includes(requestedBatches)?requestedBatches:choices.find(n=>n>=requestedBatches);
  const trace = requestedTrace<=batches?requestedTrace:0;
  const plan = plans[count+'/'+chunks];
  const rows = useMemo(()=>buildSchedule(count,batches,chunks),[count,batches,chunks]);
  const {inventory,messages} = useMemo(()=>inspectSchedule(rows,chunks,trace),[rows,chunks,trace]);
  const details = rows.map((_,stage)=>plan.nodes.filter(node=>node.stage===stage));
  const idle = (rows[0].length-2*batches*chunks)/chunks;
  const message = messages.find(entry=>entry.boundary+entry.pass===focused);
  return <>
    <header className="bar"><h1>1F1B microbatch schedule</h1><span className="summary">{plan.model} · TP {plan.tp} × FSDP {plan.fsdp} · sequence {plan.seq}</span></header>
    <main className="schedule-site">
      <div className="controls scenario-controls">
        <Select label="Variant" value={chunks} choices={[{value:1,label:'Standard 1F1B'},{value:2,label:'Interleaved 1F1B'}]} onChange={value=>{setChunks(value);setFocused(null);}}/>
        <Select label="Device groups" value={count} choices={[2,4,8]} onChange={value=>{setCount(value);setSelected(0);setChunk(0);setFocused(null);}}/>
        <Select label="Microbatches" value={batches} choices={choices} onChange={value=>{setBatches(value);setFocused(null);}}/>
        <Select label="Trace" value={trace} choices={Array.from({length:batches+1},(_,value)=>({value,label:value?'Microbatch '+value:'All microbatches'}))} onChange={value=>{setTrace(value);setFocused(null);}}/>
      </div>
      <p className="schedule-readout" aria-live="polite">{rows[0].length/chunks} time units · {idle} idle per stage</p>
      <Timeline rows={rows} inventory={inventory} chunks={chunks} trace={trace} selected={selected} details={details} message={message}/>
      <Connectivity key={count+'/'+chunks} plan={plan} count={count} chunks={chunks} selected={selected} selectedChunk={selectedChunk} setSelected={(stage,chunk)=>{setSelected(stage);setChunk(chunk);}}
        focused={focused} setFocused={setFocused} messages={messages}>
      <section className="selected-stage" aria-labelledby="detail-heading">
        <div className="section-head"><h2 id="detail-heading">Stage {selected} details</h2><span className="small muted">Per rank</span></div>
        <table className="balance stage-details"><thead><tr><th scope="col">Layers</th><th scope="col">Parameters</th><th scope="col">Forward FLOPs / microbatch</th></tr></thead>
          <tbody>{details[selected].map(node=><tr key={node.chunk}><td>{node.first}–{node.last}</td><td>{bytes(node.params)}</td><td>{flops(node.flops)}</td></tr>)}</tbody>
        </table>
        <a className="mlp-link" href="#mlp">Inspect layer MLP</a>
      </section>
      </Connectivity>
      <MlpDrilldown key={[count,chunks,selected,selectedChunk].join("/")} plan={plan} stage={selected}
        chunk={details[selected][selectedChunk]} chunks={chunks} microbatch={trace||1}/>
      <p className="small muted scope">A full-stage task takes one time unit. Each interleaved chunk takes half a unit. Ready and used mark producer completion and consumer start. The schedule assumes balanced work and omits transfer time and recomputation.</p>
    </main>
  </>;
}
createRoot(document.getElementById('app')).render(<App/>);
