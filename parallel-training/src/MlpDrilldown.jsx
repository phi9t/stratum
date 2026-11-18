import {useEffect,useMemo,useState} from 'react';
import {allNodes} from './graph/semantic.js';
import {graphHash,readGraphState} from './graph/state.js';
import GraphExplorer from './graph/GraphExplorer.jsx';
import {rankGroups} from './mlp.js';
import {fixtureCatalog} from './fixtures/index.js';
import shape from './mlp-shape.json';

const choices=fixtureCatalog;
function fixtureFromHash() {
  const id=new URLSearchParams(location.hash.slice(1)).get('fixture');
  return choices.some(choice=>choice.id===id)?id:'mlp';
}
export default function MlpDrilldown({plan,stage,chunk,chunks,microbatch=1}) {
  const [fixture,setFixture]=useState(fixtureFromHash);
  const [layer,setLayer]=useState(chunk.first);
  const [reshard,setReshard]=useState(true);
  const [rank,setRank]=useState(0);
  const [axis,setAxis]=useState('both');
  const groups=rankGroups(stage,plan.tp,plan.fsdp,rank);
  const graph=useMemo(()=>{
    return choices.find(choice=>choice.id===fixture).create({N:plan.seq*plan.microBatch,D:shape.width,K:shape.featuresPerRank,
      reshard,stage,tp:plan.tp,fsdp:plan.fsdp,rank:groups.rank,layer,microbatch});
  },[fixture,plan,reshard,stage,rank,layer,microbatch]);
  const [state,setState]=useState(()=>readGraphState(location.hash,graph));
  useEffect(()=>{
    const navigate=()=>{
      const id=fixtureFromHash();
      const target=id===fixture?graph:choices.find(choice=>choice.id===id).create();
      setFixture(id);
      setState(readGraphState(location.hash,target));
    };
    addEventListener('hashchange',navigate);
    return ()=>removeEventListener('hashchange',navigate);
  },[graph]);
  const nodes=allNodes(graph);
  const selected=nodes.some(node=>node.id===state.selected)?state.selected:nodes[0].id;
  const collapsed=new Set([...state.collapsed].filter(id=>graph.modules.some(module=>module.id===id)));
  // Memoize the membership identity so selection changes do not restart ELK layout.
  const collapseKey=[...collapsed].sort().join(',');
  const stableCollapsed=useMemo(()=>collapsed,[graph,collapseKey]);
  function update(patch) {
    const next={...state,fixture,selected,collapsed:stableCollapsed,...patch};
    setState(next);
    history.pushState(null,'',graphHash(next));
  }
  function select(id) {
    update({selected:id});
    const node=nodes.find(node=>node.id===id);
    if (node?.collective) setAxis(node.collective.axis);
  }
  function chooseFixture(id) {
    setFixture(id);
    const nextGraph=choices.find(choice=>choice.id===id).create();
    update({fixture:id,selected:allNodes(nextGraph)[0].id,collapsed:new Set(),panel:'dependencies'});
  }
  const op=nodes.find(node=>node.id===selected);
  return <section id="mlp" className="mlp" aria-labelledby="mlp-heading">
    <div className="section-head"><h2 id="mlp-heading">Model dependencies</h2>
      <span className="small muted">{graph.scope}{fixture==='mlp'?' · Stage '+stage+(chunks>1?' · C'+(chunk.chunk+1):'')+' · layer '+layer:''}</span>
    </div>
    <div className="controls mlp-controls">
      <label>Graph<select aria-label="Graph" value={fixture} onChange={event=>chooseFixture(event.target.value)}>
        {choices.map(choice=><option key={choice.id} value={choice.id}>{choice.label}</option>)}
      </select></label>
      {fixture==='mlp'&&<>
        <label>Layer<select aria-label="Layer" value={layer} onChange={event=>setLayer(Number(event.target.value))}>
          {Array.from({length:chunk.last-chunk.first+1},(_,i)=>chunk.first+i).map(n=><option key={n}>{n}</option>)}
        </select></label>
        <label>Weights after forward<select aria-label="Weights after forward" value={reshard?'reshard':'retain'} onChange={event=>setReshard(event.target.value==='reshard')}>
          <option value="reshard">Release gathered weights</option><option value="retain">Keep gathered weights</option>
        </select></label>
      </>}
      <label>Operation<select aria-label="Operation" value={selected} onChange={event=>select(event.target.value)}>
        {nodes.map(node=><option key={node.id} value={node.id}>{node.label}</option>)}
      </select></label>
    </div>
    {fixture==='mlp'&&<div className="mesh-context">
      <div>
        <h3>Rank {groups.rank} · FSDP {groups.row} × TP {groups.column}</h3>
        <div className="rank-axis">
          <button aria-pressed={axis==='tp'} onClick={()=>setAxis(axis==='tp'?'both':'tp')}>TP row {groups.row}</button>
          <button aria-pressed={axis==='fsdp'} onClick={()=>setAxis(axis==='fsdp'?'both':'fsdp')}>FSDP column {groups.column}</button>
        </div>
        <div className="rank-mesh" style={{gridTemplateColumns:'repeat('+plan.tp+', minmax(0,1fr))'}} role="group" aria-label="Stage ranks, TP across rows and FSDP down columns">
          {graph.realization.meshes[0].ranks.map((number,i)=>{
            const tp=axis!=='fsdp'&&groups.tp.includes(number);
            const fsdp=axis!=='tp'&&groups.fsdp.includes(number);
            return <button key={number} className={tp?'rank-tp':fsdp?'rank-fsdp':''}
              onClick={()=>setRank(i)} aria-label={'Rank '+number} aria-pressed={i===rank}>{number}</button>;
          })}
        </div>
      </div>
      <div>
        <p className="small muted">N = {plan.seq*plan.microBatch} · D = {shape.width} · K = {shape.featuresPerRank} per TP rank</p>
        {axis!=='fsdp'&&<p className="small muted">TP ranks {groups.tp.join(', ')} cooperate on the same input.</p>}
        {axis!=='tp'&&<p className="small muted">FSDP ranks {groups.fsdp.join(', ')} share the same TP shard across data inputs.</p>}
        {op.collective&&<p className="small">Selected collective — {op.collective.operation} over {op.collective.axis==='tp'?groups.tp.join(', '):groups.fsdp.join(', ')}</p>}
      </div>
    </div>}
    {fixture==='sparse'&&<p className="small muted">Illustrative Civet-style lookup/mixer motif. The topology and dimensions are authored teaching choices.</p>}
    <GraphExplorer graph={graph} selected={selected} onSelect={select} collapsed={stableCollapsed}
      onToggle={id=>{const next=new Set(stableCollapsed);next.has(id)?next.delete(id):next.add(id);update({collapsed:next});}}
      panel={state.panel} onPanel={panel=>update({panel})}/>
    <p className="small muted scope">{fixture==='mlp'?'Unfused training scenario with decoder-layer FSDP units.':'Illustrative forward dependencies.'} The 1F1B timeline shows the Qwen3 stage scenario above.</p>
  </section>;
}
