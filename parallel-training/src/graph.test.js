import assert from 'node:assert/strict';
import test from 'node:test';
import {validateGraph,allNodes,allEdges} from './graph/semantic.js';
import {projectGraph} from './graph/projection.js';
import {qwenMlp,attentionResidual,sparseMixer} from './fixtures/index.js';

test('all fixtures have typed resolvable ports and no view coordinates',()=>{
  for (const graph of [qwenMlp({N:3,D:8,K:4}),attentionResidual(),sparseMixer()]) {
    assert.deepEqual(validateGraph(graph),[]);
    assert.equal(graph.schema,'stratum-semantic-graph/v1');
    for (const node of allNodes(graph)) {
      assert.equal(node.position,undefined);
      assert.equal(node.column,undefined);
      for (const port of node.ports) assert.ok(graph.tensors.some(t=>t.id===port.tensor));
    }
  }
});

test('contraction preserves every external tensor edge and original endpoint identity',()=>{
  const graph=qwenMlp({N:3,D:8,K:4});
  const collapsed=new Set(['forward','backward']);
  const view=projectGraph(graph,collapsed);
  assert.equal(view.edges.length+view.internalEdges.length,allEdges(graph).length);
  assert.equal(new Set(view.edges.map(e=>e.id)).size,view.edges.length);
  const pair=view.edges.filter(e=>e.source==='forward'&&e.target==='backward');
  assert.ok(pair.length>=2,'distinct saved and weight paths survive contraction');
  for (const edge of view.edges) {
    assert.equal(edge.members.length,1);
    assert.ok(edge.original.source.port);
    assert.ok(edge.original.target.port);
    const node=view.nodes.find(n=>n.id===edge.source);
    assert.ok(node.ports.some(p=>p.id===edge.sourceHandle));
  }
  assert.deepEqual(projectGraph(graph,new Set()).edges.map(e=>e.id),allEdges(graph).map(e=>e.id));
});

test('malformed ports, tensor shapes, module ownership and cycles fail validation',()=>{
  const pristine=qwenMlp({N:3,D:8,K:4});
  for (const mutate of [
    g=>{allEdges(g)[0].target.port='missing';},
    g=>{g.tensors[0].shape=[-1,8];},
    g=>{g.model.nodes[0].module='missing';},
    g=>{g.modules[0].parent=g.modules[0].id;},
    g=>{allEdges(g)[0].tensor='missing';},
  ]) {
    const graph=structuredClone(pristine);
    mutate(graph);
    assert.ok(validateGraph(graph).length);
  }
});

test('MLP shape contract and phase boundaries work across dimensions and weight policies',()=>{
  for (const reshard of [true,false]) {
    const graph=qwenMlp({N:7,D:12,K:6,reshard});
    const tensors=Object.fromEntries(graph.tensors.map(t=>[t.id,t]));
    assert.deepEqual(tensors['gate/G'].shape,[7,6]);
    assert.deepEqual(tensors['down/P'].shape,[7,12]);
    assert.deepEqual(tensors['dw-down/dWdown'].shape,[6,12]);
    assert.deepEqual(tensors['dw-gate/dWgate'].shape,[12,6]);
    assert.equal(graph.realization.nodes.some(n=>n.id==='gather-b'),reshard);
    assert.equal(allNodes(graph).filter(n=>n.kind==='gemm').length,9);
    assert.ok(graph.execution.instances.every(i=>i.phase&&i.node&&i.rank!==undefined));
    assert.equal(graph.observations.length,0);
  }
});

test('ELK lays out every visible node and routes each port without overlap',async()=>{
  const {layoutGraph}=await import('./graph/layout.js');
  for (const graph of [qwenMlp({N:3,D:8,K:4}),attentionResidual(),sparseMixer()]) {
    for (const collapsed of [new Set(),new Set(['forward','attention','lookup-block'])]) {
      const view=await layoutGraph(projectGraph(graph,collapsed));
      assert.ok(Number.isFinite(view.width)&&view.width>0);
      for (const [i,node] of view.nodes.entries()) {
        assert.ok(node.position.x>=0&&node.position.y>=0);
        assert.ok(node.position.x+node.width<=view.width);
        for (const other of view.nodes.slice(i+1)) {
          const overlap=node.position.x<other.position.x+other.width&&other.position.x<node.position.x+node.width&&
            node.position.y<other.position.y+other.height&&other.position.y<node.position.y+node.height;
          assert.equal(overlap,false);
        }
      }
      assert.ok(view.edges.every(edge=>edge.sections.length));
      for (const edge of view.edges) for (const section of edge.sections) {
        const points=[section.startPoint,...(section.bendPoints??[]),section.endPoint];
        for (const [i,point] of points.entries()) {
          assert.ok(Number.isFinite(point.x)&&Number.isFinite(point.y));
          assert.ok(point.x>=0&&point.y>=0&&point.x<=view.width&&point.y<=view.height);
          if (i) assert.ok(Math.min(Math.abs(point.x-points[i-1].x),Math.abs(point.y-points[i-1].y))<1e-8);
        }
      }
    }
  }
});

test('selection survives collapse, and a new graph rejects unrelated URL identity',async()=>{
  const {readGraphState,graphHash}=await import('./graph/state.js');
  const graph=qwenMlp();
  const snapshot={fixture:'mlp',selected:'dw-up',collapsed:new Set(['backward']),panel:'dependencies'};
  const restored=readGraphState(graphHash(snapshot),graph);
  assert.equal(restored.selected,'dw-up');
  assert.equal(projectGraph(graph,restored.collapsed).representative.get(restored.selected),'backward');
  assert.equal(readGraphState(graphHash(snapshot),attentionResidual()).selected,'weights');
});

test('plan adapter retains parallel input occurrences and unknown shapes honestly',async()=>{
  const {adaptPlan}=await import('./graph/plan-adapter.js');
  const built={
    graph:{id:'fixture-plan',nodes:[
      {id:'a',label:'producer',namespace:'pp0/layer1/attention',outputsMetadata:[
        {id:'q',attrs:[{key:'shape',value:'[2, 4]'}]},{id:'k',attrs:[{key:'shape',value:'[2, 4]'}]}],incomingEdges:[]},
      {id:'b',label:'consumer',namespace:'pp0/layer1/mlp',outputsMetadata:[],incomingEdges:[
        {sourceNodeId:'a',sourceNodeOutputId:'q',targetNodeInputId:'0'},
        {sourceNodeId:'a',sourceNodeOutputId:'k',targetNodeInputId:'0'},
      ]},
    ]},ops:[],resolved:{m:{name:'Adapter test'}},
  };
  const graph=adaptPlan(built);
  assert.deepEqual(validateGraph(graph),[]);
  const view=projectGraph(graph,new Set(['module:pp0/layer1/attention','module:pp0/layer1/mlp']));
  assert.equal(view.edges.length,2);
  assert.notEqual(view.edges[0].targetHandle,view.edges[1].targetHandle);
  assert.deepEqual(view.edges.map(edge=>edge.original.upstream.sourceNodeOutputId),['q','k']);
});

test('attention and lookup fixtures preserve branch shapes and declared scope',()=>{
  const attention=attentionResidual({N:5,D:12,A:3});
  const tensors=new Map(attention.tensors.map(t=>[t.id,t]));
  assert.deepEqual(tensors.get('scores/S').shape,[5,5]);
  assert.deepEqual(tensors.get('context/C').shape,[5,3]);
  assert.deepEqual(tensors.get('add/Y').shape,[5,12]);
  const sparse=sparseMixer({N:5,D:12,V:20});
  assert.deepEqual(sparse.parameters.find(p=>p.id==='E').shape,[20,12]);
  assert.equal(sparse.scope,'illustrative');
  assert.deepEqual(sparse.sources,[]);
  assert.equal(sparse.realization.nodes.length,0);
});

test('runtime identity, mesh membership and parameter shape drift fail closed',()=>{
  const graph=qwenMlp();
  for (const mutate of [
    g=>{g.realization.groups[0].ranks[0]=999;},
    g=>{g.execution.instances[0].rank=999;},
    g=>{g.parameters[0].shape=[1,1];},
    g=>{
      const edge=allEdges(g)[0];
      const node=allNodes(g).find(n=>n.id===edge.target.node);
      node.ports.push({id:'out:cycle',direction:'output',tensor:edge.tensor});
      edge.source={node:node.id,port:'out:cycle'};
    },
  ]) {
    const broken=structuredClone(graph);
    mutate(broken);
    assert.ok(validateGraph(broken).length);
  }
});

test('outer contraction wins and expanding restores the same port/tensor identities',()=>{
  const graph=qwenMlp();
  const before=projectGraph(graph,new Set());
  const hidden=projectGraph(graph,new Set(['block','forward']));
  assert.deepEqual([...new Set(hidden.representative.values())],['block']);
  assert.equal(hidden.edges.length,0);
  const after=projectGraph(graph,new Set());
  assert.deepEqual(after,before);
});

test('one tensor can feed two input ports on the same operator without edge merging',async()=>{
  const {defineGraph}=await import('./graph/semantic.js');
  const graph=defineGraph({
    id:'repeated-operand',title:'Repeated tensor',scope:'illustrative',
    modules:[{id:'inputs',label:'Inputs'},{id:'compute',label:'Compute'}],parameters:[],
    shapes:{X:{shape:[2,4],dtype:'fp32',layout:'replicated'}},
    records:[
      {id:'x',label:'X',kind:'input',formula:'X',phase:'forward',module:'inputs'},
      {id:'mul',label:'Multiply',kind:'compute',formula:'X ⊙ X',phase:'forward',module:'compute',
        inputs:[['x','X','data','left'],['x','X','data','right']]},
    ],
  });
  const view=projectGraph(graph,new Set(['inputs','compute']));
  assert.equal(view.edges.length,2);
  assert.notEqual(view.edges[0].targetHandle,view.edges[1].targetHandle);
  assert.equal(view.edges[0].tensor,view.edges[1].tensor);
});
