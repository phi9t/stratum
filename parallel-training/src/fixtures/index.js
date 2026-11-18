import qwenShape from '../mlp-shape.json' with {type:'json'};
import plans from '../plans.json' with {type:'json'};
import {meshRealization} from '../graph/realization.js';
import {defineGraph} from '../graph/semantic.js';
import {mlpRecords} from './mlp-records.js';

const typed=(shape,layout='replicated',dtype='bf16')=>({shape,layout,dtype});
const moduleList=[
  {id:'block',label:'Block'},
  {id:'forward',label:'Forward',parent:'block'},
  {id:'backward',label:'Backward',parent:'block'},
];
const qwenSource={
  id:'qwen3-code',title:'Transformers Qwen3MLP',revision:'c93057d4835cd31752bb56f59989dd27696eb45b',
  url:'https://github.com/huggingface/transformers/blob/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/qwen3/modeling_qwen3.py',
  scope:'Gate/up/down forward topology. Backward is an authored derivative expansion.',
};
const tpSource={
  id:'tp-mappings',title:'Megatron TP mappings',revision:'56ef6ed61ccb3edffa138ec581c212d479d9ec06',
  url:'https://github.com/NVIDIA/Megatron-LM/blob/56ef6ed61ccb3edffa138ec581c212d479d9ec06/megatron/core/tensor_parallel/mappings.py',
  scope:'Replicated-output TP convention for the declared scenario.',
};


export function qwenMlp({N=plans['4/1'].seq*plans['4/1'].microBatch,D=qwenShape.width,K=qwenShape.featuresPerRank,reshard=true,stage=0,tp=8,fsdp=2,rank=stage*tp*fsdp,layer=1,microbatch=1}={}) {
  const shapes={};
  for (const value of ['X','P','dY','dXgate','dXup','dXpartial','dX']) shapes[value]=typed([N,D]);
  for (const value of ['G','U','H','dH','dG','dU']) shapes[value]=typed([N,K],'TP feature shard');
  for (const value of ['Wgate','Wup','dWgate','dWup']) shapes[value]=typed([D,K],'TP columns');
  for (const value of ['Wdown','dWdown']) shapes[value]=typed([K,D],'TP rows');
  shapes['Other gradients']={shape:null,unknownReason:'Other decoder-layer parameters are outside this MLP expansion.',dtype:'bf16',layout:'TP-local parameters'};
  for (const name of ['Wgate','Wup','Wdown']) shapes[name].parameter=name;
  const records=mlpRecords(reshard);
  records.find(node=>node.id==='y').outputs=['Y'];
  records.find(node=>node.id==='dx').outputs=['dX'];
  shapes.Y=typed([N,D]);
  const graph=defineGraph({
    id:'qwen3-mlp',title:'Qwen3 MLP',scope:'derived scenario',
    modules:moduleList,records,shapes,sources:[qwenSource,tpSource],
    parameters:[
      {id:'Wgate',shape:[D,K],layout:'TP columns'},
      {id:'Wup',shape:[D,K],layout:'TP columns'},
      {id:'Wdown',shape:[K,D],layout:'TP rows'},
    ],
    context:{rank,layer,microbatch},realization:meshRealization({stage,tp,fsdp}),
  });
  graph.realization.policy={reshardAfterForward:reshard,unit:'decoder-layer',normalization:'configured externally'};
  return graph;
}

function record(id,label,formula,kind,module,inputs=[],outputs=[]) {
  return {id,label,formula,kind,module,phase:'forward',inputs,outputs};
}
export function attentionResidual({N=8,D=16,A=8}={}) {
  return defineGraph({
    id:'attention-residual',title:'Attention + residual',scope:'illustrative',
    modules:[{id:'block',label:'Attention block'},{id:'attention',label:'Attention',parent:'block'},{id:'residual',label:'Residual',parent:'block'}],
    sources:[{id:'attention-paper',title:'Attention Is All You Need',url:'https://arxiv.org/abs/1706.03762',
      scope:'Scaled dot-product attention and residual structure. Teaching dimensions, forward only.'}],
    parameters:[{id:'Wq',shape:[D,A]},{id:'Wk',shape:[D,A]},{id:'Wv',shape:[D,A]},{id:'Wo',shape:[A,D]}],
    shapes:{
      X:typed([N,D]),Wq:typed([D,A]),Wk:typed([D,A]),Wv:typed([D,A]),Wo:typed([A,D]),
      Q:typed([N,A]),K:typed([N,A]),V:typed([N,A]),S:typed([N,N]),P:typed([N,N]),
      C:typed([N,A]),O:typed([N,D]),Y:typed([N,D]),
    },
    records:[
      record('weights','Projection weights','Wq, Wk, Wv and Wo','input','attention'),
      record('x','Residual input','X · [N, D]','input','residual'),
      record('q','Query projection','Q = X Wq','gemm','attention',[['x','X'],['weights','Wq','weight']]),
      record('k','Key projection','K = X Wk','gemm','attention',[['x','X'],['weights','Wk','weight']]),
      record('v','Value projection','V = X Wv','gemm','attention',[['x','X'],['weights','Wv','weight']]),
      record('scores','Scaled scores','S = Q Kᵀ / √A','gemm','attention',[['q','Q'],['k','K']]),
      record('softmax','Causal softmax','P = softmax(causal_mask(S))','compute','attention',[['scores','S']]),
      record('context','Weighted values','C = P V','gemm','attention',[['softmax','P'],['v','V']]),
      record('out','Output projection','O = C Wo','gemm','attention',[['context','C'],['weights','Wo','weight']]),
      record('add','Residual add','Y = X + O','compute','residual',[['x','X'],['out','O']],['Y']),
    ],
  });
}

export function sparseMixer({N=8,D=16,V=32}={}) {
  return defineGraph({
    id:'sparse-mixer',title:'Sparse lookup + dense mixer',scope:'illustrative',
    modules:[{id:'block',label:'Illustrative sparse mixer'},
      {id:'lookup-block',label:'Sparse lookup',parent:'block'},{id:'mixer',label:'Dense mixer',parent:'block'}],
    sources:[],
    parameters:[{id:'E',shape:[V,D]},{id:'W',shape:[D,D]}],
    shapes:{X:typed([N,D]),I:typed([N],'token indices','int64'),E:typed([V,D],'lookup table'),
      L:typed([N,D]),W:typed([D,D]),Z:typed([N,D]),G:typed([N,D]),H:typed([N,D]),Y:typed([N,D])},
    records:[
      record('table','Sparse table','E · [V, D]','input','lookup-block'),
      record('indices','Lookup indices','I · [N], integer row ids','input','lookup-block'),
      record('x','Dense input','X · [N, D]','input','mixer'),
      record('weight','Mixer weights','W · [D, D]','input','mixer'),
      record('lookup','Sparse lookup','L = E[I]','lookup','lookup-block',[['table','E','weight'],['indices','I']]),
      record('dense','Dense projection','Z = X W','gemm','mixer',[['x','X'],['weight','W','weight']]),
      record('gate','Mixer gate','G = sigmoid(Z)','compute','mixer',[['dense','Z']]),
      record('mix','Gated lookup','H = G ⊙ L','compute','mixer',[['gate','G'],['lookup','L']]),
      record('add','Residual add','Y = X + H','compute','mixer',[['x','X'],['mix','H']],['Y']),
    ],
  });
}
