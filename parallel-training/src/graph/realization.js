export function meshRealization({stage=0,tp=8,fsdp=2,module='block'}={}) {
  const id='stage/'+stage;
  const ranks=Array.from({length:tp*fsdp},(_,i)=>stage*tp*fsdp+i);
  return {
    meshes:[{id,axes:[{id:'fsdp',size:fsdp},{id:'tp',size:tp}],ranks}],
    groups:[
      ...Array.from({length:fsdp},(_,f)=>({id:id+'/tp/'+f,axis:'tp',mesh:id,coordinate:f,
        ranks:ranks.slice(f*tp,(f+1)*tp)})),
      ...Array.from({length:tp},(_,t)=>({id:id+'/fsdp/'+t,axis:'fsdp',mesh:id,coordinate:t,
        ranks:Array.from({length:fsdp},(_,f)=>ranks[f*tp+t])})),
    ],
    placements:[{module,mesh:id,ranks}],
  };
}
