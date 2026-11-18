import {allNodes} from './semantic.js';
export function readGraphState(hash,graph) {
  const query=new URLSearchParams(hash.replace(/^#/,''));
  const id=query.get('fixture');
  const selected=allNodes(graph).find(node=>node.id===query.get('operation'))?.id??allNodes(graph)[0].id;
  const collapsed=new Set((query.get('collapsed')??'').split(',').filter(id=>graph.modules.some(m=>m.id===id)));
  return {fixture:id,selected,collapsed,panel:query.get('panel')==='evidence'?'evidence':'dependencies'};
}
export function graphHash({fixture,selected,collapsed,panel}) {
  return '#'+new URLSearchParams({fixture,operation:selected,collapsed:[...collapsed].sort().join(','),panel});
}
