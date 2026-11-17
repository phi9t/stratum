import assert from 'node:assert/strict';
import test from 'node:test';
import {mlpGraph, rankGroups} from './mlp.js';

test('weight gradients depend on saved inputs and activation gradients depend on weights', () => {
  for (const reshard of [true,false]) {
    const {nodes,edges} = mlpGraph(reshard);
    const parents = id => edges.filter(edge=>edge.target===id).map(edge=>edge.source).sort();
    assert.deepEqual(parents('dw-down'),['dy','h']);
    assert.deepEqual(parents('dh'),['dy',reshard?'gather-b':'gather-f'].sort());
    assert.deepEqual(parents('dx'),['dx-local']);
    assert.deepEqual(parents('dx-local'),['dx-gate','dx-up']);
    assert.deepEqual(parents('rs'),['dw-down','dw-gate','dw-up','other']);
    assert.equal(nodes.filter(node=>node.kind==='gemm').length,9);
    assert.equal(nodes.some(node=>node.id==='gather-b'),reshard);
    const ids = nodes.map(node=>node.id);
    for (const edge of edges) assert.ok(ids.indexOf(edge.source)<ids.indexOf(edge.target));
  }
});

test('rank groups preserve stage, fixed data row and fixed TP column', () => {
  assert.deepEqual(rankGroups(2,8,2,11),{rank:43,tp:[40,41,42,43,44,45,46,47],fsdp:[35,43],row:1,column:3});
});
