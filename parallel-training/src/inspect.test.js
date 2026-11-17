import assert from 'node:assert/strict';
import test from 'node:test';
import {buildSchedule} from './schedule.js';
import {inspectSchedule} from './inspect.js';

test('each forward opens state, each backward closes it, and the boundary messages run in opposite directions', () => {
  for (const chunks of [1,2]) {
    const rows = buildSchedule(4,8,chunks);
    const {inventory,messages} = inspectSchedule(rows,chunks,2);
    assert.deepEqual(inventory.map(values=>Math.max(...values)),chunks===1?[4,3,2,1]:[11,9,7,5]);
    inventory.forEach((values,stage)=>{
      assert.equal(values[0],0);
      assert.equal(values.at(-1),0);
      rows[stage].forEach((task,slot)=>assert.equal(values[slot+1]-values[slot],task?.pass==='F'?1:task?.pass==='B'?-1:0));
    });
    assert.equal(messages.length,2*(4*chunks-1));
    for (const {boundary,from,to,pass,ready,use} of messages) {
      assert.equal(Math.min(from,to),boundary);
      assert.equal(to-from,pass==='F'?1:-1);
      assert.ok(use>=ready);
    }
    if (chunks===2) {
      assert.ok(messages.some(m=>m.from===3 && m.to===4 && m.pass==='F'));
      assert.ok(messages.some(m=>m.from===4 && m.to===3 && m.pass==='B'));
    }
  }
});
