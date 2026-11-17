// The interleaved order groups a full wave of microbatches by chunk.
// Each physical stage runs one task per slot and sees completed work from prior slots.
export function buildSchedule(stageCount, microbatches, chunks) {
  const groupSize = chunks === 1 ? 1 : stageCount;
  const forward = [];
  const backward = [];
  for (let first = 1; first <= microbatches; first += groupSize) {
    for (let chunk = 0; chunk < chunks; chunk++) {
      for (let batch = first; batch < Math.min(first + groupSize, microbatches + 1); batch++) {
        forward.push({pass: 'F', batch, chunk});
        backward.push({pass: 'B', batch, chunk: chunks - 1 - chunk});
      }
    }
  }
  const orders = Array.from({length: stageCount}, (_, stage) => {
    const distance = stageCount - stage - 1;
    const warmup = Math.min(forward.length, chunks === 1 ? distance : 2 * distance + groupSize);
    const order = forward.slice(0, warmup);
    forward.slice(warmup).forEach((task, index) => order.push(task, backward[index]));
    return order.concat(backward.slice(forward.length - warmup));
  });
  const rows = orders.map(() => []);
  const complete = {F: new Set(), B: new Set()};
  const key = (virtualStage, batch) => `${virtualStage}/${batch}`;
  const lastStage = stageCount * chunks - 1;
  while (orders.some(order => order.length)) {
    const ready = orders.map(([task], stage) => {
      if (!task) return null;
      const virtual = task.chunk * stageCount + stage;
      const {batch, pass} = task;
      const allowed = pass === 'F'
        ? virtual === 0 || complete.F.has(key(virtual - 1, batch))
        : complete.F.has(key(virtual, batch)) &&
          (virtual === lastStage || complete.B.has(key(virtual + 1, batch)));
      return allowed ? task : null;
    });
    if (!ready.some(Boolean)) throw new Error('Schedule stalled before all tasks finished');
    ready.forEach((task, stage) => {
      rows[stage].push(task);
      if (task) {
        orders[stage].shift();
        complete[task.pass].add(key(task.chunk * stageCount + stage, task.batch));
      }
    });
  }
  return rows;
}
