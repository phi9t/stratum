export function inspectSchedule(rows, chunks, trace) {
  const inventory = rows.map(tasks => {
    let live = 0;
    return [0, ...tasks.map(task => {
      if (task) live += task.pass === 'F' ? 1 : -1;
      return live;
    })];
  });
  const messages = [];
  if (trace) {
    const stages = rows.length;
    const intervals = Array.from({length: stages * chunks}, () => ({}));
    rows.forEach((tasks, stage) => tasks.forEach((task, slot) => {
      if (task?.batch === trace) intervals[task.chunk * stages + stage][task.pass] = slot;
    }));
    for (let v = 0; v < intervals.length - 1; v++) {
      messages.push(
        {boundary:v, from:v, to:v+1, pass:'F', ready:(intervals[v].F+1)/chunks, use:intervals[v+1].F/chunks},
        {boundary:v, from:v+1, to:v, pass:'B', ready:(intervals[v+1].B+1)/chunks, use:intervals[v].B/chunks}
      );
    }
    messages.sort((a,b)=>a.ready-b.ready);
  }
  return {inventory,messages};
}
