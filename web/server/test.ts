const input = ['cats', 'dogs'];

const out = Object.fromEntries(input.map((el, i) => [i, el]));
console.log(out);