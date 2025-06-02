# Environment

**.env.dev** and **.env.prod**

- `PORT`: number
- `WHITELIST`: string (`;` separated strings, E.g: origin1;origin2;origin3;...;originN)

**cross-env**

- `NODE_ENV`: "dev" | "prod"

If deploying the server use node arg [--max-old-space-size](https://nodejs.org/api/cli.html#--max-old-space-sizesize-in-mib) to set max RAM usage so that app doesn't crash.

```
node --max-old-space-size=SIZE_MiB
```