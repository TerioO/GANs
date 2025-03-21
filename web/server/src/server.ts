import { app } from "./app";
import env from "./config/env";

const server = app();

console.log(env);

server.listen(env.PORT, () => {
    console.log(`Server running on PORT: ${env.PORT}`);
});
