import { CorsOptions } from "cors";
import env from "./env";

const whitelist = env.WHITELIST;

export const corsOptions: CorsOptions = {
    origin: (origin, callback) => {
        if (env.NODE_ENV === "dev") {
            if ((origin && whitelist.includes(origin)) || !origin) callback(null, true);
            else callback(new Error("Not allowed by CORS"));
        } else if (env.NODE_ENV === "prod") {
            if (origin && whitelist.includes(origin)) callback(null, true);
            else callback(new Error("Not allowed by CORS"));
        }
    },
    optionsSuccessStatus: 204,
    credentials: true
};
