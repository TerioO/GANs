import express from "express";
import cors from "cors";
import { corsOptions } from "./config/cors";
import { errorHandler } from "./middleware/erorrHandler";
import apiRouter from "./routers/apiR";
import { IApiError } from "./types/global-types";

export const app = () => {
    const app = express();

    // Middleware:
    app.use(cors(corsOptions));
    app.use(express.json());
    app.use(express.urlencoded({ extended: true }));

    // Routers:
    app.use(apiRouter);

    // 404 Route:
    app.use((req, res, next) => {
        const contentType = req.headers["content-type"];

        if (contentType?.includes("text/html")) {
            res.status(404).send(`<h1>404 - NOT FOUND</h1>`);
        } else if (contentType?.includes("text/plain")) {
            res.status(404).send("404 - NOT FOUND");
        } else if (contentType?.includes("application/json")) {
            const response: IApiError = {
                isError: true,
                message: "404 - NOT FOUND"
            };
            res.status(404).json(response);
        }
    });

    // Error Handler:
    app.use(errorHandler);

    return app;
};
