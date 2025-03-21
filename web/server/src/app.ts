import express from "express";
import cors from "cors";
import { corsOptions } from "./config/cors";
import { errorHandler } from "./middleware/erorrHandler";
import apiRouter from "./routers/apiR";

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
        if (req.accepts("html")) res.status(404).send(`<h1>404 - NOT FOUND</h1>`);
        else if (req.accepts("text/plain")) res.status(404).send("404 - NOT FOUND");
        else if (req.accepts("application/json")) res.status(404).json({ message: "404 - NOT FOUND" });
    });

    // Error Handler:
    app.use(errorHandler);

    return app;
};
