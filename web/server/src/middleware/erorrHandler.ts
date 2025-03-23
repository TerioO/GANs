import { Request, Response, NextFunction } from "express";
import createHttpError from "http-errors";
import type { IApiError } from "../types/global-types";
import env from "../config/env";

export const errorHandler = (
    error: unknown,
    req: Request,
    res: Response,
    next: NextFunction
) => {
    let statusCode = 500;
    const apiError: IApiError = {
        isError: true,
        message: "Unknown Server Error"
    };
    if (createHttpError.isHttpError(error)) {
        statusCode = error.statusCode;
        apiError.message = error.message;
    } else if (error instanceof Error) {
        if (env.NODE_ENV == "dev") console.log(error);
    }
    res.status(statusCode).json(apiError);
    if (env.NODE_ENV == "dev") console.log(error);
};
