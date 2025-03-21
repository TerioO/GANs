import { Request, Response, NextFunction } from "express";
import createHttpError from "http-errors";
import type { IApiError } from "../types/global-types";

export const errorHandler = (error: unknown, req: Request, res: Response, next: NextFunction) => {
    let statusCode = 500;
    const apiError: IApiError = {
        isError: true,
        message: "Unknown Server Error"
    };
    if (error instanceof Error) {
        apiError.message = error.message;
    } else if (createHttpError.isHttpError(error)) {
        statusCode = error.statusCode;
        apiError.message = error.message;
    }
    res.status(statusCode).json(apiError);
    console.log(error);
};
