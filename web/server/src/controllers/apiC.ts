import { Request, Response, NextFunction } from "express";
import type { IMsgResponse } from "../types/global-types";
import createHttpError from "http-errors";

export const getServerStatus = async (req: Request, res: Response, next: NextFunction) => {
    try {
        setTimeout(() => {
            const response: IMsgResponse = { message: "Server ON" };
            res.status(200).json(response);
        }, 1000);
    } catch (error) {
        next(error);
    }
};
