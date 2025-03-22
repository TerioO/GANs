import { Request, Response, NextFunction } from "express";
import type { IMsgResponse } from "../types/global-types";
import createHttpError from "http-errors";

export const getServerStatus = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const response: IMsgResponse = { message: "Server ON" };
        res.status(200).json(response);
    } catch (error) {
        next(error);
    }
};
