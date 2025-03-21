import { Request, Response, NextFunction } from "express";
import createHttpError from "http-errors";
import { IMsgResponse } from "../types/global-types";

export const getServerStatus = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const response: IMsgResponse = { message: "Server ON" }; 
        res.status(200).json(response);
    } catch (error) {
        next(error);
    }
};

export const getMlE1 = async (req: Request, res: Response, next: NextFunction) => {};
