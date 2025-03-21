import { Request, Response, NextFunction } from "express";
import createHttpError from "http-errors";

export const getServerStatus = async (req: Request, res: Response, next: NextFunction) => {
    try {
        res.status(200).json({ message: "Server ON" });
    } catch (error) {
        next(error);
    }
};

export const getMlE1 = async (req: Request, res: Response, next: NextFunction) => {};
