import { Request, Response, NextFunction } from "express";
import { evalOnnxModelCGAN, evalOnnxModelGAN } from "../helpers/evalONNXmodels";
import { IOnnxCganRequest, IOnnxGanRequest } from "../types/global-types";
import { onnxCganNames, onnxGanNames } from "../types/constant-types";
import createHttpError from "http-errors";

export const runGAN = async (
    req: Request<unknown, unknown, IOnnxGanRequest["payload"], unknown>,
    res: Response,
    next: NextFunction
) => {
    try {
        const batchSize = req.body.batchSize;
        if (!batchSize) throw createHttpError(400, "batchSize required");
        if (batchSize < 1 || batchSize > 64)
            throw createHttpError(400, "batchSize must be in this interval: [1,64]");

        const modelName = req.body.modelName;
        if(!onnxGanNames.includes(modelName)) throw createHttpError(400, "Invalid modelName");

        const data = await evalOnnxModelGAN(modelName, batchSize);
        
        res.status(200).json(data);
    } catch (error) {
        if (createHttpError.isHttpError(error)) next(error);
        else next(createHttpError(500, "Failed to run model"));
    }
};

export const runCGAN = async (
    req: Request<unknown, unknown, IOnnxCganRequest["payload"], unknown>,
    res: Response,
    next: NextFunction
) => {
    try {
        const batchSize = req.body.batchSize;
        if (!batchSize) throw createHttpError(400, "batchSize required");
        if (batchSize < 1 || batchSize > 64)
            throw createHttpError(400, "batchSize must be in this interval: [1,64]");

        const modelName = req.body.modelName;
        if(!onnxCganNames.includes(modelName)) throw createHttpError(400, "Invalid modelName");

        const label = req.body.label;
        if(typeof label !== "number") throw createHttpError(400, "label must be a number");

        const data = await evalOnnxModelCGAN(modelName, batchSize, label);
        
        res.status(200).json(data);
    } catch (error) {
        console.log(error);
        if (createHttpError.isHttpError(error)) next(error);
        else next(createHttpError(500, "Failed to run model"));

    }
};