import { Request, Response, NextFunction } from "express";
import { evalOnnxModel } from "../helpers/evalONNXmodels";
import { IOnnxRequest } from "../types/global-types";
import { onnxModelNames } from "../types/constant-types";
import createHttpError from "http-errors";

export const runGAN = async (
    req: Request<unknown, unknown, IOnnxRequest["payload"], unknown>,
    res: Response,
    next: NextFunction
) => {
    try {
        const batchSize = parseInt(req.body.batchSize);
        if (!batchSize) throw createHttpError(400, "batchSize must be a number");
        if (batchSize < 1 || batchSize > 64)
            throw createHttpError(400, "batchSize must be in this interval: [1,64]");

        const modelName = req.body.modelName;
        if(!onnxModelNames.includes(modelName)) throw createHttpError(400, "Invalid modelName");

        const data = await evalOnnxModel(modelName, batchSize);

        res.status(200).json(data);
    } catch (error) {
        if (createHttpError.isHttpError(error)) next(error);
        else next(createHttpError(500, "Failed to run model"));
    }
};
