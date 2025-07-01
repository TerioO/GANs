import { Request, Response, NextFunction } from "express";
import { evalOnnxModelCGAN, evalOnnxModelGAN } from "../helpers/evalONNXmodels";
import { IOnnxCganRequest, IOnnxGanRequest } from "../types/global-types";
import { onnxCganModels, onnxCganNames, onnxGanModels, onnxGanNames } from "../types/constant-types";
import createHttpError from "http-errors";
import env from "../config/env";

export const runGAN = async (
  req: Request<unknown, unknown, IOnnxGanRequest["payload"], unknown>,
  res: Response,
  next: NextFunction
) => {
  try {
    const modelName = req.body.modelName;
    if (!onnxGanNames.includes(modelName)) throw createHttpError(400, "Invalid modelName");

    const batchSize = req.body.batchSize;
    if (!batchSize) throw createHttpError(400, "batchSize required");

    const maxBatchSize = env.NODE_ENV === "dev" 
      ? onnxGanModels[modelName].maxBatchSize.dev 
      : env.NODE_ENV === "prod" 
        ? onnxGanModels[modelName].maxBatchSize.prod 
        : 32;
    if (batchSize < 1 || batchSize > maxBatchSize)
      throw createHttpError(400, `batchSize must be in this interval: [1,${maxBatchSize}]`);

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
    const modelName = req.body.modelName;
    if (!onnxCganNames.includes(modelName)) throw createHttpError(400, "Invalid modelName");

    const batchSize = req.body.batchSize;
    if (!batchSize) throw createHttpError(400, "batchSize required");

    const maxBatchSize = env.NODE_ENV === "dev" 
      ? onnxCganModels[modelName].maxBatchSize.dev 
      : env.NODE_ENV === "prod" 
        ? onnxCganModels[modelName].maxBatchSize.prod 
        : 32;
    if (batchSize < 1 || batchSize > maxBatchSize)
      throw createHttpError(400, `batchSize must be in this interval: [1,${maxBatchSize}]`);

    const label = req.body.label;
    if (typeof label !== "number") throw createHttpError(400, "label must be a number");

    const data = await evalOnnxModelCGAN(modelName, batchSize, label);

    res.status(200).json(data);
  } catch (error) {
    if (createHttpError.isHttpError(error)) next(error);
    else next(createHttpError(500, "Failed to run model"));
  }
};